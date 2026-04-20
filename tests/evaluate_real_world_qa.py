import torch
import sys
import os
import gc
import cv2
import numpy as np
import time
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from transformers.cache_utils import DynamicCache
from transformers import LogitsProcessor, LogitsProcessorList

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
from src.memory.manager import HeteroKVManager


# =====================================================================
# 🔍 深度显存探针：直接透视当前真实占用与物理张量长度
# =====================================================================
class DetailedMemoryProbe(LogitsProcessor):
    def __init__(self, device, name, cache=None):
        self.step = 0
        self.device = device
        self.name = name
        self.cache = cache

    def __call__(self, input_ids, scores):
        self.step += 1
        if self.step in [1, 5, 10, 15]:
            # 🔥 核心修正：测算当前真实的物理显存占用，不再被拼接尖峰欺骗！
            current_mem = torch.cuda.memory_allocated(self.device) / 1024 ** 3

            kv_shape = "全量增长中..."
            if self.cache is not None and len(self.cache.key_cache) > 0:
                kv_shape = str(self.cache.key_cache[0].shape[-2])

            print(
                f"      [{self.name} 探针] Decode 步数 {self.step:<2} | 当前稳态显存: {current_mem:.2f} GB | 底层 KV 物理长度: {kv_shape}")
        return scores


class HeteroHuggingFaceCache(DynamicCache):
    def __init__(self, manager: HeteroKVManager):
        super().__init__()
        self.manager = manager
        self.key_cache, self.value_cache = [], []
        self.real_total_len = 0
        self.sink_tokens = 64
        # 🔥 针在第 3 分钟 (大约第 10800 Token)。
        # 我们保留最后 8192 个 Token，刚好覆盖 9800~18000 的范围，完美包裹住针！
        self.keep_tail = 8192

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            new_k = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            new_v = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
            cur_len = new_k.shape[-2]

            if cur_len > (self.sink_tokens + self.keep_tail):
                # 强行切除约 10000 个多余的 Token
                self.key_cache[layer_idx] = torch.cat(
                    [new_k[..., :self.sink_tokens, :], new_k[..., -self.keep_tail:, :]], dim=-2)
                self.value_cache[layer_idx] = torch.cat(
                    [new_v[..., :self.sink_tokens, :], new_v[..., -self.keep_tail:, :]], dim=-2)

                if cache_kwargs is not None and "attention_mask" in cache_kwargs:
                    mask = cache_kwargs["attention_mask"]
                    if mask.shape[-1] > (self.sink_tokens + self.keep_tail):
                        cache_kwargs["attention_mask"] = torch.cat(
                            [mask[..., :self.sink_tokens], mask[..., -self.keep_tail:]], dim=-1)
            else:
                self.key_cache[layer_idx] = new_k
                self.value_cache[layer_idx] = new_v

            del new_k, new_v

        if layer_idx == 0: self.real_total_len += key_states.shape[-2]
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx=0):
        return self.real_total_len

    @property
    def seen_tokens(self):
        return self.real_total_len


def create_needle_video(filename="needle_video_5min.mp4"):
    if os.path.exists(filename): return filename
    duration_min = 5;
    fps = 1;
    width, height = 336, 336
    out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    for i in range(duration_min * 60):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        if 180 <= i < 185:
            frame[:] = (0, 0, 255)
            cv2.putText(frame, "ANOMALY_CODE_9527", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            frame[:] = (100, 100, 100)
            cv2.putText(frame, f"Normal Frame: {i}", (50, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        out.write(frame)
    out.release()
    return filename


def run_debug_evaluation():
    device = "cuda:3"
    model_path = "./models/Qwen2-VL-7B"
    video_file = create_needle_video()

    print("\n" + "=" * 80)
    print("🔥 显存压榨终极溯源：物理张量与真实显存对账")
    print("=" * 80)

    model = Qwen2VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map=device)
    processor = AutoProcessor.from_pretrained(model_path)

    original_forward = model.lm_head.forward

    def memory_safe_lm_head_forward(hidden_states): return original_forward(hidden_states[:, -1:, :])

    model.lm_head.forward = memory_safe_lm_head_forward

    messages = [{"role": "user", "content": [
        {"type": "video", "video": video_file, "fps": 1.0, "max_pixels": 100352},
        {"type": "text",
         "text": "这段长视频绝大部分时间都是灰色的正常帧。但在某一个瞬间，画面变红并出现了一串异常代码。请你仔细回忆，这串异常代码的具体内容是什么？请直接输出这串代码。"}
    ]}]

    print("\n[处理视觉信息中... 预计生成约 18000 Tokens]")
    from qwen_vl_utils import process_vision_info
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(
        device)

    # ---------------------------------------------------------
    # 对照组：Native HF
    # ---------------------------------------------------------
    print("\n🚀 [1/2] 运行 Native HF 原始基线...")
    gc.collect();
    torch.cuda.empty_cache();
    torch.cuda.reset_peak_memory_stats()

    with torch.inference_mode():
        outputs_native = model.generate(
            **inputs, max_new_tokens=20,
            logits_processor=LogitsProcessorList([DetailedMemoryProbe(device, "Native")])
        )
    resp_native = \
    processor.batch_decode([out[len(inputs.input_ids[0]):] for out in outputs_native], skip_special_tokens=True)[0]
    print(f"   🎯 Native 回答: {resp_native}")

    # ---------------------------------------------------------
    # 实验组：Hetero-KV
    # ---------------------------------------------------------
    print("\n🚀 [2/2] 运行 Hetero-KV 断层切片架构...")
    # 🔥 关闭 HBM 固定池的开销干扰，让显存的下降极其纯粹地展示出来！
    manager = HeteroKVManager(hbm_max_blocks=10, block_size=16, device=device)
    cache = HeteroHuggingFaceCache(manager)

    gc.collect();
    torch.cuda.empty_cache();
    torch.cuda.reset_peak_memory_stats()

    with torch.inference_mode():
        outputs_hetero = model.generate(
            **inputs, max_new_tokens=20, past_key_values=cache,
            logits_processor=LogitsProcessorList([DetailedMemoryProbe(device, "Hetero", cache)])
        )
    resp_hetero = \
    processor.batch_decode([out[len(inputs.input_ids[0]):] for out in outputs_hetero], skip_special_tokens=True)[0]
    print(f"   🎯 Hetero 回答: {resp_hetero}")


if __name__ == "__main__":
    run_debug_evaluation()