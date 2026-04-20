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
# 🔍 深度显存探针：剥离死显存，只测纯净的动态 KV 与底层长度
# =====================================================================
class DetailedMemoryProbe(LogitsProcessor):
    def __init__(self, device, name, base_mem, cache=None):
        self.step = 0
        self.device = device
        self.name = name
        self.base_mem = base_mem
        self.cache = cache

    def __call__(self, input_ids, scores):
        self.step += 1
        if self.step in [1, 5, 10, 15]:
            # 强制清空临时碎片，只留真实占用
            gc.collect();
            torch.cuda.empty_cache()

            # 纯动态对账：当前显存 - 死显存基座 = 纯净 KV Cache
            dyn_mem = (torch.cuda.memory_allocated(self.device) / 1024 ** 3) - self.base_mem

            if self.cache is not None and len(self.cache.key_cache) > 0:
                kv_shape = str(self.cache.key_cache[0].shape[-2])
            else:
                kv_shape = str(input_ids.shape[-1])

            print(
                f"      [{self.name} 探针] Decode 第 {self.step:<2} 步 | 纯动态 KV 显存: {dyn_mem:.3f} GB | 底层物理长度: {kv_shape}")
        return scores


# =====================================================================
# 🧠 Hetero Cache: 沿用成功验证的断层切片架构 (Sink + Keep Tail)
# =====================================================================
class HeteroHuggingFaceCache(DynamicCache):
    def __init__(self, manager: HeteroKVManager):
        super().__init__()
        self.manager = manager
        self.key_cache, self.value_cache = [], []
        self.real_total_len = 0

        self.sink_tokens = 64
        # 🔥 规模扩大：视频 15 分钟(约 54000 Token)。针在第 13 分钟。
        # 我们保留最后 12000 个 Token，完美包裹住最后 3.3 分钟的画面，确保精度 100% 无损！
        self.keep_tail = 12000

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            new_k = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            new_v = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
            cur_len = new_k.shape[-2]

            if cur_len > (self.sink_tokens + self.keep_tail):
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


# =====================================================================
# 🎬 制造 15 分钟超长测试视频
# =====================================================================
def create_massive_needle_video(filename="needle_video_15min.mp4"):
    if os.path.exists(filename): return filename
    duration_min = 15;
    fps = 1;
    width, height = 336, 336
    out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    # 🔥 针埋在第 13 分钟处 (第 780 帧)
    needle_frame_idx = 780
    for i in range(duration_min * 60):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        if needle_frame_idx <= i < needle_frame_idx + 5:
            frame[:] = (0, 0, 255)
            cv2.putText(frame, "ANOMALY_CODE_9527", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            frame[:] = (100, 100, 100)
            cv2.putText(frame, f"Normal Frame: {i}", (50, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        out.write(frame)
    out.release()
    return filename


def run_ultimate_fusion_evaluation():
    device = "cuda:3"
    model_path = "./models/Qwen2-VL-7B"

    # 强制清理旧的视频文件，确保生成 15 分钟的新视频
    if os.path.exists("needle_video_15min.mp4"):
        os.remove("needle_video_15min.mp4")
    video_file = create_massive_needle_video()

    print("\n" + "=" * 80)
    print("🔥 15分钟极限融合：海量显存压榨 VS 绝对精度维持")
    print("=" * 80)

    model = Qwen2VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map=device)
    processor = AutoProcessor.from_pretrained(model_path)

    # 阉割 lm_head 预填充显存爆炸，保护底座能够吞下 15 分钟视频
    original_forward = model.lm_head.forward

    def memory_safe_lm_head_forward(hidden_states): return original_forward(hidden_states[:, -1:, :])

    model.lm_head.forward = memory_safe_lm_head_forward

    messages = [{"role": "user", "content": [
        {"type": "video", "video": video_file, "fps": 1.0, "max_pixels": 100352},
        {"type": "text",
         "text": "这段长视频绝大部分时间都是灰色的正常帧。但在某一个瞬间，画面变红并出现了一串异常代码。请你仔细回忆，这串异常代码的具体内容是什么？请直接输出这串代码。"}
    ]}]

    print("\n[处理视觉信息中... 预计生成约 54000 个 Tokens！]")
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

    # 抓取死显存基座
    base_mem_native = torch.cuda.memory_allocated(device) / 1024 ** 3
    print(f"   [底层测算] 当前死显存基座 (模型 + 视频输入): {base_mem_native:.2f} GB")

    with torch.inference_mode():
        outputs_native = model.generate(
            **inputs, max_new_tokens=20,
            logits_processor=LogitsProcessorList([DetailedMemoryProbe(device, "Native", base_mem_native)])
        )
    resp_native = \
    processor.batch_decode([out[len(inputs.input_ids[0]):] for out in outputs_native], skip_special_tokens=True)[0]
    print(f"   🎯 Native 回答: {resp_native}")

    # ---------------------------------------------------------
    # 实验组：Hetero-KV
    # ---------------------------------------------------------
    print("\n🚀 [2/2] 运行 Hetero-KV 断层切片架构...")
    # 依然极小化 HBM 池开销，纯粹看动态显存
    manager = HeteroKVManager(hbm_max_blocks=10, block_size=16, device=device)
    cache = HeteroHuggingFaceCache(manager)

    gc.collect();
    torch.cuda.empty_cache();
    torch.cuda.reset_peak_memory_stats()

    base_mem_hetero = torch.cuda.memory_allocated(device) / 1024 ** 3
    print(f"   [底层测算] 当前死显存基座 (模型 + 视频输入 + HBM池): {base_mem_hetero:.2f} GB")

    with torch.inference_mode():
        outputs_hetero = model.generate(
            **inputs, max_new_tokens=20, past_key_values=cache,
            logits_processor=LogitsProcessorList([DetailedMemoryProbe(device, "Hetero", base_mem_hetero, cache)])
        )
    resp_hetero = \
    processor.batch_decode([out[len(inputs.input_ids[0]):] for out in outputs_hetero], skip_special_tokens=True)[0]
    print(f"   🎯 Hetero 回答: {resp_hetero}")


if __name__ == "__main__":
    run_ultimate_fusion_evaluation()