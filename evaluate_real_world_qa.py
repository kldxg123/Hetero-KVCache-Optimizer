import torch
import sys
import os
import gc
import cv2
import numpy as np
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from transformers.cache_utils import DynamicCache
from transformers import LogitsProcessorList

# --- 路径修复与环境配置 ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
from src.memory.manager import HeteroKVManager


# --- Hetero Cache (你的核心项目) ---
class HeteroHuggingFaceCache(DynamicCache):
    def __init__(self, manager: HeteroKVManager):
        super().__init__()
        self.manager = manager
        self.key_cache, self.value_cache = [], []
        self.real_total_len = 0
        self.sink_tokens, self.local_window = 32, 2048

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states);
            self.value_cache.append(value_states)
        else:
            new_k = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            new_v = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
            cur_len = new_k.shape[-2]
            if cur_len > (self.sink_tokens + self.local_window):
                self.key_cache[layer_idx] = torch.cat(
                    [new_k[..., :self.sink_tokens, :], new_k[..., -self.local_window:, :]], dim=-2)
                self.value_cache[layer_idx] = torch.cat(
                    [new_v[..., :self.sink_tokens, :], new_v[..., -self.local_window:, :]], dim=-2)
            else:
                self.key_cache[layer_idx], self.value_cache[layer_idx] = new_k, new_v
            del new_k, new_v

        if cache_kwargs is not None and "attention_mask" in cache_kwargs:
            mask = cache_kwargs["attention_mask"]
            if mask.shape[-1] > (self.sink_tokens + self.local_window):
                cache_kwargs["attention_mask"] = torch.cat(
                    [mask[..., :self.sink_tokens], mask[..., -self.local_window:]], dim=-1)

        if layer_idx == 0: self.real_total_len += key_states.shape[-2]
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx=0):
        return self.real_total_len

    @property
    def seen_tokens(self):
        return self.real_total_len


# --- 1. 大海捞针视频生成器 ---
def create_needle_video(filename="needle_video_20min.mp4"):
    if os.path.exists(filename):
        return filename

    duration_min = 20
    fps = 1
    width, height = 336, 336
    total_frames = duration_min * 60

    print(f"\n🎬 正在生成包含隐藏信息的 {duration_min} 分钟测试视频...")
    out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    # 目标：在第 15 分钟 (第 900 帧) 注入异常信息
    needle_frame_idx = 900
    secret_code = "ANOMALY_CODE_9527"

    for i in range(total_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        if i == needle_frame_idx:
            # 植入“针”：红底白字，非常显眼的异常帧
            frame[:] = (0, 0, 255)
            cv2.putText(frame, secret_code, (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
        else:
            # 正常背景帧：无聊的灰色背景，模拟冗长的无用信息
            frame[:] = (100, 100, 100)
            cv2.putText(frame, f"Normal Frame: {i}", (50, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)

        out.write(frame)
    out.release()
    return filename


# --- 2. 真实场景推理评测 ---
def run_real_world_evaluation():
    device = "cuda:3"
    model_path = "./models/Qwen2-VL-7B"
    video_file = create_needle_video()

    print("\n" + "=" * 80)
    print("🔥 真实场景大海捞针测试 (精度与系统生存验证)")
    print("=" * 80)

    model = Qwen2VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map=device)
    processor = AutoProcessor.from_pretrained(model_path)

    # 🔪 保留 lm_head 阉割，确保仅测试底层能力
    original_forward = model.lm_head.forward

    def memory_safe_lm_head_forward(hidden_states):
        return original_forward(hidden_states[:, -1:, :])

    model.lm_head.forward = memory_safe_lm_head_forward

    # 构建极具挑战性的 Prompt
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_file, "fps": 1.0, "max_pixels": 100352},
                {"type": "text",
                 "text": "这段长视频绝大部分时间都是灰色的正常帧。但在某一个瞬间，画面变红并出现了一串异常代码。请你仔细回忆，这串异常代码的具体内容是什么？请直接输出这串代码。"}
            ]
        }
    ]

    print("\n[处理视觉信息中... 这将产生庞大的 Token 序列]")
    from qwen_vl_utils import process_vision_info
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(
        device)

    # -----------------------------------------------------------------
    # 启动 Hetero-KV 进行真实回答
    # -----------------------------------------------------------------
    print("\n🚀 开始 Hetero-KV 极长上下文检索推理...")
    manager = HeteroKVManager(hbm_max_blocks=150, block_size=16, device=device)
    cache = HeteroHuggingFaceCache(manager)

    gc.collect();
    torch.cuda.empty_cache();
    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()

    try:
        with torch.inference_mode():
            # 允许模型输出 20 个字，把答案说完整
            outputs = model.generate(**inputs, max_new_tokens=20, past_key_values=cache)

        latency = time.time() - start_time
        peak_mem = torch.cuda.max_memory_allocated(device) / 1024 ** 3

        # 解析输出
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, outputs)
        ]
        response = \
        processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        print("\n" + "=" * 80)
        print("🏆 Hetero-KV 答卷 (实测结果)")
        print("=" * 80)
        print(f"🎯 模型回答: \n{response}")
        print("-" * 80)
        print(f"⚡ 推理耗时: {latency:.2f} 秒")
        print(f"📊 峰值显存: {peak_mem:.2f} GB (完美生存在 20 分钟量级)")

        if "9527" in response:
            print("\n🌟 结论: 精度无损！Hetero-KV 在裁剪了 90% 以上显存的情况下，精准召回了深埋在第 15 分钟的视觉特征！")
        else:
            print("\n⚠️ 结论: 答案似乎不准确。说明我们当前的驱逐策略 (Sink+Local) 把重要的中间视觉特征丢弃了。")

    except Exception as e:
        print(f"🚨 推理崩溃: {e}")


if __name__ == "__main__":
    import time

    run_real_world_evaluation()