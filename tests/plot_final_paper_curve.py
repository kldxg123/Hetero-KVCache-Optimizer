import torch
import sys
import os
import gc
import cv2
import numpy as np
import matplotlib.pyplot as plt
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from transformers.cache_utils import DynamicCache
from transformers import LogitsProcessor, LogitsProcessorList

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
from src.memory.manager import HeteroKVManager


class SteadyStateMemoryProbe(LogitsProcessor):
    def __init__(self, device):
        self.step = 0
        self.device = device
        self.steady_mem = 0.0

    def __call__(self, input_ids, scores):
        self.step += 1
        # 抓取第 5 步的真实物理稳态显存
        if self.step == 5:
            self.steady_mem = torch.cuda.memory_allocated(self.device) / 1024 ** 3
        return scores


class HeteroHuggingFaceCache(DynamicCache):
    def __init__(self, manager: HeteroKVManager):
        super().__init__()
        self.manager = manager
        self.key_cache, self.value_cache = [], []
        self.real_total_len = 0
        self.sink_tokens = 64
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
                self.key_cache[layer_idx] = torch.cat(
                    [new_k[..., :self.sink_tokens, :], new_k[..., -self.keep_tail:, :]], dim=-2)
                self.value_cache[layer_idx] = torch.cat(
                    [new_v[..., :self.sink_tokens, :], new_v[..., -self.keep_tail:, :]], dim=-2)
                if cache_kwargs is not None and "attention_mask" in cache_kwargs:
                    mask = cache_kwargs["attention_mask"]
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


def create_dummy_video(duration_min, filename):
    if os.path.exists(filename): return filename
    fps = 1;
    width, height = 336, 336
    out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    for _ in range(duration_min * 60):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = (100, 100, 100)
        out.write(frame)
    out.release()
    return filename


def run_paper_evaluation():
    device = "cuda:3"
    model_path = "./models/Qwen2-VL-7B"
    test_durations = [1, 2, 4, 8]  # 分钟

    print("\n" + "=" * 80)
    print("🔥 毕业论文级评测：O(1) 常数级稳态显存验证")
    print("=" * 80)

    model = Qwen2VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map=device)
    processor = AutoProcessor.from_pretrained(model_path)

    original_forward = model.lm_head.forward

    def memory_safe_lm_head_forward(hidden_states): return original_forward(hidden_states[:, -1:, :])

    model.lm_head.forward = memory_safe_lm_head_forward

    results_native = []
    results_hetero = []

    for d in test_durations:
        f = create_dummy_video(d, f"dummy_{d}min.mp4")
        messages = [{"role": "user", "content": [{"type": "video", "video": f, "fps": 1.0, "max_pixels": 100352},
                                                 {"type": "text", "text": "总结"}]}]
        from qwen_vl_utils import process_vision_info
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        img_in, vid_in = process_vision_info(messages)
        inputs = processor(text=[text], images=img_in, videos=vid_in, padding=True, return_tensors="pt").to(device)

        # --- Native ---
        print(f"\n🚀 测试 {d} 分钟 - Native HF")
        gc.collect();
        torch.cuda.empty_cache()
        probe_native = SteadyStateMemoryProbe(device)
        with torch.inference_mode():
            model.generate(**inputs, max_new_tokens=10, logits_processor=LogitsProcessorList([probe_native]))
        print(f"   📊 Native 稳态显存: {probe_native.steady_mem:.2f} GB")
        results_native.append(probe_native.steady_mem)

        # --- Hetero ---
        print(f"🚀 测试 {d} 分钟 - Hetero-KV")
        manager = HeteroKVManager(hbm_max_blocks=10, block_size=16, device=device)
        cache = HeteroHuggingFaceCache(manager)
        gc.collect();
        torch.cuda.empty_cache()
        probe_hetero = SteadyStateMemoryProbe(device)
        with torch.inference_mode():
            model.generate(**inputs, max_new_tokens=10, past_key_values=cache,
                           logits_processor=LogitsProcessorList([probe_hetero]))
        print(f"   📊 Hetero 稳态显存: {probe_hetero.steady_mem:.2f} GB")
        results_hetero.append(probe_hetero.steady_mem)

        del inputs, cache;
        gc.collect();
        torch.cuda.empty_cache()

    # --- 画图 ---
    plt.figure(figsize=(10, 6))
    plt.plot(test_durations, results_native, 'ro-', linewidth=2, markersize=8, label='Native HF (O(N) Growth)')
    plt.plot(test_durations, results_hetero, 'bo-', linewidth=2, markersize=8, label='Hetero-KV (O(1) Constant)')

    plt.xlabel('Video Context Duration (Minutes)', fontsize=12)
    plt.ylabel('Steady-State Allocated Memory (GB)', fontsize=12)
    plt.title('Memory Wall Breakthrough: Hetero-KV vs Native HF', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig('paper_ready_curve.png', dpi=300, bbox_inches='tight')
    print("\n✅ 论文高清配图已生成：paper_ready_curve.png")


if __name__ == "__main__":
    run_paper_evaluation()