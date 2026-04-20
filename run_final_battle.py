import argparse
import subprocess
import sys
import os
import gc
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt


# =====================================================================
# 子进程执行器 (Worker) - 保证绝对的物理内存隔离
# =====================================================================
def run_worker(method, duration):
    # 路径与环境配置
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path: sys.path.insert(0, project_root)
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

    from src.memory.manager import HeteroKVManager
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    from transformers.cache_utils import DynamicCache
    from transformers import LogitsProcessor, LogitsProcessorList
    from qwen_vl_utils import process_vision_info

    device = "cuda:3"
    model_path = "./models/Qwen2-VL-7B"
    video_file = f"v_iso_{duration}.mp4"

    # 生成物理视频 (如果不存在)
    if not os.path.exists(video_file):
        out = cv2.VideoWriter(video_file, cv2.VideoWriter_fourcc(*'mp4v'), 1, (336, 336))
        for _ in range(duration * 60): out.write(np.zeros((336, 336, 3), dtype=np.uint8))
        out.release()

    # 加载模型并进行 lm_head 物理切割
    model = Qwen2VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map=device)
    processor = AutoProcessor.from_pretrained(model_path)

    original_forward = model.lm_head.forward

    def memory_safe_lm_head_forward(hidden_states):
        return original_forward(hidden_states[:, -1:, :])

    model.lm_head.forward = memory_safe_lm_head_forward

    # 探针
    class MemoryResetProbe(LogitsProcessor):
        def __init__(self): self.step = 0

        def __call__(self, input_ids, scores):
            self.step += 1
            if self.step == 1:
                gc.collect();
                torch.cuda.empty_cache();
                torch.cuda.reset_peak_memory_stats()
            return scores

    # Hetero Cache
    class HeteroHuggingFaceCache(DynamicCache):
        def __init__(self, manager):
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
                if new_k.shape[-2] > (self.sink_tokens + self.local_window):
                    self.key_cache[layer_idx] = torch.cat(
                        [new_k[..., :self.sink_tokens, :], new_k[..., -self.local_window:, :]], dim=-2)
                    self.value_cache[layer_idx] = torch.cat(
                        [new_v[..., :self.sink_tokens, :], new_v[..., -self.local_window:, :]], dim=-2)
                else:
                    self.key_cache[layer_idx], self.value_cache[layer_idx] = new_k, new_v
                del new_k, new_v
            if layer_idx == 0: self.real_total_len += key_states.shape[-2]
            return self.key_cache[layer_idx], self.value_cache[layer_idx]

        def get_seq_length(self, layer_idx=0):
            return self.real_total_len

        @property
        def seen_tokens(self):
            return self.real_total_len

    # 开始测试
    messages = [{"role": "user", "content": [{"type": "video", "video": video_file, "fps": 1.0, "max_pixels": 100352},
                                             {"type": "text", "text": "summary"}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=process_vision_info(messages)[0], videos=process_vision_info(messages)[1],
                       padding=True, return_tensors="pt").to(device)

    try:
        if method == "hetero":
            manager = HeteroKVManager(hbm_max_blocks=150, block_size=16, device=device)
            cache = HeteroHuggingFaceCache(manager)
            with torch.inference_mode():
                model.generate(**inputs, max_new_tokens=5, past_key_values=cache,
                               logits_processor=LogitsProcessorList([MemoryResetProbe()]))
        else:
            with torch.inference_mode():
                model.generate(**inputs, max_new_tokens=5, logits_processor=LogitsProcessorList([MemoryResetProbe()]))

        peak = torch.cuda.max_memory_allocated(device) / 1024 ** 3
        # 约定输出格式供主进程解析
        print(f"RESULT_PEAK:{peak:.4f}")

    except torch.OutOfMemoryError:
        print("RESULT_OOM:OOM")


# =====================================================================
# 主进程调度器 (Master) - 负责发号施令与绘图
# =====================================================================
def run_master():
    test_durations = [1, 2, 4, 8, 16]
    methods = ["hetero", "native"]
    results = {m: [] for m in methods}

    print("\n" + "=" * 80 + "\n🔥 终极隔离测试：多进程科学对账版\n" + "=" * 80)

    for m in methods:
        print(f"\n🚀 开始独立测试序列: {m.upper()}")
        for d in test_durations:
            print(f"   ⏳ 正在子进程中运行 {m.upper()} - {d} min ... ", end="", flush=True)
            # 启动独立子进程，保证物理隔离
            cmd = [sys.executable, __file__, "--worker", "--method", m, "--duration", str(d)]
            try:
                # 捕获输出
                output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)

                # 解析输出结果
                if "RESULT_OOM:OOM" in output:
                    print("🚨 OOM")
                    results[m].append(float('inf'))
                else:
                    peak = None
                    for line in output.split('\n'):
                        if line.startswith("RESULT_PEAK:"):
                            peak = float(line.split(":")[1])
                    if peak is not None:
                        print(f"✅ {peak:.2f} GB")
                        results[m].append(peak)
                    else:
                        print("❌ 运行失败 (未获取到峰值)")
                        results[m].append(float('inf'))
            except subprocess.CalledProcessError as e:
                print(f"💥 进程崩溃")
                results[m].append(float('inf'))

    # 绘图：展示两条真实曲线
    plt.figure(figsize=(10, 6))

    for m, color, label in zip(methods, ['blue', 'red'], ['Hetero-KV (Ours)', 'Native HF']):
        valid_idx = [i for i, v in enumerate(results[m]) if v != float('inf')]
        x = [test_durations[i] for i in valid_idx]
        y = [results[m][i] for i in valid_idx]
        plt.plot(x, y, color=color, marker='o', linewidth=2, label=f"{label} (Measured)")

        # 线性拟合与外推
        if len(x) > 1:
            poly = np.polyfit(x, y, 1)
            trend_x = np.linspace(0, 24, 100)
            plt.plot(trend_x, np.poly1d(poly)(trend_x), color=color, linestyle='--', alpha=0.5,
                     label=f"{label} (Trend)")

    plt.axhline(y=24, color='darkred', linestyle=':', label='24GB GPU Limit')
    plt.axhline(y=40, color='orange', linestyle=':', label='40GB GPU Limit')

    plt.xlabel('Video Duration (min)');
    plt.ylabel('Peak GPU Memory (GB)')
    plt.title('Rigorous Isolation Benchmark: The Twin Memory Walls')
    plt.legend();
    plt.grid(True)
    plt.savefig('final_battle_isolated.png', dpi=300)
    print("\n✅ 独立测试完成，绝对科学客观的对比图已生成: final_battle_isolated.png")

    # 清理视频
    for d in test_durations:
        f = f"v_iso_{d}.mp4"
        if os.path.exists(f): os.remove(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--method", type=str, choices=["hetero", "native"])
    parser.add_argument("--duration", type=int)
    args = parser.parse_args()

    if args.worker:
        run_worker(args.method, args.duration)
    else:
        run_master()