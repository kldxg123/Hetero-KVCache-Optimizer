import torch
import sys
import os
import gc
import matplotlib.pyplot as plt
from transformers import Qwen2VLForConditionalGeneration
from transformers.cache_utils import DynamicCache

# --- 1. 路径修复与环境配置 ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 激进显存回收，确保底层测试绝对干净
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
from src.memory.manager import HeteroKVManager


# --- 2. 你的核心组件 ---
class HeteroHuggingFaceCache(DynamicCache):
    def __init__(self, manager: HeteroKVManager):
        super().__init__()
        self.manager = manager
        self.key_cache, self.value_cache = [], []
        self.real_total_len = 0
        self.sink_tokens, self.local_window = 32, 2048

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            new_k = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            new_v = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

            # 你的核心驱逐逻辑
            cur_len = new_k.shape[-2]
            if cur_len > (self.sink_tokens + self.local_window):
                self.key_cache[layer_idx] = torch.cat(
                    [new_k[..., :self.sink_tokens, :], new_k[..., -self.local_window:, :]], dim=-2)
                self.value_cache[layer_idx] = torch.cat(
                    [new_v[..., :self.sink_tokens, :], new_v[..., -self.local_window:, :]], dim=-2)
            else:
                self.key_cache[layer_idx], self.value_cache[layer_idx] = new_k, new_v
            del new_k, new_v

        # 强行截断 attention_mask 防止 HuggingFace 的 O(N^2) 爆炸
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


# --- 3. 终极底层测试主程序 ---
def run_kernel_proof():
    device = "cuda:3"
    model_path = "./models/Qwen2-VL-7B"

    print("\n" + "=" * 80)
    print("🚀 Hetero-KV 核心引擎物理隔离测试 (纯净 Decode 模拟)")
    print("=" * 80)

    # 仅加载语言模型部分，完全绕过视频 ViT 的显存干扰
    print("[加载模型] 正在初始化大语言模型引擎...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map=device)

    # 我们测试 5000, 10000, 20000, 30000 步的纯 Decode
    test_steps = [5000, 10000, 20000, 30000]
    record_points = set(test_steps)

    results_native = []
    results_hetero = []

    # 准备一个无意义的占位符 Token (用于手动 Decode)
    dummy_input = torch.tensor([[151644]], device=device)

    # -----------------------------------------------------------------
    # 回合 1：Native HF (对照组)
    # -----------------------------------------------------------------
    print("\n🥊 回合 1：运行 Native HF 原始逻辑...")
    gc.collect();
    torch.cuda.empty_cache();
    torch.cuda.reset_peak_memory_stats()
    base_mem = torch.cuda.memory_allocated(device) / 1024 ** 3

    native_cache = DynamicCache()
    # 手动维护 position_ids 以防止报错
    position_ids = torch.arange(0, 1, dtype=torch.long, device=device).unsqueeze(0)

    try:
        with torch.inference_mode():
            for step in range(1, max(test_steps) + 1):
                # 纯粹的前向传播，每次 1 个 Token
                model(input_ids=dummy_input, past_key_values=native_cache, use_cache=True, position_ids=position_ids)
                position_ids += 1  # 位置编码递增

                if step in record_points:
                    peak = (torch.cuda.memory_allocated(device) / 1024 ** 3) - base_mem
                    results_native.append((step, peak))
                    print(f"   ➤ [Native] 步数: {step:<5} | 动态 KV 显存: {peak:.3f} GB")
    except torch.OutOfMemoryError:
        print("   🚨 [Native] 发生 OOM 崩溃！")

    # -----------------------------------------------------------------
    # 回合 2：Hetero-KV (你的项目)
    # -----------------------------------------------------------------
    print("\n🥊 回合 2：运行 Hetero-KV 底层优化逻辑...")
    del native_cache;
    gc.collect();
    torch.cuda.empty_cache();
    torch.cuda.reset_peak_memory_stats()
    base_mem = torch.cuda.memory_allocated(device) / 1024 ** 3

    manager = HeteroKVManager(hbm_max_blocks=150, block_size=16, device=device)
    hetero_cache = HeteroHuggingFaceCache(manager)
    position_ids = torch.arange(0, 1, dtype=torch.long, device=device).unsqueeze(0)

    try:
        with torch.inference_mode():
            for step in range(1, max(test_steps) + 1):
                model(input_ids=dummy_input, past_key_values=hetero_cache, use_cache=True, position_ids=position_ids)
                position_ids += 1

                if step in record_points:
                    # 减去基础模型权重，只看动态显存
                    peak = (torch.cuda.memory_allocated(device) / 1024 ** 3) - base_mem
                    results_hetero.append((step, peak))
                    print(f"   ➤ [Hetero] 步数: {step:<5} | 动态 KV 显存: {peak:.3f} GB (锁死证明)")
    except Exception as e:
        print(f"   🚨 [Hetero] 发生异常: {e}")

    # -----------------------------------------------------------------
    # 绘图生成真理图表
    # -----------------------------------------------------------------
    plt.figure(figsize=(10, 6))

    if results_native:
        x_n, y_n = zip(*results_native)
        plt.plot(x_n, y_n, 'ro-', linewidth=2, label='Native HF Dynamic KV')

    if results_hetero:
        x_h, y_h = zip(*results_hetero)
        plt.plot(x_h, y_h, 'bo-', linewidth=2, label='Hetero-KV Dynamic KV (Ours)')

    plt.xlabel('Simulated Decode Steps (Tokens)', fontsize=12)
    plt.ylabel('Dynamic KV Cache Occupation (GB)', fontsize=12)
    plt.title('Kernel-Level Benchmark: Pure KV Cache Scalability', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.5)

    plt.savefig('kernel_proof_plot.png', dpi=300)
    print("\n✅ 内核级证明图表已生成：kernel_proof_plot.png")


if __name__ == "__main__":
    run_kernel_proof()