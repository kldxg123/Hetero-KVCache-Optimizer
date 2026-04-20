import sys
import os
import torch
import gc

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
from src.memory.manager import HeteroKVManager


def debug_step_by_step_rigorous():
    device = "cuda:3"  # 必须明确设备
    print("\n" + "=" * 80)
    print("🔍 严谨版诊断：Hetero-KV 物理显存精算对账")
    print("=" * 80)

    manager = HeteroKVManager(hbm_max_blocks=150, block_size=16, device=device)
    sink, local = 32, 2048

    k_cache = torch.randn(1, 64, 100, 128, dtype=torch.bfloat16, device=device)
    test_steps = [1000, 5000, 10000, 20000]

    for step in test_steps:
        new_kv = torch.randn(1, 64, step, 128, dtype=torch.bfloat16, device=device)

        # 核心驱逐逻辑
        full_k = torch.cat([k_cache, new_kv], dim=-2)
        if full_k.shape[-2] > (sink + local):
            k_cache = torch.cat([full_k[..., :sink, :], full_k[..., -local:, :]], dim=-2)
        else:
            k_cache = full_k

        del full_k, new_kv
        gc.collect()
        torch.cuda.empty_cache()

        # 🔥 修正：必须明确查询 device="cuda:3"
        actual_mem_mb = torch.cuda.memory_allocated(device=device) / 1024 ** 2

        # 理论值计算 (bfloat16 占 2 Bytes)
        theoretical_mb = (1 * 64 * k_cache.shape[-2] * 128 * 2) / 1024 ** 2

        print(f"📈 步长 {step:<5} Tokens | 物理长度: {k_cache.shape[-2]:<5} | "
              f"理论应占: {theoretical_mb:.2f} MB | 实际占用: {actual_mem_mb:.2f} MB")


if __name__ == "__main__":
    debug_step_by_step_rigorous()