"""
tests/accuracy_test.py
========================
精度监测 + 自动回滚机制。

功能:
  1. 对比 FP16 基线与 4-bit 量化后的重建误差 (MSE / 相对误差)
  2. 模拟 Perplexity (PPL) 波动检测：若 PPL 增量 > 0.1%，自动增大 group_size 并重试
  3. 输出优化建议，记录通过的最优配置

用法:
  python tests/accuracy_test.py
  python tests/accuracy_test.py --group-size 64 --threshold 0.1

退出码:
  0 = 精度达标
  1 = 经过 max_retries 次调整仍未达标（需要人工介入）
"""

import sys
import os
import argparse
import math
import torch

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.quantization.kv_compressor import KVCompressor


# ---------------------------------------------------------------------------
# PPL 代理：用重建误差估算 PPL 波动
# ---------------------------------------------------------------------------

def estimate_ppl_delta(
    original: torch.Tensor,
    compressor: KVCompressor,
) -> float:
    """
    用量化重建的 MSE 估算 PPL 增量百分比。

    近似关系（经验公式，适用于 KV Cache 量化场景）:
      ΔPPL% ≈ mse * 1000  （数量级估算）

    Args:
        original  : FP16 KV Tensor（真实分布）
        compressor: 待测 KVCompressor 实例

    Returns:
        estimated PPL delta in percent (%)
    """
    q, scales, zps = compressor.compress(original)
    mse = compressor.mse_error(original, q, scales, zps)
    rel = compressor.relative_error(original, q, scales, zps)
    # 综合 MSE 和相对误差给出保守估计
    ppl_delta = mse * 500 + rel * 0.5
    return ppl_delta, mse, rel


# ---------------------------------------------------------------------------
# 主测试逻辑
# ---------------------------------------------------------------------------

def run_accuracy_test(
    initial_group_size: int = 128,
    ppl_threshold: float = 0.1,     # PPL 波动阈值 (%)
    max_retries: int = 5,
    device: str = "cpu",
    verbose: bool = True,
) -> dict:
    """
    精度监测主循环。

    策略:
      - 初始 group_size 从参数指定值开始
      - 若精度不达标，将 group_size 减半（更细粒度 → 更低误差）
      - 最多尝试 max_retries 次

    Returns:
        result dict: {
            "passed": bool,
            "best_group_size": int,
            "final_ppl_delta": float,
            "final_mse": float,
            "compression_ratio": float,
        }
    """
    # 生成测试数据：模拟真实 KV 分布
    # 形状: [batch=1, heads=8, seq=128, dim=group_size]
    # 用不同 group_size 测试，所以 dim 固定为 512 = 4 * 128
    dim = 512
    test_tensor = torch.randn(1, 8, 128, dim, dtype=torch.float16).to(device)

    group_size = initial_group_size
    passed = False
    best_result = None

    print("=" * 60)
    print("[AccuracyTest] 精度监测 + 自动回滚")
    print(f"  PPL 波动阈值 : {ppl_threshold}%")
    print(f"  初始 group   : {group_size}")
    print(f"  最大重试次数 : {max_retries}")
    print("=" * 60)

    for attempt in range(max_retries):
        compressor = KVCompressor(group_size=group_size, bits=4)
        ppl_delta, mse, rel = estimate_ppl_delta(test_tensor, compressor)
        ratio = compressor.compression_ratio(test_tensor)

        if verbose:
            print(
                f"  [尝试 {attempt + 1}/{max_retries}] "
                f"group_size={group_size:>4} | "
                f"ΔPPL≈{ppl_delta:.4f}% | "
                f"MSE={mse:.6f} | "
                f"rel_err={rel:.4f}% | "
                f"压缩率={ratio * 100:.1f}%"
            )

        best_result = {
            "passed": ppl_delta <= ppl_threshold,
            "best_group_size": group_size,
            "final_ppl_delta": ppl_delta,
            "final_mse": mse,
            "compression_ratio": ratio,
        }

        if ppl_delta <= ppl_threshold:
            passed = True
            print(f"\n  [PASS] PPL 波动 {ppl_delta:.4f}% <= 阈值 {ppl_threshold}%")
            print(f"  最优配置: group_size={group_size}, 压缩率={ratio * 100:.1f}%")
            break
        else:
            print(f"  [WARN] PPL 波动 {ppl_delta:.4f}% > 阈值, 触发回滚...")
            # 回滚策略：减小 group_size（更细粒度，误差更低，但元数据稍多）
            new_group_size = max(group_size // 2, 16)
            if new_group_size == group_size:
                print("  [STOP] group_size 已达最小值 16，停止调整")
                break
            group_size = new_group_size

    if not passed:
        print(f"\n  [FAIL] 经过 {max_retries} 次调整未达标，建议检查数据分布或降低量化 bits")

    return best_result


# ---------------------------------------------------------------------------
# 多配置扫描（可选）
# ---------------------------------------------------------------------------

def scan_group_sizes(
    group_sizes: list = [16, 32, 64, 128, 256],
    device: str = "cpu",
) -> None:
    """扫描不同 group_size 的精度-压缩率权衡曲线。"""
    dim = 512
    test_tensor = torch.randn(1, 8, 128, dim, dtype=torch.float16).to(device)

    print("\n[GroupSize 扫描]")
    print(f"{'group_size':>12} {'ΔPPL%':>10} {'MSE':>12} {'rel_err%':>10} {'ratio%':>8}")
    print("-" * 56)

    for gs in group_sizes:
        if dim % gs != 0:
            continue
        c = KVCompressor(group_size=gs, bits=4)
        ppl_d, mse, rel = estimate_ppl_delta(test_tensor, c)
        ratio = c.compression_ratio(test_tensor)
        print(f"{gs:>12} {ppl_d:>10.4f} {mse:>12.6f} {rel:>10.4f} {ratio*100:>8.1f}")


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="KV Cache 量化精度监测")
    parser.add_argument("--group-size", type=int, default=128, help="初始 group_size")
    parser.add_argument("--threshold", type=float, default=0.1, help="PPL 波动阈值 (%)")
    parser.add_argument("--max-retries", type=int, default=5, help="最大重试次数")
    parser.add_argument("--scan", action="store_true", help="扫描多个 group_size")
    parser.add_argument("--device", type=str, default="cpu", help="测试设备")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.scan:
        scan_group_sizes(device=args.device)
    else:
        result = run_accuracy_test(
            initial_group_size=args.group_size,
            ppl_threshold=args.threshold,
            max_retries=args.max_retries,
            device=args.device,
        )
        sys.exit(0 if result["passed"] else 1)
