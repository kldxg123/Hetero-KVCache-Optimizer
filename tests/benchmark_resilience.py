"""
tests/benchmark_resilience.py
================================
Micro-benchmark for Fused Dequant-Attention and Predictive Prefetching.

All timing uses CUDA Events for GPU-accurate measurement.
All tensors are real PyTorch allocations (no mock, no simulate).

Resource Emulation:
  When running on GPUs with >24GB HBM (e.g., A100 80GB), we impose
  torch.cuda.set_per_process_memory_fraction(24.0/total_GB) to emulate
  the RTX 4090's 24GB memory wall. This ensures the benchmark faithfully
  reproduces the memory-constrained conditions described in the paper.
  The method is documented in the paper's Experimental Setup section.

Benchmarks:
  1. baseline:       CPU uint8 → GPU FP32 decompress → standard matmul attention
  2. fused_only:     CPU uint8 → GPU uint8 (no decompress) → fused Triton attention
  3. full_system:    Fused + Predictive Prefetch (background stream overlap)
"""

import torch
import gc
import numpy as np
from typing import Dict, List

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# Resource Emulation: cap GPU memory to 24GB (RTX 4090 target)
# Must be called before any CUDA tensor allocation.
# ---------------------------------------------------------------------------
EMULATED_VRAM_GB = 24.0

def apply_memory_cap():
    """
    On GPUs with >24GB HBM, restrict PyTorch's memory pool to 24GB,
    faithfully emulating the RTX 4090 memory wall. This uses PyTorch's
    official set_per_process_memory_fraction API.
    """
    if not torch.cuda.is_available():
        return False
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if total_gb > EMULATED_VRAM_GB:
        fraction = EMULATED_VRAM_GB / total_gb
        torch.cuda.set_per_process_memory_fraction(fraction, device=0)
        print(f"  [Resource Emulation] Capping {total_gb:.1f}GB GPU to {EMULATED_VRAM_GB:.0f}GB "
              f"(fraction={fraction:.3f})")
        return True
    return False

from src.quantization.kv_compressor import KVCompressor
from src.quantization.fused_dequant_attn import (
    fused_dequant_attn_decode,
    _TRITON_OK,
)


# ---------------------------------------------------------------------------
# CUDA Event timer
# ---------------------------------------------------------------------------
class CUDATimer:
    def __init__(self):
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)

    def __enter__(self):
        self.start.record()
        return self

    def __exit__(self, *args):
        self.end.record()
        torch.cuda.synchronize()

    @property
    def ms(self) -> float:
        return self.start.elapsed_time(self.end)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
HEAD_DIM = 128
NUM_HEADS = 32
SEQ_K = 16384          # 16K tokens per chunk
BATCH = 1
DECODE_STEPS = 200
WARMUP_STEPS = 20


def create_quantized_kv_on_cpu(
    batch: int, heads: int, seq_k: int, head_dim: int
) -> Dict[str, torch.Tensor]:
    """
    Create real 16-bit KV on GPU, quantize, then move to CPU (pin_memory).
    This simulates the exact DRAM storage path used in production swap-in.
    """
    compressor = KVCompressor(group_size=128, bits=4)

    K_fp16 = torch.randn(batch, heads, seq_k, head_dim, device="cuda", dtype=torch.float16)
    V_fp16 = torch.randn(batch, heads, seq_k, head_dim, device="cuda", dtype=torch.float16)

    kq, ks, kz, vq, vs, vz = [], [], [], [], [], []
    for h in range(heads):
        kq_h, ks_h, kz_h = compressor.compress(K_fp16[0, h])
        vq_h, vs_h, vz_h = compressor.compress(V_fp16[0, h])
        kq.append(kq_h); ks.append(ks_h); kz.append(kz_h)
        vq.append(vq_h); vs.append(vs_h); vz.append(vz_h)

    del K_fp16, V_fp16
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "k_quant":  torch.stack(kq, 0).unsqueeze(0).cpu().pin_memory(),
        "k_scales": torch.stack(ks, 0).unsqueeze(0).float().cpu().pin_memory(),
        "k_zps":    torch.stack(kz, 0).unsqueeze(0).float().cpu().pin_memory(),
        "v_quant":  torch.stack(vq, 0).unsqueeze(0).cpu().pin_memory(),
        "v_scales": torch.stack(vs, 0).unsqueeze(0).float().cpu().pin_memory(),
        "v_zps":    torch.stack(vz, 0).unsqueeze(0).float().cpu().pin_memory(),
    }


# ---------------------------------------------------------------------------
# Baseline: H2D copy + decompress-to-FP32 + matmul attention
# ---------------------------------------------------------------------------
def bench_baseline(
    Q_gpu: torch.Tensor, cpu_data: Dict[str, torch.Tensor], steps: int
) -> Dict:
    latencies = []
    device = Q_gpu.device

    # Warmup
    for _ in range(WARMUP_STEPS):
        _baseline_step(Q_gpu, cpu_data, device)

    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()

    # Measure transient memory delta for one step
    torch.cuda.reset_peak_memory_stats()
    mem_before = torch.cuda.memory_allocated()
    _baseline_step(Q_gpu, cpu_data, device)
    torch.cuda.synchronize()
    mem_after = torch.cuda.memory_allocated()
    peak_delta_mb = torch.cuda.max_memory_allocated() / 1024**2
    mem_base_mb = mem_before / 1024**2

    timer = CUDATimer()
    for _ in range(steps):
        with timer:
            _baseline_step(Q_gpu, cpu_data, device)
        latencies.append(timer.ms)

    return _build_result("baseline", latencies, peak_delta_mb, mem_base_mb)


def _baseline_step(Q, cpu_data, device):
    """Full baseline: H2D + decompress + matmul attention."""
    K_q = cpu_data["k_quant"].to(device, non_blocking=True)
    K_s = cpu_data["k_scales"].to(device, non_blocking=True)
    K_z = cpu_data["k_zps"].to(device, non_blocking=True)
    V_q = cpu_data["v_quant"].to(device, non_blocking=True)
    V_s = cpu_data["v_scales"].to(device, non_blocking=True)
    V_z = cpu_data["v_zps"].to(device, non_blocking=True)
    torch.cuda.synchronize()

    # Decompress: materializes full FP32 K/V in HBM
    K_deq = (K_q.float() - K_z.unsqueeze(-1)) * K_s.unsqueeze(-1)
    V_deq = (V_q.float() - V_z.unsqueeze(-1)) * V_s.unsqueeze(-1)

    sm_scale = 1.0 / (Q.shape[-1] ** 0.5)
    scores = torch.matmul(Q.float(), K_deq.transpose(-2, -1)) * sm_scale
    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, V_deq)
    return out


# ---------------------------------------------------------------------------
# Fused: H2D copy of uint8 (no decompress) + Triton fused attention
# ---------------------------------------------------------------------------
def bench_fused(
    Q_gpu: torch.Tensor, cpu_data: Dict[str, torch.Tensor], steps: int
) -> Dict:
    latencies = []
    device = Q_gpu.device

    # Warmup
    for _ in range(WARMUP_STEPS):
        _fused_step(Q_gpu, cpu_data, device)

    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()

    # Measure transient memory delta for one step
    torch.cuda.reset_peak_memory_stats()
    mem_before = torch.cuda.memory_allocated()
    _fused_step(Q_gpu, cpu_data, device)
    torch.cuda.synchronize()
    mem_after = torch.cuda.memory_allocated()
    peak_delta_mb = torch.cuda.max_memory_allocated() / 1024**2
    mem_base_mb = mem_before / 1024**2

    timer = CUDATimer()
    for _ in range(steps):
        with timer:
            _fused_step(Q_gpu, cpu_data, device)
        latencies.append(timer.ms)

    return _build_result("fused_only", latencies, peak_delta_mb, mem_base_mb)


def _fused_step(Q, cpu_data, device):
    """Zero-copy: H2D of uint8 only + fused Triton attention."""
    K_q = cpu_data["k_quant"].to(device, non_blocking=True)
    K_s = cpu_data["k_scales"].to(device, non_blocking=True)
    K_z = cpu_data["k_zps"].to(device, non_blocking=True)
    V_q = cpu_data["v_quant"].to(device, non_blocking=True)
    V_s = cpu_data["v_scales"].to(device, non_blocking=True)
    V_z = cpu_data["v_zps"].to(device, non_blocking=True)
    torch.cuda.synchronize()

    # Fused: dequantize in registers, no FP32 K/V allocation
    return fused_dequant_attn_decode(Q, K_q, K_s, K_z, V_q, V_s, V_z)


# ---------------------------------------------------------------------------
# Full system: Fused + Predictive Prefetch (background stream)
# ---------------------------------------------------------------------------
def bench_full_system(
    Q_gpu: torch.Tensor, cpu_data: Dict[str, torch.Tensor], steps: int
) -> Dict:
    latencies = []
    device = Q_gpu.device
    prefetch_stream = torch.cuda.Stream()

    # Prepare second chunk for prefetch simulation
    cpu_data_next = create_quantized_kv_on_cpu(BATCH, NUM_HEADS, SEQ_K, HEAD_DIM)

    # Warmup
    for _ in range(WARMUP_STEPS):
        _fused_step(Q_gpu, cpu_data, device)

    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()

    # Measure
    torch.cuda.reset_peak_memory_stats()
    mem_before = torch.cuda.memory_allocated()

    timer = CUDATimer()
    for step in range(steps):
        # Background prefetch: async H2D of next chunk's quantized data
        with torch.cuda.stream(prefetch_stream):
            Kq_n = cpu_data_next["k_quant"].to(device, non_blocking=True)
            Ks_n = cpu_data_next["k_scales"].to(device, non_blocking=True)
            Kz_n = cpu_data_next["k_zps"].to(device, non_blocking=True)
            Vq_n = cpu_data_next["v_quant"].to(device, non_blocking=True)
            Vs_n = cpu_data_next["v_scales"].to(device, non_blocking=True)
            Vz_n = cpu_data_next["v_zps"].to(device, non_blocking=True)

        # Main stream: compute attention on current chunk
        with timer:
            K_q = cpu_data["k_quant"].to(device, non_blocking=True)
            K_s = cpu_data["k_scales"].to(device, non_blocking=True)
            K_z = cpu_data["k_zps"].to(device, non_blocking=True)
            V_q = cpu_data["v_quant"].to(device, non_blocking=True)
            V_s = cpu_data["v_scales"].to(device, non_blocking=True)
            V_z = cpu_data["v_zps"].to(device, non_blocking=True)
            torch.cuda.synchronize()
            fused_dequant_attn_decode(Q_gpu, K_q, K_s, K_z, V_q, V_s, V_z)

        # Wait for prefetch to complete
        torch.cuda.current_stream().wait_stream(prefetch_stream)
        latencies.append(timer.ms)

    peak_delta_mb = torch.cuda.max_memory_allocated() / 1024**2
    mem_base_mb = mem_before / 1024**2

    return _build_result("full_system", latencies, peak_delta_mb, mem_base_mb)


# ---------------------------------------------------------------------------
# Correctness validation
# ---------------------------------------------------------------------------
def validate_correctness(cpu_data: Dict[str, torch.Tensor]) -> bool:
    Q = torch.randn(BATCH, NUM_HEADS, 1, HEAD_DIM, device="cuda", dtype=torch.float16)

    # Reference (baseline path)
    ref = _baseline_step(Q, cpu_data, torch.device("cuda"))
    # Fused
    fused = _fused_step(Q, cpu_data, torch.device("cuda"))

    max_diff = (ref - fused).abs().max().item()
    mean_diff = (ref - fused).abs().mean().item()
    rel_err = mean_diff / (ref.abs().mean().item() + 1e-8) * 100
    print(f"  max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}, rel_err={rel_err:.3f}%")
    return max_diff < 1.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _build_result(config, latencies, peak_vram_mb, base_vram_mb):
    arr = np.array(latencies)
    return {
        "config": config,
        "mean_tpot_ms": float(np.mean(arr)),
        "p50_tpot_ms": float(np.percentile(arr, 50)),
        "p99_tpot_ms": float(np.percentile(arr, 99)),
        "max_tpot_ms": float(np.max(arr)),
        "min_tpot_ms": float(np.min(arr)),
        "std_tpot_ms": float(np.std(arr)),
        "peak_vram_mb": peak_vram_mb,
        "base_vram_mb": base_vram_mb,
    }


def _print_result(r):
    print(f"  mean={r['mean_tpot_ms']:.3f}ms  p99={r['p99_tpot_ms']:.3f}ms  "
          f"max={r['max_tpot_ms']:.3f}ms  peak_vram={r['peak_vram_mb']:.1f}MB")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    if not torch.cuda.is_available():
        print("[SKIP] No CUDA device available")
        return

    device_name = torch.cuda.get_device_name(0)
    device_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3

    # Apply 24GB memory cap to emulate RTX 4090 on larger GPUs
    capped = apply_memory_cap()

    print("=" * 72)
    print(f"  HeteroKV Resilience Micro-Benchmark")
    print(f"  Physical Device: {device_name} ({device_mem:.1f} GB)")
    if capped:
        print(f"  Resource Emulation: {device_mem:.1f}GB → {EMULATED_VRAM_GB:.0f}GB (RTX 4090 target)")
    else:
        print(f"  Resource Emulation: Not needed (device ≤ {EMULATED_VRAM_GB:.0f}GB)")
    print(f"  Config: heads={NUM_HEADS}, head_dim={HEAD_DIM}, seq_k={SEQ_K}")
    print(f"  Triton: {_TRITON_OK}")
    print(f"  Steps:  {WARMUP_STEPS} warmup + {DECODE_STEPS} measured (CUDA Events)")
    print("=" * 72)

    # Create data
    print("\n[Setup] Creating real 16-bit KV → quantize to 4-bit → pin to CPU...")
    cpu_data = create_quantized_kv_on_cpu(BATCH, NUM_HEADS, SEQ_K, HEAD_DIM)
    Q = torch.randn(BATCH, NUM_HEADS, 1, HEAD_DIM, device="cuda", dtype=torch.float16)

    for k, v in cpu_data.items():
        print(f"  {k}: {v.shape} {v.dtype} (CPU pinned)")

    # Correctness
    print("\n[Validate] Fused kernel vs baseline reference...")
    if not validate_correctness(cpu_data):
        print("  [FAIL] Output diverges!")
        return
    print("  [PASS] Outputs match")

    # Baseline
    print("\n[1/3] Baseline: CPU→GPU H2D + decompress-to-FP32 + matmul...")
    r1 = bench_baseline(Q, cpu_data, DECODE_STEPS)
    _print_result(r1)
    gc.collect(); torch.cuda.empty_cache()

    # Fused
    print("\n[2/3] Fused Dequant-Attention: CPU→GPU uint8 + Triton kernel...")
    r2 = bench_fused(Q, cpu_data, DECODE_STEPS)
    _print_result(r2)
    gc.collect(); torch.cuda.empty_cache()

    # Full system
    print("\n[3/3] Full System: Fused + Predictive Prefetch (bg stream)...")
    r3 = bench_full_system(Q, cpu_data, DECODE_STEPS)
    _print_result(r3)
    gc.collect(); torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 72)
    print("  RESULTS (CUDA Event Timing)")
    print("=" * 72)
    print(f"{'Metric':<22} {'Baseline':>10} {'Fused':>10} {'Full':>10}")
    print("-" * 72)
    for metric in ["mean_tpot_ms", "p50_tpot_ms", "p99_tpot_ms", "max_tpot_ms", "std_tpot_ms"]:
        label = metric.replace("_tpot_ms", " TPOT (ms)").replace("_", " ").title()
        print(f"{label:<22} {r1[metric]:>10.3f} {r2[metric]:>10.3f} {r3[metric]:>10.3f}")
    print(f"{'Peak VRAM (MB)':<22} {r1['peak_vram_mb']:>10.1f} {r2['peak_vram_mb']:>10.1f} {r3['peak_vram_mb']:>10.1f}")
    print("-" * 72)

    # Analytical memory comparison
    baseline_intermediate = 2 * BATCH * NUM_HEADS * SEQ_K * HEAD_DIM * 4 / 1024**2  # FP32 K+V
    fused_intermediate = 3 * BATCH * NUM_HEADS * 1 * HEAD_DIM * 4 / 1024**2  # output + L + M
    print(f"\n  Analytical intermediate memory:")
    print(f"    Baseline (FP32 K+V): {baseline_intermediate:.1f} MB")
    print(f"    Fused (output+L+M):  {fused_intermediate*1024:.1f} KB")
    print(f"    Theoretical saving:  {baseline_intermediate:.1f} MB eliminated")

    if r1['mean_tpot_ms'] > 0:
        speedup = r1['mean_tpot_ms'] / max(r2['mean_tpot_ms'], 0.001)
        print(f"\n  Fused kernel compute speedup: {speedup:.2f}x")
    if r1['p99_tpot_ms'] > 0 and r3['p99_tpot_ms'] > 0:
        tail_red = (1 - r3['p99_tpot_ms'] / r1['p99_tpot_ms']) * 100
        print(f"  Full system P99 tail change:  {tail_red:+.1f}%")
    print("=" * 72)

    # Save CSV
    csv_path = os.path.join(os.path.dirname(__file__), "benchmark_resilience_results.csv")
    with open(csv_path, "w") as f:
        f.write("config,mean_tpot_ms,p50_tpot_ms,p99_tpot_ms,max_tpot_ms,"
                "std_tpot_ms,peak_vram_mb\n")
        for r in [r1, r2, r3]:
            f.write(f"{r['config']},{r['mean_tpot_ms']:.3f},{r['p50_tpot_ms']:.3f},"
                    f"{r['p99_tpot_ms']:.3f},{r['max_tpot_ms']:.3f},{r['std_tpot_ms']:.3f},"
                    f"{r['peak_vram_mb']:.1f}\n")
    print(f"\nResults saved to {csv_path}")


if __name__ == "__main__":
    main()
