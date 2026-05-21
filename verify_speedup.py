#!/usr/bin/env python3
"""
verify_speedup.py
=================
Empirically verify the speedup claim for the _block_mean_kernel / Heavy Hitter Oracle.

Measurement methodology:
  - Python baseline: time.time() wall-clock (captures CPU-GPU sync overhead)
  - Triton kernel:   torch.cuda.Event (pure GPU-side timing)
  - Config: 128K context, block_size=1 (matching paper's ablation setup)
"""

import sys
import os
import time
import math
import torch

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.policy.heavy_hitter import HeavyHitterOracle


class LegacyPythonOracle:
    """Frozen pre-kernelization Python-loop implementation."""
    def __init__(self, block_size=16, sink_tokens=32, local_window=128):
        self.block_size = block_size
        self.sink_tokens = sink_tokens
        self.local_window = local_window
        self.token_scores = None

    def update(self, recent_attention):
        seq_len = recent_attention.shape[0]
        if self.token_scores is None or self.token_scores.shape[0] < seq_len:
            new_scores = torch.zeros(seq_len, dtype=torch.float32, device=recent_attention.device)
            if self.token_scores is not None:
                new_scores[:self.token_scores.shape[0]] = self.token_scores
            self.token_scores = new_scores
        self.token_scores[:seq_len] += recent_attention

    def get_eviction_candidates(self, current_seq_len, evict_num_blocks):
        if current_seq_len <= self.sink_tokens + self.local_window:
            return []
        num_blocks = math.ceil(current_seq_len / self.block_size)
        sink_blocks = math.ceil(self.sink_tokens / self.block_size)
        local_blocks = math.ceil(self.local_window / self.block_size)
        if self.token_scores is None:
            candidates = []
            for idx in range(sink_blocks, num_blocks - local_blocks):
                candidates.append(idx)
                if len(candidates) == evict_num_blocks:
                    break
            return candidates
        block_scores = torch.zeros(num_blocks, device=self.token_scores.device)
        for i in range(num_blocks):
            start_idx = i * self.block_size
            end_idx = min(start_idx + self.block_size, current_seq_len)
            block_scores[i] = self.token_scores[start_idx:end_idx].mean()
        safe_mask = torch.zeros(num_blocks, dtype=torch.bool, device=self.token_scores.device)
        safe_mask[:sink_blocks] = True
        safe_mask[max(0, num_blocks - local_blocks):] = True
        block_scores[safe_mask] = float('inf')
        sorted_indices = torch.argsort(block_scores)
        candidates = []
        for idx in sorted_indices:
            if block_scores[idx] == float('inf'):
                break
            candidates.append(idx.item())
            if len(candidates) == evict_num_blocks:
                break
        return candidates


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Paper's exact configuration: 128K context, block_size=1
    SEQ_LEN = 131_072
    BLOCK_SIZE = 1
    EVICT_BLOCKS = 16_384
    SINK = 64
    LOCAL = 8192

    print(f"\nConfig: seq_len={SEQ_LEN:,}  block_size={BLOCK_SIZE}  evict_blocks={EVICT_BLOCKS:,}")
    print(f"        sink={SINK}  local_window={LOCAL}")

    # Prepare synthetic attention scores
    scores = torch.rand(SEQ_LEN, device=device, dtype=torch.float32) * 0.5
    scores[10000:10100] = 0.001
    scores[30000:30100] = 0.002
    scores[50000:50100] = 0.003
    scores[70000:70100] = 0.004

    # -------------------------------------------------------
    # 1) Legacy Python baseline — time.time() wall-clock
    # -------------------------------------------------------
    legacy = LegacyPythonOracle(block_size=BLOCK_SIZE, sink_tokens=SINK, local_window=LOCAL)
    legacy.update(scores)

    print("\n--- Legacy Python Baseline (time.time) ---")
    # Single warmup to prime caches
    _ = legacy.get_eviction_candidates(SEQ_LEN, EVICT_BLOCKS)
    torch.cuda.synchronize(device)

    # Timed run
    t0 = time.time()
    legacy_result = legacy.get_eviction_candidates(SEQ_LEN, EVICT_BLOCKS)
    torch.cuda.synchronize(device)
    t1 = time.time()
    t_py_s = t1 - t0
    t_py_ms = t_py_s * 1000
    print(f"Python baseline: {t_py_s:.4f} s  ({t_py_ms:.1f} ms)")
    print(f"Eviction candidates (first 10): {legacy_result[:10]}")

    # -------------------------------------------------------
    # 2) Triton-fused path — torch.cuda.Event
    # -------------------------------------------------------
    modern = HeavyHitterOracle(block_size=BLOCK_SIZE, sink_tokens=SINK, local_window=LOCAL)
    modern.update(scores)

    print("\n--- Triton-Fused Oracle (cuda.Event) ---")
    # Warmup
    for _ in range(10):
        _ = modern.get_eviction_candidates(SEQ_LEN, EVICT_BLOCKS)
    torch.cuda.synchronize(device)

    # Timed runs (50 iterations for stability)
    NUM_REPEATS = 50
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    start_evt.record()
    for _ in range(NUM_REPEATS):
        modern_result = modern.get_eviction_candidates(SEQ_LEN, EVICT_BLOCKS)
    end_evt.record()
    torch.cuda.synchronize(device)
    t_triton_total_ms = start_evt.elapsed_time(end_evt)
    t_triton_ms = t_triton_total_ms / NUM_REPEATS
    print(f"Triton kernel ({NUM_REPEATS} iters): {t_triton_ms:.4f} ms per call")
    print(f"Eviction candidates (first 10): {modern_result[:10].tolist()}")

    # -------------------------------------------------------
    # 3) Correctness check
    # -------------------------------------------------------
    legacy_tensor = torch.tensor(legacy_result, dtype=torch.long, device=device)
    match = torch.equal(legacy_tensor, modern_result)
    print(f"\nCorrectness: {'PASS' if match else 'FAIL'}")

    # -------------------------------------------------------
    # 4) Compute speedup
    # -------------------------------------------------------
    speedup = t_py_ms / t_triton_ms
    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"  Python baseline:  {t_py_s:.4f} s  ({t_py_ms:.1f} ms)")
    print(f"  Triton kernel:    {t_triton_ms:.4f} ms")
    print(f"  Speedup:          {speedup:,.1f}x")
    print(f"{'='*60}")

    # The paper claims 22,333x (~66s -> sub-10ms)
    # Report if there's a discrepancy
    claimed_speedup = 22333
    if abs(speedup - claimed_speedup) / claimed_speedup > 0.05:
        print(f"\n[!] DISCREPANCY: measured {speedup:,.1f}x vs claimed {claimed_speedup:,}x")
        print(f"    Difference: {abs(speedup - claimed_speedup):,.0f}x")
    else:
        print(f"\n[OK] Measured speedup is within 5% of claimed value.")


if __name__ == "__main__":
    main()
