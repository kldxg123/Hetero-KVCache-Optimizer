#!/usr/bin/env python3
"""
Quick Memory Test for HeteroKV - Simplified version
Tests O(1) memory behavior without full model loading
"""

import torch
import sys
import time
from typing import Dict

sys.path.insert(0, '/home/app-ahr/Hetero-KVCache-Optimizer/src')

from memory.manager import HeteroKVManager

print("=" * 70)
print("HeteroKV Memory Test - O(1) Behavior Verification")
print("=" * 70)

# Initialize manager with three-zone architecture
manager = HeteroKVManager(
    num_layers=32,
    sink_tokens=64,
    hbm_budget_tokens=8192,  # Total budget for Sink + Tail
    device='cuda',
    enable_quant=True,
)

print(f"\nConfiguration:")
print(f"  - Sink: {manager.sink_tokens} tokens")
print(f"  - Tail budget: {manager.hbm_budget_tokens - manager.sink_tokens} tokens")
print(f"  - HeavyHitter budget: {manager._heavyhitter_budget} tokens")
print(f"  - Total HBM: {manager.max_hbm_tokens()} tokens = O(1)")

print(f"\n{'Context Pairs':<15} {'Input Tokens':<15} {'Peak Mem (MB)':<15} {'Status':<10}")
print("-" * 60)

# Test different context lengths
test_cases = [1000, 2000, 4000, 8000, 16000, 32000]
results = []

layer_idx = 0
batch_size, num_heads, head_dim = 1, 32, 128

for num_pairs in test_cases:
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    try:
        # Simulate prefill: large batch of tokens at once
        num_tokens = num_pairs * 10  # Rough estimation
        key_states = torch.randn(batch_size, num_heads, num_tokens, head_dim, device='cuda', dtype=torch.float16)
        value_states = torch.randn(batch_size, num_heads, num_tokens, head_dim, device='cuda', dtype=torch.float16)

        # Simulate prefill update (sink + tail extraction, body to DRAM)
        manager.update(layer_idx, key_states, value_states, mode='prefill')

        # Get peak memory
        peak_mem = torch.cuda.max_memory_allocated() / 1024**2

        # Check memory limit (24GB)
        if peak_mem > 24 * 1024:
            print(f"{num_pairs:<15} {num_tokens:<15} {peak_mem:<15.1f} {'LIMIT EXCEEDED':<10}")
            results.append({
                'pairs': num_pairs,
                'tokens': num_tokens,
                'peak_mb': peak_mem,
                'status': 'LIMIT_EXCEEDED'
            })
            break

        status = "OK" if peak_mem < 20 * 1024 else "HIGH"
        print(f"{num_pairs:<15} {num_tokens:<15} {peak_mem:<15.1f} {status:<10}")

        results.append({
            'pairs': num_pairs,
            'tokens': num_tokens,
            'peak_mb': peak_mem,
            'status': status
        })

        del key_states, value_states

    except RuntimeError as e:
        if "out of memory" in str(e):
            peak_mem = torch.cuda.max_memory_allocated() / 1024**2
            print(f"{num_pairs:<15} {num_tokens:<15} {peak_mem:<15.1f} {'OOM':<10}")
            results.append({
                'pairs': num_pairs,
                'tokens': num_tokens,
                'peak_mb': peak_mem,
                'status': 'OOM'
            })
            break
        else:
            print(f"ERROR: {e}")
            break

print("-" * 60)

# Analyze results
if len(results) >= 2:
    first_mem = results[0]['peak_mb']
    last_mem = results[-1]['peak_mb']
    mem_growth = last_mem - first_mem
    growth_pct = (mem_growth / first_mem) * 100 if first_mem > 0 else 0

    print(f"\nMemory Growth Analysis:")
    print(f"  - First test ({results[0]['pairs']} pairs): {first_mem:.1f} MB")
    print(f"  - Last test ({results[-1]['pairs']} pairs): {last_mem:.1f} MB")
    print(f"  - Growth: {mem_growth:.1f} MB ({growth_pct:.1f}%)")

    if growth_pct < 10:  # Less than 10% growth = O(1)
        print(f"  ✓ SUCCESS: Memory growth is minimal ({growth_pct:.1f}%) ≈ O(1) behavior!")
    elif growth_pct < 50:
        print(f"  ⚠ MODERATE: Memory growth is ({growth_pct:.1f}%) - sublinear but not ideal O(1)")
    else:
        print(f"  ✗ FAIL: Memory growth is too high ({growth_pct:.1f}%) - not O(1)")

    # Check if any test exceeded 24GB limit
    limit_tests = [r for r in results if r['status'] in ['LIMIT_EXCEEDED', 'OOM']]
    if limit_tests:
        print(f"\n⚠ WARNING: {len(limit_tests)} test(s) exceeded 24GB limit")
    else:
        print(f"\n✓ All tests stayed within 24GB limit")

print("=" * 70)
print("Test completed. Results reflect HeteroKV memory efficiency.")
print("=" * 70)
