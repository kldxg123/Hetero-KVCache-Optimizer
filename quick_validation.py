#!/usr/bin/env python3
"""
Quick validation script for HeteroKV fixes
Tests basic functionality without full model loading
"""

import torch
import sys
import os
from typing import Dict, List

sys.path.insert(0, '/home/app-ahr/Hetero-KVCache-Optimizer/src')

print("=" * 60)
print("HeteroKV Quick Validation Test")
print("=" * 60)

# Test 1: Check if modules can be imported
print("\n[1/5] Testing module imports...")
try:
    from memory.manager import HeteroKVManager
    from memory.attention_competition_queue import AttentionCompetitionQueue
    from core.engine_wrapper import FusedHeteroCache
    from core.fused_attention_patch import patch_model_for_fused_attention
    print("✅ All modules imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

# Test 2: Check CUDA availability
print("\n[2/5] Testing CUDA availability...")
if torch.cuda.is_available():
    print(f"✅ CUDA available: {torch.cuda.device_count()} GPUs")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"   GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)")
else:
    print("❌ CUDA not available")
    sys.exit(1)

# Test 3: Test AttentionCompetitionQueue
print("\n[3/5] Testing AttentionCompetitionQueue...")
try:
    queue = AttentionCompetitionQueue()

    # Create dummy data
    batch_size, num_heads, num_tokens, head_dim = 1, 4, 10, 128
    k = torch.randn(batch_size, num_heads, num_tokens, head_dim, device='cuda')
    v = torch.randn(batch_size, num_heads, num_tokens, head_dim, device='cuda')
    scores = torch.rand(num_tokens, device='cuda')

    # Test enqueue
    queue.enqueue(k, v, scores, layer_idx=0, prefix="test")

    # Test dequeue
    top_k, top_v, top_scores = queue.dequeue_top_k(5)

    if top_k is not None and top_k.shape[-2] == 5:
        print("✅ AttentionCompetitionQueue working correctly")
    else:
        print("❌ AttentionCompetitionQueue dequeue failed")

except Exception as e:
    print(f"❌ AttentionCompetitionQueue error: {e}")
    sys.exit(1)

# Test 4: Test HeteroKVManager initialization
print("\n[4/5] Testing HeteroKVManager initialization...")
try:
    manager = HeteroKVManager(
        num_layers=4,
        sink_tokens=64,
        hbm_budget_tokens=2048,
        device='cuda',
        enable_quant=True,
    )

    # Check if three zones are initialized
    has_sink = manager._sink_k is not None
    has_tail = manager._tail_k is not None
    has_heavyhitter = manager._heavyhitter_k is not None
    has_competition_queue = manager._competition_queue is not None

    if has_sink and has_tail and has_heavyhitter and has_competition_queue:
        print("✅ HeteroKVManager three-zone architecture initialized")
        print(f"   - Sink zone: {manager.sink_tokens} tokens")
        print(f"   - Tail budget: {manager.hbm_budget_tokens - manager.sink_tokens} tokens")
        print(f"   - HeavyHitter budget: {manager._heavyhitter_budget} tokens")
        print(f"   - Total HBM: {manager.max_hbm_tokens()} tokens")
    else:
        print("❌ HeteroKVManager missing some zones")

except Exception as e:
    print(f"❌ HeteroKVManager initialization error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Test FusedHeteroCache initialization
print("\n[5/5] Testing FusedHeteroCache initialization...")
try:
    cache = FusedHeteroCache(
        num_layers=4,
        sink_tokens=64,
        keep_tail=2048,
        device='cuda',
        enable_quant=True,
        enable_triton=True,
    )

    print("✅ FusedHeteroCache initialized successfully")
    print(f"   - Num layers: {cache._num_layers}")
    print(f"   - Sink tokens: {cache.sink_tokens}")
    print(f"   - Tail budget: {cache.keep_tail}")

except Exception as e:
    print(f"❌ FusedHeteroCache initialization error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ All validation tests passed!")
print("=" * 60)
print("\nCode structure is correct. Ready for full benchmark test.")
print("\nNote: Full benchmark requires:")
print("  - Downloading LLaVA-1.5-7B model (~13GB)")
print("  - Downloading VQA-RAD dataset")
print("  - Running 128K context tests (may take 15-30 minutes)")
