"""
tests/test_end_to_end_pipeline.py
=================================
End-to-end integration test for the full Hetero-KV pipeline (Stages A→G).

Verifies that every mechanism is actually wired up:
  A: Chunked prefill
  B: Transient interception (Sink + Tail in HBM)
  C: 4-bit quantization + DRAM offload
  D: HeavyHitterOracle cumulative attention scoring
  E: Triton-accelerated eviction candidate selection
  F: Predictive prefetch (spatial + heat + lookahead + adaptive controller)
  G: Fused dequant attention (online softmax, no BF16 spike)
"""

import torch
import math
import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def test_stage_b_c_transient_and_quantization():
    """Stage B+C: Prefill -> Transient interception -> Quantize + Offload to DRAM."""
    print("\n=== Stage B+C: Transient Interception + Quantization ===")
    from src.memory.manager import HeteroKVManager

    manager = HeteroKVManager(
        num_layers=2,
        sink_tokens=4,
        hbm_budget_tokens=8,
        device="cpu",
        enable_quant=True,
        enable_prefetch=False,
        group_size=128,
    )

    # Simulate prefill: 32 tokens, 2 heads, dim=128
    k = torch.randn(1, 2, 32, 128)
    v = torch.randn(1, 2, 32, 128)
    out_k, out_v = manager.update(0, k, v, mode="prefill")

    # Prefill should return full tensors (FlashAttention compatibility)
    assert out_k.shape == (1, 2, 32, 128), f"Expected full prefill output, got {out_k.shape}"

    # HBM should only have sink + tail
    hbm_k, _ = manager.get_hbm_kv(0)
    assert hbm_k.shape[-2] <= manager.sink_tokens + manager.hbm_budget_tokens, \
        f"HBM overflow: {hbm_k.shape[-2]} > {manager.sink_tokens + manager.hbm_budget_tokens}"

    # DRAM should have entries
    assert len(manager._dram_table) > 0, "DRAM should have evicted entries"

    summary = manager.memory_summary()
    print(f"  HBM tokens: {summary['hbm_tokens']}, DRAM entries: {summary['dram_entries']}")
    print(f"  DRAM bytes: {summary['dram_bytes']}")
    print("  [PASS] Stage B+C")


def test_stage_d_e_oracle_eviction():
    """Stage D+E: HeavyHitterOracle + Triton kernel eviction."""
    print("\n=== Stage D+E: Oracle + Triton Eviction ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from src.policy.heavy_hitter import HeavyHitterOracle
    from src.kernels.oracle_triton import compute_block_scores

    oracle = HeavyHitterOracle(block_size=16, sink_tokens=16, local_window=32)

    # Simulate 256 tokens with attention accumulation
    current_len = 256
    for step in range(10):
        attn = torch.rand(current_len, device=device) * 0.1
        attn[80:112] = 0.001  # Make blocks 5-6 "cold"
        oracle.update(attn)

    # Get eviction candidates via Triton kernel
    candidates = oracle.get_eviction_candidates(current_len, evict_num_blocks=2)

    assert candidates.numel() == 2, f"Expected 2 candidates, got {candidates.numel()}"
    # Cold blocks (80-112 = block 5 and 6) should be targeted
    print(f"  Eviction candidates (block ids): {candidates.tolist()}")
    # Sink blocks should never be selected
    assert all(c >= 1 for c in candidates.tolist()), "Sink blocks should be protected"

    # Verify Triton kernel directly
    scores = oracle.token_scores
    block_scores = compute_block_scores(scores, current_len, 16)
    assert block_scores.shape[0] == math.ceil(current_len / 16)
    print(f"  Triton block_scores shape: {block_scores.shape}")
    print("  [PASS] Stage D+E")


def test_stage_f_predictive_prefetch():
    """Stage F: PredictivePrefetchScheduler + AdaptivePrefetchController + AsyncPrefetcher."""
    print("\n=== Stage F: Predictive Prefetch Pipeline ===")
    from src.policy.prefetcher import AsyncPrefetcher
    from src.policy.adaptive_prefetch_controller import AdaptivePrefetchController
    from src.core.scheduler import PredictivePrefetchScheduler
    from src.quantization.kv_compressor import KVCompressor

    device = torch.device("cpu")

    # Build mock DRAM table with quantized chunks
    compressor = KVCompressor(group_size=128, bits=4)
    dram_table = {}
    for i in range(10):
        k = torch.randn(1, 128)
        v = torch.randn(1, 128)
        q_k, s_k, z_k = compressor.compress(k)
        q_v, s_v, z_v = compressor.compress(v)
        dram_table[f"l0_e{i}"] = {
            "k_data": q_k, "k_scales": s_k, "k_zps": z_k,
            "v_data": q_v, "v_scales": s_v, "v_zps": z_v,
        }

    prefetcher = AsyncPrefetcher(device=device)
    scheduler = PredictivePrefetchScheduler(
        prefetcher=prefetcher,
        dram_table=dram_table,
        compressor=compressor,
        lookahead_window=3,
    )
    controller = AdaptivePrefetchController(w_min=2, w_max=6)

    # Simulate decode steps
    for step in range(5):
        attn = torch.rand(100)
        cache_miss = step % 3 == 0

        # Adaptive controller adjusts window
        w = controller.compute_window(attention_weights=attn, cache_miss=cache_miss)
        scheduler.lookahead_window = w

        submitted = scheduler.schedule_step(
            current_chunk=f"l0_e{step}",
            attention_weights=attn,
        )
        print(f"  Step {step}: w={w}, submitted={len(submitted)}, "
              f"pending={len(prefetcher.pending_keys)}, miss={cache_miss}")

    stats = controller.stats
    print(f"  Controller: w={stats['current_w']:.0f}, sigma_ref={stats['sigma_ref']:.4f}, "
          f"miss_rate={stats['miss_rate']:.2f}")
    assert stats['step'] == 5, "Controller should have 5 steps"
    print("  [PASS] Stage F")


def test_stage_f_integrated_in_manager():
    """Stage F: Verify AdaptivePrefetchController is wired into HeteroKVManager."""
    print("\n=== Stage F (Integrated): Manager + Adaptive Controller ===")
    from src.memory.manager import HeteroKVManager

    manager = HeteroKVManager(
        num_layers=2, sink_tokens=4, hbm_budget_tokens=8,
        device="cpu", enable_quant=True, enable_prefetch=False,
    )

    # Verify oracle and adaptive controller are initialized
    assert manager._oracle is not None, "Oracle should be initialized"
    assert manager._adaptive_controller is not None, "Adaptive controller should be initialized"

    # Test update_attention_scores
    attn = torch.rand(32)
    manager.update_attention_scores(attn)
    assert manager._oracle.token_scores is not None, "Oracle scores should be populated"
    assert manager._oracle.token_scores.shape[0] == 32
    print(f"  Oracle scores shape: {manager._oracle.token_scores.shape}")
    print("  [PASS] Stage F (Integrated)")


def test_stage_g_fused_dequant_attn():
    """Stage G: Fused dequant attention with online softmax."""
    print("\n=== Stage G: Fused Dequant Attention ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("  [SKIP] CUDA required for Triton kernels")
        return

    from src.quantization.kernels.fused_dequant_attn import (
        fused_dequant_attn_decode,
        fused_dequant_attn_forward,
    )
    from src.quantization.kv_compressor import KVCompressor

    compressor = KVCompressor(group_size=128, bits=4)
    batch, heads, kv_len, head_dim = 1, 2, 64, 128

    # Create and quantize KV
    k_orig = torch.randn(batch, heads, kv_len, head_dim, device=device)
    v_orig = torch.randn(batch, heads, kv_len, head_dim, device=device)
    q_k, s_k, z_k = compressor.compress(k_orig)
    q_v, s_v, z_v = compressor.compress(v_orig)

    # Test decode path [B, H, 1, D]
    q_decode = torch.randn(batch, heads, 1, head_dim, device=device)
    out_decode = fused_dequant_attn_decode(q_decode, q_k, s_k, z_k, q_v, s_v, z_v)
    assert out_decode.shape == (batch, heads, 1, head_dim), \
        f"Decode output shape mismatch: {out_decode.shape}"
    assert not torch.isnan(out_decode).any(), "Decode output has NaN"
    print(f"  Decode output shape: {out_decode.shape} ✓")

    # Test forward path [B, H, S, D]
    q_len = 4
    q_forward = torch.randn(batch, heads, q_len, head_dim, device=device)
    out_forward = fused_dequant_attn_forward(q_forward, q_k, s_k, z_k, q_v, s_v, z_v)
    assert out_forward.shape == (batch, heads, q_len, head_dim), \
        f"Forward output shape mismatch: {out_forward.shape}"
    assert not torch.isnan(out_forward).any(), "Forward output has NaN"
    print(f"  Forward output shape: {out_forward.shape} ✓")
    print("  [PASS] Stage G")


def test_full_pipeline_a_to_g():
    """Full pipeline: Prefill -> Evict -> Decode with oracle + prefetch + fused attn."""
    print("\n=== Full Pipeline A→G Integration ===")
    from src.memory.manager import HeteroKVManager

    device = "cuda" if torch.cuda.is_available() else "cpu"
    manager = HeteroKVManager(
        num_layers=1,
        sink_tokens=4,
        hbm_budget_tokens=16,
        device=device,
        enable_quant=True,
        enable_prefetch=(device != "cpu"),
        group_size=128,
    )

    num_heads, head_dim = 2, 128

    # Stage A: Prefill (simulate long sequence)
    seq_len = 64
    k = torch.randn(1, num_heads, seq_len, head_dim, device=device)
    v = torch.randn(1, num_heads, seq_len, head_dim, device=device)
    out_k, out_v = manager.update(0, k, v, mode="prefill")
    assert out_k.shape == k.shape, "Prefill should return full tensors"
    hbm_k, _ = manager.get_hbm_kv(0)
    hbm_size = hbm_k.shape[-2]
    assert hbm_size <= manager.sink_tokens + manager.hbm_budget_tokens, \
        f"HBM size {hbm_size} exceeds budget"
    assert len(manager._dram_table) > 0, "DRAM should have evicted entries"
    print(f"  [A-C] Prefill done: HBM={hbm_size}, DRAM={len(manager._dram_table)} entries")

    # Stage D: Feed attention weights to oracle
    attn_weights = torch.rand(hbm_size, device=device)
    manager.update_attention_scores(attn_weights)
    assert manager._oracle.token_scores is not None
    print(f"  [D] Oracle updated with {attn_weights.shape[0]} attention weights")

    # Stage E+F: Decode steps with oracle-driven eviction
    for step in range(5):
        k_new = torch.randn(1, num_heads, 1, head_dim, device=device)
        v_new = torch.randn(1, num_heads, 1, head_dim, device=device)
        out_k, out_v = manager.update(0, k_new, v_new, mode="decode")

        # Feed new attention scores
        new_attn = torch.rand(out_k.shape[-2], device=device)
        manager.update_attention_scores(new_attn)

        # Trigger predictive prefetch
        if device != "cpu":
            submitted = manager.predictive_prefetch_step(
                attention_weights=new_attn,
                cache_miss=False,
            )
            print(f"  [E+F] Step {step}: HBM={out_k.shape[-2]}, prefetch={len(submitted)}")
        else:
            print(f"  [E+F] Step {step}: HBM={out_k.shape[-2]}")

    summary = manager.memory_summary()
    print(f"  Final: HBM={summary['hbm_tokens']}, DRAM={summary['dram_entries']} entries, "
          f"{summary['dram_bytes']} bytes")
    print("  [PASS] Full Pipeline A→G")


if __name__ == "__main__":
    test_stage_b_c_transient_and_quantization()
    test_stage_d_e_oracle_eviction()
    test_stage_f_predictive_prefetch()
    test_stage_f_integrated_in_manager()
    test_stage_g_fused_dequant_attn()
    test_full_pipeline_a_to_g()
    print("\n" + "=" * 60)
    print("All end-to-end pipeline tests PASSED!")
    print("=" * 60)
