import os
import sys

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.memory.manager import HeteroKVManager
from src.memory.query_aware_retriever import HybridRetrievalStrategy
from src.core.fused_attention_patch import heterokv_safe_attention_forward
from src.quantization.kv_compressor import KVCompressor


def test_prefill_returns_short_kv_and_tracks_positions():
    manager = HeteroKVManager(
        num_layers=1,
        sink_tokens=2,
        hbm_budget_tokens=6,
        device="cpu",
        enable_quant=True,
        enable_prefetch=False,
        group_size=4,
    )
    k = torch.randn(1, 2, 8, 4, dtype=torch.float16)
    v = torch.randn(1, 2, 8, 4, dtype=torch.float16)

    out_k, out_v = manager.update(0, k, v, mode="prefill", seq_offset=0)

    assert out_k.shape[-2] == 6
    assert out_v.shape[-2] == 6
    assert manager._key_cache[0].shape[-2] == 6
    assert manager.count_dram_tokens(0) == 2
    assert manager.get_key_positions(0).tolist() == [0, 1, 4, 5, 6, 7]
    entry = next(iter(manager._dram.table.values()))
    for key in ("k_data", "k_scales", "k_zps", "v_data", "v_scales", "v_zps"):
        assert key in entry


def test_incremental_prefill_stays_bounded():
    manager = HeteroKVManager(
        num_layers=1,
        sink_tokens=2,
        hbm_budget_tokens=6,
        device="cpu",
        enable_quant=True,
        enable_prefetch=False,
        group_size=4,
    )
    k0 = torch.randn(1, 2, 8, 4, dtype=torch.float16)
    v0 = torch.randn(1, 2, 8, 4, dtype=torch.float16)
    manager.update(0, k0, v0, mode="prefill", seq_offset=0)

    k1 = torch.randn(1, 2, 4, 4, dtype=torch.float16)
    v1 = torch.randn(1, 2, 4, 4, dtype=torch.float16)
    out_k, _ = manager.update(0, k1, v1, mode="prefill", seq_offset=8)

    assert out_k.shape[-2] == 6
    assert manager._key_cache[0].shape[-2] == 6
    assert manager.get_key_positions(0).tolist() == [0, 1, 8, 9, 10, 11]
    assert manager.count_dram_tokens(0) >= 2


def test_dot_product_retrieval_hits_token_level_target():
    compressor = KVCompressor(group_size=4, bits=4)
    target_k = torch.zeros(1, 1, 2, 4, dtype=torch.float16)
    target_v = torch.randn(1, 1, 2, 4, dtype=torch.float16)
    other_k = torch.zeros(1, 1, 2, 4, dtype=torch.float16)
    other_v = torch.randn(1, 1, 2, 4, dtype=torch.float16)
    query = torch.zeros(1, 1, 1, 4, dtype=torch.float16)

    target_k[..., 1, 0] = 8.0
    query[..., 0, 0] = 8.0
    other_k[..., :, 1] = 1.0

    def pack(k, v):
        qk, sk, zk = compressor.compress(k)
        qv, sv, zv = compressor.compress(v)
        return {
            "k_data": qk,
            "k_scales": sk,
            "k_zps": zk,
            "v_data": qv,
            "v_scales": sv,
            "v_zps": zv,
        }

    dram_table = {
        "l0_e0": pack(other_k, other_v),
        "l0_e1": pack(target_k, target_v),
    }
    retriever = HybridRetrievalStrategy(device="cpu", enable=True, alpha=1.0)
    retriever.register_chunk("l0_e0", 0, 2, historical_attention=0.0)
    retriever.register_chunk("l0_e1", 2, 4, historical_attention=0.0)

    selected, method = retriever.retrieve_chunks(
        query_key=query,
        candidate_keys=["l0_e0", "l0_e1"],
        top_k=1,
        dram_table=dram_table,
        compressor=compressor,
    )

    assert method == "dot_product"
    assert selected == ["l0_e1"]


def test_dot_product_retrieval_handles_gqa_heads():
    compressor = KVCompressor(group_size=4, bits=4)
    target_k = torch.zeros(1, 4, 2, 4, dtype=torch.float16)
    target_v = torch.randn(1, 4, 2, 4, dtype=torch.float16)
    other_k = torch.zeros(1, 4, 2, 4, dtype=torch.float16)
    other_v = torch.randn(1, 4, 2, 4, dtype=torch.float16)
    query = torch.zeros(1, 28, 1, 4, dtype=torch.float16)

    target_k[:, 1, 1, 0] = 8.0
    query[:, 8, 0, 0] = 8.0  # query head 8 belongs to KV head 1 when groups=7
    other_k[:, :, :, 1] = 1.0

    def pack(k, v):
        qk, sk, zk = compressor.compress(k)
        qv, sv, zv = compressor.compress(v)
        return {
            "k_data": qk,
            "k_scales": sk,
            "k_zps": zk,
            "v_data": qv,
            "v_scales": sv,
            "v_zps": zv,
        }

    dram_table = {
        "l0_e0": pack(other_k, other_v),
        "l0_e1": pack(target_k, target_v),
    }
    retriever = HybridRetrievalStrategy(device="cpu", enable=True, alpha=1.0)
    retriever.register_chunk("l0_e0", 0, 2, historical_attention=0.0)
    retriever.register_chunk("l0_e1", 2, 4, historical_attention=0.0)

    selected, method = retriever.retrieve_chunks(
        query_key=query,
        candidate_keys=["l0_e0", "l0_e1"],
        top_k=1,
        dram_table=dram_table,
        compressor=compressor,
    )

    assert method == "dot_product"
    assert selected == ["l0_e1"]


def test_query_history_reranker_uses_multiple_query_tokens():
    compressor = KVCompressor(group_size=4, bits=4)
    spike_k = torch.zeros(1, 1, 2, 4, dtype=torch.float16)
    spike_v = torch.randn(1, 1, 2, 4, dtype=torch.float16)
    consensus_k = torch.zeros(1, 1, 2, 4, dtype=torch.float16)
    consensus_v = torch.randn(1, 1, 2, 4, dtype=torch.float16)
    query = torch.zeros(1, 1, 2, 4, dtype=torch.float16)

    query[..., 0, 0] = 4.0
    query[..., 1, 1] = 4.0
    spike_k[..., 1, 1] = 7.0
    consensus_k[..., 0, 0] = 4.0
    consensus_k[..., 1, 1] = 4.0

    def pack(k, v):
        qk, sk, zk = compressor.compress(k)
        qv, sv, zv = compressor.compress(v)
        return {
            "k_data": qk,
            "k_scales": sk,
            "k_zps": zk,
            "v_data": qv,
            "v_scales": sv,
            "v_zps": zv,
        }

    dram_table = {
        "l0_spike": pack(spike_k, spike_v),
        "l0_consensus": pack(consensus_k, consensus_v),
    }
    retriever = HybridRetrievalStrategy(
        device="cpu",
        enable=True,
        alpha=1.0,
        score_reduce="query_top_r_mean",
        top_r=2,
    )
    retriever.register_chunk("l0_spike", 0, 2, historical_attention=0.0)
    retriever.register_chunk("l0_consensus", 2, 4, historical_attention=0.0)

    selected, method = retriever.retrieve_chunks(
        query_key=query,
        candidate_keys=["l0_spike", "l0_consensus"],
        top_k=1,
        dram_table=dram_table,
        compressor=compressor,
    )

    assert method == "dot_product"
    assert selected == ["l0_consensus"]


def test_source_token_overlap_scores_rare_query_terms():
    manager = HeteroKVManager(
        num_layers=1,
        sink_tokens=2,
        hbm_budget_tokens=6,
        device="cpu",
        enable_quant=True,
        enable_prefetch=False,
        group_size=4,
        enable_method_d=True,
        method_d_source_token_boost=1.0,
        method_d_source_query_tokens=4,
    )
    manager.set_source_token_ids(torch.tensor([10, 11, 12, 20, 21, 30, 31, 40, 41, 20, 21]))
    manager._chunk_position_ranges = {
        "l0_e0": (0, 3),
        "l0_e1": (3, 7),
    }

    query_end = 7
    unrelated = manager._method_d_source_token_score("l0_e0", query_end)
    related = manager._method_d_source_token_score("l0_e1", query_end)

    assert related > unrelated


def test_source_overlap_filter_drops_zero_overlap_false_positive():
    compressor = KVCompressor(group_size=4, bits=4)
    manager = HeteroKVManager(
        num_layers=1,
        sink_tokens=2,
        hbm_budget_tokens=6,
        device="cpu",
        enable_quant=True,
        enable_prefetch=False,
        group_size=4,
        enable_method_d=True,
        method_d_source_token_boost=1.0,
        method_d_source_query_tokens=2,
        method_d_require_source_overlap=True,
    )

    class DummyQueryAware:
        last_scores = {"l0_bad": 1000.0, "l0_good": 1.0}
        last_best_token_offsets = {"l0_bad": 0, "l0_good": 0}

    class DummyRetriever:
        query_aware_retriever = DummyQueryAware()

        def retrieve_chunks(self, **kwargs):
            return ["l0_bad"], "dot_product"

    def pack(k, v, positions):
        qk, sk, zk = compressor.compress(k)
        qv, sv, zv = compressor.compress(v)
        return {
            "k_data": qk,
            "k_scales": sk,
            "k_zps": zk,
            "v_data": qv,
            "v_scales": sv,
            "v_zps": zv,
            "positions": positions,
        }

    bad_k = torch.randn(1, 1, 2, 4, dtype=torch.float16)
    bad_v = torch.randn(1, 1, 2, 4, dtype=torch.float16)
    good_k = torch.randn(1, 1, 2, 4, dtype=torch.float16)
    good_v = torch.randn(1, 1, 2, 4, dtype=torch.float16)
    manager._method_d_retriever = DummyRetriever()
    manager._dram.store_entry("l0_bad", pack(bad_k, bad_v, torch.tensor([0, 1])))
    manager._dram.store_entry("l0_good", pack(good_k, good_v, torch.tensor([2, 3])))
    manager._chunk_position_ranges = {"l0_bad": (0, 2), "l0_good": (2, 4)}
    manager._chunk_eviction_order = ["l0_bad", "l0_good"]
    manager._seq_offsets = [4]
    manager.set_source_token_ids(torch.tensor([100, 101, 200, 201]))

    query = torch.randn(1, 1, 1, 4, dtype=torch.float16)
    _, _, count, method = manager.decompress_dram_chunks_method_d(0, query, top_k=1)
    selected = manager.get_last_method_d_selection(0)

    assert count == 2
    assert "source_filtered" in method
    assert selected[0]["chunk_key"] == "l0_good"
    assert selected[0]["source_token_score"] > 0.0


def test_method_d_retrieval_preserves_query_dtype():
    manager = HeteroKVManager(
        num_layers=1,
        sink_tokens=2,
        hbm_budget_tokens=6,
        device="cpu",
        enable_quant=True,
        enable_prefetch=False,
        group_size=4,
        enable_method_d=True,
    )
    k = torch.randn(1, 2, 8, 4, dtype=torch.bfloat16)
    v = torch.randn(1, 2, 8, 4, dtype=torch.bfloat16)
    manager.update(0, k, v, mode="prefill", seq_offset=0)

    query = torch.randn(1, 2, 1, 4, dtype=torch.bfloat16)
    dram_k, dram_v, count, _ = manager.decompress_dram_chunks_method_d(0, query, top_k=1)

    assert count > 0
    assert dram_k.dtype == torch.bfloat16
    assert dram_v.dtype == torch.bfloat16


def test_short_kv_attention_extends_mask_for_retrieved_prefix():
    class DummyAttention:
        num_key_value_groups = 2
        training = False

    query = torch.randn(1, 2, 1, 4, dtype=torch.float32)
    key = torch.randn(1, 1, 4, 4, dtype=torch.float32)
    value = torch.randn(1, 1, 4, 4, dtype=torch.float32)
    short_mask = torch.zeros(1, 1, 1, 2, dtype=torch.float32)
    key_positions = torch.arange(4)
    cache_position = torch.tensor([3])

    out, weights = heterokv_safe_attention_forward(
        DummyAttention(),
        query,
        key,
        value,
        short_mask,
        cache_position,
        key_positions,
        scaling=0.5,
    )

    assert out.shape == (1, 1, 2, 4)
    assert weights.shape[-1] == 4


def test_source_fusion_can_route_retrieved_prefix_output():
    class DummyAttention:
        num_key_value_groups = 1
        training = False

    query = torch.tensor([[[[1.0, 0.0]]]])
    key = torch.tensor([[[[0.0, 1.0], [1.0, 0.0]]]])
    value = torch.tensor([[[[10.0, 0.0], [0.0, 20.0]]]])
    key_positions = torch.tensor([0, 1])
    cache_position = torch.tensor([1])

    out, weights = heterokv_safe_attention_forward(
        DummyAttention(),
        query,
        key,
        value,
        attention_mask=None,
        cache_position=cache_position,
        key_positions=key_positions,
        scaling=1.0,
        retrieved_count=1,
        retrieval_source_fusion_alpha=1.0,
    )

    assert out.shape == (1, 1, 1, 2)
    assert weights.shape[-1] == 2
    assert torch.allclose(out[0, 0, 0], torch.tensor([10.0, 0.0]), atol=1e-5)
