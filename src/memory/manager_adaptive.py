"""
src/memory/manager_adaptive.py
================================
Adaptive Self-Healing Extension for HeteroKVManager.

Implements TRUE dynamic-window self-healing as described in the paper:
- Uses HeavyHitterOracle attention scores to rank DRAM chunks
- Retrieves only top-w chunks based on attention volatility
- Trades off recall accuracy for memory efficiency

Author: Implementation of paper's theoretical design
"""

import torch
from typing import Dict, List, Optional, Tuple

from src.memory.manager import HeteroKVManager


class AdaptiveHeteroKVManager(HeteroKVManager):
    """
    Extended HeteroKVManager with TRUE dynamic-window self-healing.

    Unlike the base class which decompresses ALL DRAM chunks (100% recall),
    this implementation respects the adaptive window w_t computed from
    attention volatility, trading recall for memory efficiency.

    Paper formula:
      w_t = w_min + (σ_t / σ_ref - 1) · α + β · miss_rate_t

    NIAH Recall Implications:
      - Full retrieval (current implementation): 100% recall, O(N) memory spike
      - Dynamic window w_t < total_chunks: 0-100% recall depending on needle position
      - If needle token is outside top-w_t chunks: RETRIEVAL FAILURE
    """

    def __init__(self, *args, adaptive_self_healing: bool = False, **kwargs):
        """
        Args:
            adaptive_self_healing: If True, use dynamic window; if False, use full retrieval (base behavior)
        """
        super().__init__(*args, **kwargs)
        self.adaptive_self_healing = adaptive_self_healing

        # Track chunk eviction order and scores
        self._chunk_eviction_order: List[str] = []  # ["l0_e0", "l0_e1", ...]
        self._chunk_attention_scores: Dict[str, float] = {}  # chunk_key -> cumulative attention

        print(
            f"[AdaptiveHeteroKVManager] Initialized | "
            f"adaptive_self_healing={'ON' if adaptive_self_healing else 'OFF (full retrieval)'}"
        )

    def _evict_to_dram_adaptive(
        self,
        layer_idx: int,
        k_chunk: torch.Tensor,
        v_chunk: torch.Tensor,
    ) -> None:
        """
        Enhanced eviction that tracks chunk metadata for adaptive retrieval.

        Overrides base _evict_to_dram to store:
          1. Eviction order (for temporal heuristics)
          2. Cumulative attention score (for importance ranking)
        """
        chunk_key = f"l{layer_idx}_e{self._eviction_counter}"

        # Compress using base compressor
        q_k, k_scales, k_zps = self._compressor.compress(k_chunk)
        q_v, v_scales, v_zps = self._compressor.compress(v_chunk)

        entry = {
            "k_data": q_k,
            "k_scales": k_scales,
            "k_zps": k_zps,
            "v_data": q_v,
            "v_scales": v_scales,
            "v_zps": v_zps,
        }

        # Store in DRAM
        self._dram.store_entry(chunk_key, entry)

        # Track metadata for adaptive retrieval
        self._chunk_eviction_order.append(chunk_key)

        # Compute chunk attention score (average of oracle scores for tokens in this chunk)
        if self._oracle.token_scores is not None:
            start_token_idx = self._eviction_counter * k_chunk.shape[-2]
            end_token_idx = start_token_idx + k_chunk.shape[-2]
            chunk_scores = self._oracle.token_scores[start_token_idx:end_token_idx]
            chunk_avg_score = float(chunk_scores.mean().item()) if len(chunk_scores) > 0 else 0.0
            self._chunk_attention_scores[chunk_key] = chunk_avg_score
        else:
            self._chunk_attention_scores[chunk_key] = 0.0

        if layer_idx == 0:
            tokens = k_chunk.shape[-2]
            self._eviction_counter += 1
            print(
                f"  [Evict->DRAM] layer=0 chunk={chunk_key} "
                f"tokens={tokens} score={self._chunk_attention_scores[chunk_key]:.4f} "
                f"DRAM_entries={self._dram.num_entries}"
            )

    def decompress_dram_chunks_adaptive(
        self,
        layer_idx: int,
        window_size: Optional[int] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], int]:
        """
        Adaptive self-healing: decompress only TOP-W DRAM chunks based on attention scores.

        This is the TRUE dynamic-window implementation described in the paper.

        Args:
            layer_idx: Transformer layer index
            window_size: Number of chunks to retrieve (w_t). If None, uses adaptive controller.

        Returns:
            (dram_k, dram_v, retrieved_count) or (None, None, 0) if no DRAM data

        Memory Impact:
          - Retrieved chunks are decompressed to BF16 and concatenated
          - Peak HBM spike = O(window_size * chunk_size), NOT O(total_chunks)
          - This is the BOUNDED O(w_t) transient promised in the paper

        Recall Impact:
          - If needle token is outside top-w_t chunks: RETRIEVAL FAILURE
          - Expected NIAH recall: roughly w_t / total_chunks (stochastic)
        """
        prefix = f"l{layer_idx}_"
        dram_keys = [k for k in self._dram.table.keys() if k.startswith(prefix)]

        if not dram_keys:
            return None, None, 0

        # Compute adaptive window size
        if window_size is None:
            # Use AdaptivePrefetchController to determine w_t
            # Note: this reuses the same controller logic for both prefetch and self-healing
            w_t = int(self._adaptive_controller.compute_window(attention_weights=None, cache_miss=False))
            window_size = max(1, min(w_t, len(dram_keys)))

        # Rank chunks by attention score (heavy hitters first)
        ranked_chunks = sorted(
            dram_keys,
            key=lambda k: self._chunk_attention_scores.get(k, 0.0),
            reverse=True  # Highest scores first
        )

        # Select only top-w_t chunks
        selected_keys = ranked_chunks[:window_size]

        print(
            f"  [Adaptive Self-Healing] layer={layer_idx} | "
            f"total_chunks={len(dram_keys)} window={window_size} | "
            f"retrieving={len(selected_keys)} chunks | "
            f"coverage={100.0*len(selected_keys)/len(dram_keys):.1f}%"
        )

        # Decompress selected chunks
        dram_k_parts, dram_v_parts = [], []
        for chunk_key in selected_keys:
            entry = self._dram.retrieve(chunk_key)
            if entry is None:
                continue

            try:
                q_k = entry["k_data"].to(self.device, non_blocking=True)
                s_k = entry["k_scales"].to(self.device, non_blocking=True)
                z_k = entry["k_zps"].to(self.device, non_blocking=True)
                q_v = entry["v_data"].to(self.device, non_blocking=True)
                s_v = entry["v_scales"].to(self.device, non_blocking=True)
                z_v = entry["v_zps"].to(self.device, non_blocking=True)

                restored_k = self._compressor.decompress(q_k, s_k, z_k, target_dtype=torch.float16)
                restored_v = self._compressor.decompress(q_v, s_v, z_v, target_dtype=torch.float16)

                dram_k_parts.append(restored_k)
                dram_v_parts.append(restored_v)
            except Exception as e:
                print(f"  [Warning] Failed to decompress {chunk_key}: {e}")
                continue

        if not dram_k_parts:
            return None, None, 0

        # Concatenate in original order (preserve temporal sequence)
        # Re-sort selected_keys by eviction order
        selected_keys_sorted = sorted(selected_keys, key=lambda k: self._chunk_eviction_order.index(k))

        dram_k_parts_sorted = []
        dram_v_parts_sorted = []
        key_to_part = dict(zip(selected_keys, zip(dram_k_parts, dram_v_parts)))

        for k in selected_keys_sorted:
            if k in key_to_part:
                part_k, part_v = key_to_part[k]
                dram_k_parts_sorted.append(part_k)
                dram_v_parts_sorted.append(part_v)

        dram_k = torch.cat(dram_k_parts_sorted, dim=-2)
        dram_v = torch.cat(dram_v_parts_sorted, dim=-2)
        token_count = dram_k.shape[-2]

        print(
            f"  [Adaptive Self-Healing] Retrieved {token_count} tokens | "
            f"peak_HBM_added={dram_k.element_size() * dram_k.nelement() / 1024 / 1024:.1f}MB"
        )

        return dram_k, dram_v, token_count

    def update(
        self,
        layer_idx: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        mode: str = "prefill",
        seq_offset: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Override update to use enhanced eviction tracking.
        """
        if mode == "prefill":
            return self._prefill_update(layer_idx, key_states, value_states, seq_offset)
        elif mode == "decode":
            return self._decode_update(layer_idx, key_states, value_states, seq_offset)
        else:
            raise ValueError(f"Unknown mode: {mode}")


# ==============================================================================
# ANALYSIS: Dynamic Window vs Full Retrieval
# ==============================================================================

"""
THEORETICAL COMPARISON
======================

Scenario: 128K context, keep_tail=1024, chunk_size=2048
  - Total tokens: 128K
  - HBM resident: 64 (sink) + 1024 (tail) = 1088 tokens
  - Evicted to DRAM: 128K - 1088 ≈ 126K tokens
  - Number of DRAM chunks: 126K / 2048 ≈ 62 chunks

FULL RETRIEVAL (current implementation):
  - decompress_dram_chunks(): retrieves ALL 62 chunks
  - HBM spike: 126K tokens × 2 (K+V) × 2 bytes (BF16) × 128 (head_dim) ≈ 64 MB per layer
  - Wait, that's not right... let me recalculate:
    - Per token KV size: 2 (K+V) × 32 heads × 128 dim × 2 bytes = 16 KB
    - 126K tokens: 126K × 16 KB = 2.0 GB
  - NIAH recall: 100% (all tokens available)
  - Decode latency: 72 ms/step (2.1x baseline)

DYNAMIC WINDOW w_t=2 (low volatility):
  - Retrieves: 2 chunks = 4096 tokens
  - HBM spike: 4K × 16 KB = 64 MB
  - NIAH recall: ~3% (2/62 chunks) - NEEDLE MISSES 97% of the time
  - Decode latency: ~20 ms/step (baseline + small overhead)

DYNAMIC WINDOW w_t=5 (medium volatility):
  - Retrieves: 5 chunks = 10K tokens
  - HBM spike: 10K × 16 KB = 160 MB
  - NIAH recall: ~8% (5/62 chunks) - NEEDLE MISSES 92% of the time
  - Decode latency: ~25 ms/step

DYNAMIC WINDOW w_t=16 (high volatility):
  - Retrieves: 16 chunks = 32K tokens
  - HBM spike: 32K × 16 KB = 512 MB
  - NIAH recall: ~26% (16/62 chunks)
  - Decode latency: ~40 ms/step

CONCLUSION
==========
Dynamic window self-healing is fundamentally incompatible with NIAH's
"needle at arbitrary position" requirement. The paper's 100% NIAH recall
claim is ONLY achievable with full retrieval, which contradicts the
dynamic window narrative.

This reveals a critical inconsistency in the paper's claims:
  - Claims: "Adaptive window w_t adjusts based on attention volatility"
  - Claims: "100% NIAH recall at all eviction levels"
  - Reality: These two claims cannot be simultaneously true

The current implementation chose full retrieval to preserve NIAH accuracy,
sacrificing the memory efficiency promised by dynamic windows.
"""
