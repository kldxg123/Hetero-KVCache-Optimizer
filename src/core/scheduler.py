"""
src/core/scheduler.py
=====================
PredictivePrefetchScheduler: proactive async prefetch for DRAM-resident cold blocks.

Design:
  - Spatial locality: if chunk_i was accessed, pre-adjacent chunks are likely next.
  - Attention heat: uses HeavyHitterOracle cumulative scores to prioritize
    high-attention DRAM chunks for early prefetch.
  - Lookahead window: speculatively prefetches N chunks ahead of the current
    decode position.

Protocol:
  After each decode step, the caller invokes schedule_step(current_chunk_idx, ...).
  The scheduler submits async H2D transfers via the existing AsyncPrefetcher.
"""

import torch
from typing import Dict, List, Optional, Tuple
from collections import OrderedDict


class PredictivePrefetchScheduler:
    """
    Proactive prefetch scheduler that predicts which DRAM-resident KV chunks
    will be needed in upcoming decode steps and submits async H2D transfers
    ahead of time.

    Predictive signals:
      1. Spatial locality: adjacent chunks to the most recently accessed chunk
      2. Attention heat: cumulative attention scores from HeavyHitterOracle
      3. Lookahead: fixed-size window of future chunks
    """

    def __init__(
        self,
        prefetcher,
        dram_table: Dict[str, Dict[str, torch.Tensor]],
        compressor,
        lookahead_window: int = 3,
        max_prefetch_per_step: int = 4,
        heat_threshold: float = 0.1,
    ):
        """
        Args:
            prefetcher: AsyncPrefetcher instance with background CUDA stream
            dram_table: reference to manager._dram_table
            compressor: KVCompressor for decompression
            lookahead_window: number of adjacent chunks to speculatively prefetch
            max_prefetch_per_step: cap on concurrent prefetch tasks per step
            heat_threshold: minimum normalized attention score for heat-based prefetch
        """
        self.prefetcher = prefetcher
        self.dram_table = dram_table
        self.compressor = compressor
        self.lookahead_window = lookahead_window
        self.max_prefetch_per_step = max_prefetch_per_step
        self.heat_threshold = heat_threshold

        # Track the logical ordering of chunks (populated on first scan)
        self._chunk_order: List[str] = []
        self._chunk_idx: Dict[str, int] = {}
        self._last_accessed_chunk: Optional[str] = None
        self._step_counter = 0

        # Attention heat scores: chunk_key -> cumulative_score
        self._heat_scores: Dict[str, float] = {}

        # Chunk key to (layer, eviction_idx) parsing cache
        self._chunk_meta: Dict[str, Tuple[int, int]] = {}

        print(
            f"[PredictivePrefetchScheduler] init | "
            f"lookahead={lookahead_window} max_per_step={max_prefetch_per_step} "
            f"heat_threshold={heat_threshold}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rebuild_index(self) -> None:
        """Rebuild chunk ordering from current DRAM table state."""
        self._chunk_order = sorted(self.dram_table.keys())
        self._chunk_idx = {k: i for i, k in enumerate(self._chunk_order)}
        # Parse chunk metadata: "l{layer}_e{evict_idx}"
        for key in self._chunk_order:
            if key not in self._chunk_meta:
                try:
                    parts = key.split("_")
                    layer = int(parts[0][1:])
                    evict_idx = int(parts[1][1:])
                    self._chunk_meta[key] = (layer, evict_idx)
                except (IndexError, ValueError):
                    self._chunk_meta[key] = (0, 0)
        print(
            f"[PrefetchScheduler] index rebuilt | "
            f"total_chunks={len(self._chunk_order)}"
        )

    def schedule_step(
        self,
        current_chunk: Optional[str] = None,
        attention_weights: Optional[torch.Tensor] = None,
    ) -> List[str]:
        """
        Called after each decode step. Predicts which chunks are likely needed
        next and submits async prefetch tasks.

        Args:
            current_chunk: the chunk most recently accessed (for spatial locality)
            attention_weights: attention weights [seq_len] for heat tracking

        Returns:
            List of chunk keys that were submitted for prefetch
        """
        if self.prefetcher is None:
            return []

        submitted = []

        # Refresh index if DRAM table size changed
        if len(self.dram_table) != len(self._chunk_order):
            self.rebuild_index()

        if not self._chunk_order:
            return submitted

        # Update heat scores from attention weights
        if attention_weights is not None:
            self._update_heat_scores(attention_weights)

        # Signal 1: Spatial locality - prefetch adjacent chunks
        if current_chunk is not None and current_chunk in self._chunk_idx:
            spatial_chunks = self._get_spatial_neighbors(current_chunk)
            for ck in spatial_chunks:
                if self._submit_if_needed(ck):
                    submitted.append(ck)

        # Signal 2: Heat-based - prefetch high-attention chunks
        heat_chunks = self._get_hot_chunks()
        for ck in heat_chunks:
            if self._submit_if_needed(ck):
                submitted.append(ck)

        # Signal 3: Lookahead - prefetch sequentially ahead
        lookahead_chunks = self._get_lookahead_chunks(current_chunk)
        for ck in lookahead_chunks:
            if self._submit_if_needed(ck):
                submitted.append(ck)

        self._last_accessed_chunk = current_chunk
        self._step_counter += 1

        return submitted

    def notify_access(self, chunk_key: str) -> None:
        """Notify the scheduler that a chunk was accessed (updates locality tracker)."""
        self._last_accessed_chunk = chunk_key

    # ------------------------------------------------------------------
    # Prediction signals
    # ------------------------------------------------------------------

    def _get_spatial_neighbors(self, chunk_key: str) -> List[str]:
        """Return chunks adjacent to the given chunk in eviction order."""
        idx = self._chunk_idx.get(chunk_key, -1)
        if idx < 0:
            return []

        neighbors = []
        for delta in range(1, self.lookahead_window + 1):
            for offset in (-delta, delta):
                ni = idx + offset
                if 0 <= ni < len(self._chunk_order):
                    neighbors.append(self._chunk_order[ni])
        return neighbors

    def _get_hot_chunks(self) -> List[str]:
        """Return chunks with highest attention heat scores."""
        if not self._heat_scores:
            return []

        sorted_chunks = sorted(
            self._heat_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        # Normalize scores and filter by threshold
        if sorted_chunks:
            max_score = sorted_chunks[0][1]
            if max_score > 0:
                return [
                    ck for ck, score in sorted_chunks
                    if score / max_score >= self.heat_threshold
                ][:self.max_prefetch_per_step]
        return []

    def _get_lookahead_chunks(self, current_chunk: Optional[str]) -> List[str]:
        """Return the next N chunks in sequential order (speculative lookahead)."""
        if not self._chunk_order:
            return []

        start_idx = 0
        if current_chunk is not None and current_chunk in self._chunk_idx:
            start_idx = self._chunk_idx[current_chunk] + 1

        end_idx = min(start_idx + self.lookahead_window, len(self._chunk_order))
        return self._chunk_order[start_idx:end_idx]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _submit_if_needed(self, chunk_key: str) -> bool:
        """Submit a prefetch task if the chunk hasn't been prefetched yet."""
        if chunk_key not in self.dram_table:
            return False
        if chunk_key in self.prefetcher.pending_keys:
            return False

        entry = self.dram_table[chunk_key]
        prefetch_entry = {
            "k_data": entry["k_data"],
            "k_scales": entry["k_scales"],
            "k_zps": entry["k_zps"],
            "v_data": entry["v_data"],
            "v_scales": entry["v_scales"],
            "v_zps": entry["v_zps"],
        }
        self.prefetcher.submit_prefetch_task(chunk_key, prefetch_entry, self.compressor)
        return True

    def _update_heat_scores(self, attention_weights: torch.Tensor) -> None:
        """
        Map attention weights to DRAM chunks using chunk metadata.
        Each chunk covers a range of tokens; we sum their attention weights.
        """
        attn = attention_weights.detach().cpu().float()
        seq_len = attn.shape[0]

        for chunk_key, (layer, evict_idx) in self._chunk_meta.items():
            if chunk_key not in self._chunk_idx:
                continue
            # Approximate: use eviction order as a proxy for token range
            # In production, this would use the actual token-to-chunk mapping
            chunk_size_approx = max(1, seq_len // max(len(self._chunk_order), 1))
            token_start = evict_idx * chunk_size_approx
            token_end = min(token_start + chunk_size_approx, seq_len)

            if token_start < seq_len:
                score = attn[token_start:token_end].sum().item()
                self._heat_scores[chunk_key] = self._heat_scores.get(chunk_key, 0.0) + score

    @property
    def stats(self) -> Dict[str, int]:
        return {
            "total_chunks": len(self._chunk_order),
            "heat_tracked": len(self._heat_scores),
            "step_counter": self._step_counter,
            "pending_prefetch": len(self.prefetcher.pending_keys) if self.prefetcher else 0,
        }
