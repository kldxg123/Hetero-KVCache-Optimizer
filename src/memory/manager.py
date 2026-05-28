"""
src/memory/manager.py
======================
HeteroKVManager: A Tiered Storage System for KV Cache Management.

Presents KV cache as a heterogeneous memory hierarchy (HBM + DRAM) and provides
a unified abstraction for allocation, update, compression, and retrieval.
"""

import gc
import sys
import os
import subprocess
import math
from typing import Any, Dict, List, Optional, Tuple

import torch

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.quantization.kv_compressor import KVCompressor
from src.memory.dram_storage import DRAMStorageManager
from src.memory.attention_competition_queue import AttentionCompetitionQueue
from src.policy.prefetcher import AsyncPrefetcher
from src.policy.heavy_hitter import HeavyHitterOracle
from src.policy.adaptive_prefetch_controller import AdaptivePrefetchController
from src.core.scheduler import PredictivePrefetchScheduler


class HeteroKVManager:
    """
    Tiered KV Cache Storage System.

    Manages a heterogeneous memory hierarchy where hot KV tokens (Sink + recent Tail)
    reside in HBM, while overflow tokens are transparently compressed and offloaded
    to DRAM. This decouples the logical sequence length from the physical memory
    footprint, enabling O(1) steady-state memory for autoregressive decoding.

    Public API:
      - allocate(layer_idx, seq_len, budget_tokens)
      - update(layer_idx, key_states, value_states, mode, seq_offset)
      - compress(layer_idx, device='DRAM')
      - swap_in(layer_idx, chunk_key)
      - get_hbm_kv(layer_idx)
      - memory_summary()
    """

    def __init__(
        self,
        num_layers: int,
        sink_tokens: int = 64,
        hbm_budget_tokens: int = 8192,
        device: str = "cuda",
        enable_quant: bool = True,
        enable_prefetch: bool = True,
        group_size: int = 128,
        bits: int = 4,
        bandwidth_limiter=None,
        enable_method_d: bool = False,  # Query-aware retrieval (Method D)
        method_d_alpha: float = 1.0,  # 1.0 = pure query-aware, 0.0 = pure historical
        method_d_token_window: int = 0,
        method_d_score_reduce: str = "max",
        method_d_top_r: int = 8,
        method_d_consensus_boost: float = 0.0,
        method_d_min_position: int = 0,
        method_d_tail_guard_tokens: int = 0,
        method_d_focus_radius: int = 0,
        method_d_source_token_boost: float = 0.0,
        method_d_source_query_tokens: int = 64,
        method_d_require_source_overlap: bool = False,
        method_d_allow_source_before_min_position: bool = False,
        method_d_source_cue_focus: bool = False,
        method_d_source_cue_answer_tokens: int = 8,
        method_d_retrieve_focus_only: bool = False,
        method_d_retrieve_focus_context_tokens: int = 0,
        method_d_reuse_ttl_tokens: int = 0,
        method_d_reuse_source_threshold: float = 0.0,
        method_d_reuse_kv_cache: bool = False,
        method_d_triton_scoring: bool = False,
        method_d_triton_scoring_batch_chunks: int = 8,
        diagnostic_bf16_dram: bool = False,
    ):
        self.num_layers = num_layers
        self.sink_tokens = sink_tokens
        self.hbm_budget_tokens = hbm_budget_tokens
        self.device = device
        self.enable_quant = enable_quant
        self.group_size = group_size
        self.bits = bits
        self._bandwidth_limiter = bandwidth_limiter

        max_hbm = sink_tokens + hbm_budget_tokens

        # ════════════════════════════════════════════════════════════════
        # 三区域 HBM 分区 (用户设计：Sink + Tail + HeavyHitter)
        # ════════════════════════════════════════════════════════════════
        # Zone 1: Sink - 固定大小，系统提示 tokens
        self._sink_k: List[Optional[torch.Tensor]] = [None] * num_layers
        self._sink_v: List[Optional[torch.Tensor]] = [None] * num_layers

        # Zone 2: Tail - 固定大小，最近上下文 (滑动窗口)
        self._tail_k: List[Optional[torch.Tensor]] = [None] * num_layers
        self._tail_v: List[Optional[torch.Tensor]] = [None] * num_layers

        # Zone 3: HeavyHitter - 动态大小，高注意力 tokens
        self._heavyhitter_k: List[Optional[torch.Tensor]] = [None] * num_layers
        self._heavyhitter_v: List[Optional[torch.Tensor]] = [None] * num_layers
        self._heavyhitter_scores: List[Optional[torch.Tensor]] = [None] * num_layers
        self._heavyhitter_budget = max(hbm_budget_tokens // 2, 2048)

        # Logical token positions for short physical KV tensors.  These are
        # used by the attention wrapper to build a causal mask over non-
        # contiguous Sink/Tail/Heavy-Hitter tokens without padding to full length.
        self._sink_pos: List[Optional[torch.Tensor]] = [None] * num_layers
        self._tail_pos: List[Optional[torch.Tensor]] = [None] * num_layers
        self._heavyhitter_pos: List[Optional[torch.Tensor]] = [None] * num_layers
        self._key_positions: List[Optional[torch.Tensor]] = [None] * num_layers

        # Legacy compat: _key_cache = Sink + Tail + HeavyHitter (concatenated view)
        self._key_cache: List[Optional[torch.Tensor]] = [None] * num_layers
        self._value_cache: List[Optional[torch.Tensor]] = [None] * num_layers
        self._seq_offsets: List[int] = [0] * num_layers

        # 注意力竞争队列 (Tail驱逐 + 动态取回竞争 HeavyHitter HBM)
        self._competition_queue = AttentionCompetitionQueue()

        # Compression engine
        self._compressor = KVCompressor(group_size=group_size, bits=bits)

        # Heavy Hitter Oracle for attention-driven eviction
        self._oracle = HeavyHitterOracle(
            block_size=16,
            sink_tokens=sink_tokens,
            local_window=hbm_budget_tokens,
        )

        # Adaptive prefetch controller
        self._adaptive_controller = AdaptivePrefetchController()

        # Method D: Query-aware retrieval (HybridRetrievalStrategy)
        self._enable_method_d = enable_method_d
        self._method_d_token_window = int(method_d_token_window)
        self._method_d_consensus_boost = float(method_d_consensus_boost)
        self._method_d_min_position = int(method_d_min_position)
        self._method_d_tail_guard_tokens = int(method_d_tail_guard_tokens)
        self._method_d_focus_radius = max(0, int(method_d_focus_radius))
        self._method_d_source_token_boost = float(method_d_source_token_boost)
        self._method_d_source_query_tokens = max(1, int(method_d_source_query_tokens))
        self._method_d_require_source_overlap = bool(method_d_require_source_overlap)
        self._method_d_allow_source_before_min_position = bool(
            method_d_allow_source_before_min_position
        )
        self._method_d_source_cue_focus = bool(method_d_source_cue_focus)
        self._method_d_source_cue_answer_tokens = max(1, int(method_d_source_cue_answer_tokens))
        self._method_d_retrieve_focus_only = bool(method_d_retrieve_focus_only)
        self._method_d_retrieve_focus_context_tokens = max(
            0, int(method_d_retrieve_focus_context_tokens)
        )
        self._method_d_reuse_ttl_tokens = max(0, int(method_d_reuse_ttl_tokens))
        self._method_d_reuse_source_threshold = max(0.0, float(method_d_reuse_source_threshold))
        self._method_d_reuse_kv_cache = bool(method_d_reuse_kv_cache)
        self._method_d_triton_scoring = bool(method_d_triton_scoring)
        self._method_d_triton_scoring_batch_chunks = max(
            1, int(method_d_triton_scoring_batch_chunks)
        )
        self._method_d_reuse_cache: Dict[int, Dict[str, Any]] = {}
        self._method_d_range_votes: Dict[Tuple[int, int], int] = {}
        self._method_d_oracle_range: Optional[Tuple[int, int]] = None
        self._diagnostic_bf16_dram = bool(diagnostic_bf16_dram)
        self._method_d_retriever = None
        if enable_method_d:
            from src.memory.query_aware_retriever import HybridRetrievalStrategy
            self._method_d_retriever = HybridRetrievalStrategy(
                device=device,
                enable=True,
                alpha=method_d_alpha,
                fallback_to_method_c=True,
                score_reduce=method_d_score_reduce,
                top_r=method_d_top_r,
                use_triton_scoring=method_d_triton_scoring,
                triton_scoring_batch_chunks=method_d_triton_scoring_batch_chunks,
            )
            print(
                f"[Method D] Query-aware retrieval enabled | alpha={method_d_alpha} "
                f"triton_scoring={'ON' if method_d_triton_scoring else 'OFF'}"
            )

        # Async prefetcher for DRAM -> HBM overlap
        self._prefetcher: Optional[AsyncPrefetcher] = None
        if enable_prefetch and torch.cuda.is_available():
            self._prefetcher = AsyncPrefetcher(device=torch.device(device))

        # Predictive prefetch scheduler
        self._predictive_scheduler: Optional[PredictivePrefetchScheduler] = None

        # DRAM tier-2 storage
        self._dram = DRAMStorageManager()
        self._eviction_counter = 0

        # Adaptive self-healing: track chunk metadata for dynamic window retrieval
        self._chunk_eviction_order: List[str] = []  # Track eviction order
        self._chunk_attention_scores: Dict[str, float] = {}  # chunk_key -> avg score
        self._chunk_position_ranges: Dict[str, Tuple[int, int]] = {}
        self._last_retrieval_scores: Dict[str, float] = {}
        self._last_source_token_scores: Dict[str, float] = {}
        self._last_retrieved_positions: Dict[int, torch.Tensor] = {}
        self._last_retrieved_focus_mask: Dict[int, torch.Tensor] = {}
        self._last_method_d_selection: Dict[int, List[Dict[str, object]]] = {}
        self._source_token_ids: Optional[torch.Tensor] = None
        self._source_token_freq: Dict[int, int] = {}
        self._source_chunk_token_sets: Dict[str, set[int]] = {}
        self._source_cue_token_ids: List[List[int]] = []

        # ──────────────────────────────────────────────────────────────
        # Oracle 集成：存储最近的注意力权重
        # 用途：供 AdaptivePrefetchController 计算动态窗口 w_t
        # 时序：update_attention_scores() 存入 → get_dram_chunks_quantized_adaptive() 消费
        # ──────────────────────────────────────────────────────────────
        self._last_attention_weights: Optional[torch.Tensor] = None

    @property
    def _dram_table(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """Backward-compatible access to the DRAM storage dict."""
        return self._dram.table

    # ------------------------------------------------------------------
    # Tiered Storage API
    # ------------------------------------------------------------------

    def allocate(
        self,
        layer: int,
        budget: Optional[int] = None,
    ) -> bool:
        """
        Ensure the physical HBM pool for `layer` can accommodate up to
        `budget` tokens. If budget is None, uses hbm_budget_tokens.

        Returns True if the layer is ready for writes.
        """
        if layer < 0 or layer >= self.num_layers:
            raise ValueError(f"Invalid layer {layer}, num_layers={self.num_layers}")
        if budget is not None:
            self.hbm_budget_tokens = budget
        # Allocation is lazy in this implementation; pools are created on first update.
        # Future extensions could pre-allocate fixed-size tensors here for true zero-fragmentation.
        return True

    def update(
        self,
        layer_idx: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        mode: str = "prefill",
        seq_offset: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Unified update entry for both prefill and decode modes.

        Args:
            layer_idx: Transformer layer index.
            key_states:  New key tensor to integrate.
            value_states: New value tensor to integrate.
            mode:        Either "prefill" (seq_len > 1) or "decode" (seq_len == 1).
            seq_offset:  Logical sequence offset for this update (used for RoPE alignment).

        Returns:
            (key_states, value_states) that should be presented to the attention kernel.
            In both prefill and decode mode this returns the physically bounded
            HBM-resident short-KV view. Evicted tokens are stored in DRAM-side
            compressed chunks and must not remain as full FP16/BF16 KV in HBM.
        """
        if mode == "prefill":
            return self._prefill_update(layer_idx, key_states, value_states, seq_offset)
        elif mode == "decode":
            return self._decode_update(layer_idx, key_states, value_states, seq_offset)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def compress(self, layer: int, device: str = "DRAM") -> int:
        """
        Force-compress all HBM-resident tail tokens beyond sink_tokens for the given
        layer and move them to the specified device tier.

        Returns:
            Number of tokens compressed.
        """
        if device.upper() != "DRAM":
            raise NotImplementedError("Only DRAM compression is currently supported.")
        k_cache = self._key_cache[layer]
        v_cache = self._value_cache[layer]
        if k_cache is None or k_cache.shape[-2] <= self.sink_tokens:
            return 0

        overflow = k_cache.shape[-2] - self.sink_tokens
        k_sink = k_cache[..., : self.sink_tokens, :]
        v_sink = v_cache[..., : self.sink_tokens, :]
        k_tail = k_cache[..., self.sink_tokens :, :]
        v_tail = v_cache[..., self.sink_tokens :, :]

        self._evict_to_dram(layer, k_tail, v_tail)

        self._key_cache[layer] = k_sink
        self._value_cache[layer] = v_sink
        return overflow

    def swap_in_quantized(
        self, layer_idx: int, chunk_key: str
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        Swap a compressed chunk from DRAM back into HBM *without decompressing*.
        Returns the raw quantized tensors for use with fused_dequant_attn,
        eliminating the BF16 intermediate memory spike.

        Returns dict with keys: k_data, k_scales, k_zps, v_data, v_scales, v_zps
        (all on self.device), or None if chunk_key missing.
        """
        if chunk_key not in self._dram:
            return None

        entry = self._dram.retrieve(chunk_key)
        if entry is None:
            return None

        result = {}
        for key in ("k_data", "k_scales", "k_zps", "v_data", "v_scales", "v_zps"):
            t = entry.get(key)
            if t is not None:
                result[key] = t.to(self.device, non_blocking=True)
            else:
                return None

        if self._bandwidth_limiter is not None:
            for t in result.values():
                self._bandwidth_limiter.simulate_transfer(t)
        torch.cuda.synchronize(self.device)
        return result

    def swap_in(
        self, layer_idx: int, chunk_key: str
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Swap a compressed chunk from DRAM back into HBM.

        Returns:
            (restored_k, restored_v) as BF16 tensors, or None if chunk_key missing.
        """
        if not self._dram.contains(chunk_key):
            return None

        entry = self._dram.remove(chunk_key)

        # Fast path: prefetch hit
        if self._prefetcher is not None:
            result = self._prefetcher.fetch_if_ready(chunk_key)
            if result is not None:
                prefetched_k, _ = result
                q_v = entry["v_data"].to(self.device, non_blocking=True)
                s_v = entry["v_scales"].to(self.device, non_blocking=True)
                z_v = entry["v_zps"].to(self.device, non_blocking=True)
                if self._bandwidth_limiter is not None:
                    for t in (q_v, s_v, z_v):
                        self._bandwidth_limiter.simulate_transfer(t)
                torch.cuda.synchronize(self.device)
                restored_v = self._compressor.decompress(q_v, s_v, z_v).to(torch.bfloat16)
                return prefetched_k, restored_v

        # Slow path: synchronous decompress
        q_k = entry["k_data"].to(self.device, non_blocking=True)
        s_k = entry["k_scales"].to(self.device, non_blocking=True)
        z_k = entry["k_zps"].to(self.device, non_blocking=True)
        q_v = entry["v_data"].to(self.device, non_blocking=True)
        s_v = entry["v_scales"].to(self.device, non_blocking=True)
        z_v = entry["v_zps"].to(self.device, non_blocking=True)
        if self._bandwidth_limiter is not None:
            for t in (q_k, s_k, z_k, q_v, s_v, z_v):
                self._bandwidth_limiter.simulate_transfer(t)
        torch.cuda.synchronize(self.device)
        restored_k = self._compressor.decompress(q_k, s_k, z_k).to(torch.bfloat16)
        restored_v = self._compressor.decompress(q_v, s_v, z_v).to(torch.bfloat16)
        return restored_k, restored_v

    def get_hbm_kv(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the current HBM-resident KV tensors for the given layer."""
        if self._key_cache[layer_idx] is None:
            raise RuntimeError(f"Layer {layer_idx} has not been allocated yet.")
        return self._key_cache[layer_idx], self._value_cache[layer_idx]

    def get_seq_offset(self, layer_idx: int) -> int:
        """Return the logical sequence offset maintained for RoPE alignment."""
        return self._seq_offsets[layer_idx]

    def get_key_positions(self, layer_idx: int) -> Optional[torch.Tensor]:
        """Return logical positions for the current short physical KV tensor."""
        if layer_idx >= len(self._key_positions):
            return None
        return self._key_positions[layer_idx]

    def get_last_retrieved_positions(self, layer_idx: int) -> Optional[torch.Tensor]:
        """Return logical positions from the most recent DRAM retrieval."""
        return self._last_retrieved_positions.get(layer_idx)

    def get_last_retrieved_focus_mask(self, layer_idx: int) -> Optional[torch.Tensor]:
        """Return token-level focus mask aligned with the most recent retrieval."""
        return self._last_retrieved_focus_mask.get(layer_idx)

    def get_last_method_d_selection(self, layer_idx: int) -> List[Dict[str, object]]:
        """Return metadata for the most recent Method-D selected chunks."""
        return list(self._last_method_d_selection.get(layer_idx, []))

    def clear_method_d_reuse(self, layer_idx: int) -> None:
        """Drop sticky Method-D reuse state when a candidate fails the HBM gate."""
        self._method_d_reuse_cache.pop(layer_idx, None)

    def set_source_token_ids(self, token_ids: torch.Tensor) -> None:
        """Register source token ids for optional lexical source reranking."""
        ids = token_ids.detach().reshape(-1).cpu().long()
        self._source_token_ids = ids
        self._source_chunk_token_sets.clear()
        freq: Dict[int, int] = {}
        for token in ids.tolist():
            token = int(token)
            freq[token] = freq.get(token, 0) + 1
        self._source_token_freq = freq

    def set_source_cue_token_ids(
        self,
        cue_token_ids: List[List[int]],
        answer_tokens: Optional[int] = None,
    ) -> None:
        """Register non-oracle source cues whose following tokens are answer candidates."""
        cues: List[List[int]] = []
        for cue in cue_token_ids or []:
            cleaned = [int(token) for token in cue if int(token) >= 0]
            if cleaned:
                cues.append(cleaned)
        self._source_cue_token_ids = cues
        if answer_tokens is not None:
            self._method_d_source_cue_answer_tokens = max(1, int(answer_tokens))

    def _method_d_chunk_source_cue_score(self, chunk_key: str) -> float:
        """Return a non-oracle cue score when a chunk contains a registered cue."""
        if self._source_token_ids is None or not self._source_cue_token_ids:
            return 0.0
        start_pos, end_pos = self._chunk_position_ranges.get(chunk_key, (-1, -1))
        if start_pos < 0 or end_pos <= start_pos:
            return 0.0
        ids = self._source_token_ids
        start = max(0, min(int(start_pos), int(ids.numel())))
        end = max(start, min(int(end_pos), int(ids.numel())))
        if end <= start:
            return 0.0
        tokens = ids[start:end].tolist()
        best = 0.0
        for cue in self._source_cue_token_ids:
            n = len(cue)
            if n == 0 or n > len(tokens):
                continue
            for i in range(0, len(tokens) - n + 1):
                if tokens[i : i + n] == cue:
                    best = max(best, float(max(64, n * 8)))
                    break
        return best

    def _method_d_source_token_score(
        self,
        chunk_key: str,
        query_end: int,
    ) -> float:
        if self._source_token_ids is None or self._method_d_source_token_boost <= 0.0:
            return 0.0
        ids = self._source_token_ids
        total = int(ids.numel())
        if total == 0:
            return 0.0
        q_end = max(0, min(int(query_end), total))
        q_start = max(0, q_end - self._method_d_source_query_tokens)
        if q_end <= q_start:
            return 0.0
        max_common = max(16, int(total * 0.002))
        query_tokens = {
            int(token)
            for token in ids[q_start:q_end].tolist()
            if self._source_token_freq.get(int(token), 0) <= max_common
        }
        if not query_tokens:
            return 0.0
        chunk_tokens = self._source_chunk_token_sets.get(chunk_key)
        if chunk_tokens is None:
            start_pos, end_pos = self._chunk_position_ranges.get(chunk_key, (-1, -1))
            start = max(0, min(int(start_pos), total))
            end = max(start, min(int(end_pos), total))
            chunk_tokens = {
                int(token)
                for token in ids[start:end].tolist()
                if self._source_token_freq.get(int(token), 0) <= max_common
            }
            self._source_chunk_token_sets[chunk_key] = chunk_tokens
        overlap = query_tokens.intersection(chunk_tokens)
        if not overlap:
            return 0.0
        idf_sum = 0.0
        for token in overlap:
            freq = max(1, self._source_token_freq.get(token, 1))
            idf_sum += math.log((total + 1.0) / (freq + 1.0))
        return float(idf_sum)

    def _method_d_range_bin(self, start_pos: int, end_pos: int) -> Tuple[int, int]:
        """Coarse position bin used by optional consensus reranking."""
        width = max(1, int(self.hbm_budget_tokens // 4) or 2048)
        return (int(start_pos) // width, max(int(end_pos) - 1, int(start_pos)) // width)

    def _build_method_d_focus_mask(
        self,
        chunk_len: int,
        best_offset: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Build token-level focus mask around the dot-product best token."""
        mask = torch.zeros(int(chunk_len), dtype=torch.bool, device=device)
        if best_offset < 0 or mask.numel() == 0:
            return mask
        best = max(0, min(int(best_offset), mask.numel() - 1))
        radius = max(0, int(self._method_d_focus_radius))
        start = max(0, best - radius)
        end = min(mask.numel(), best + radius + 1)
        mask[start:end] = True
        return mask

    def _build_method_d_source_cue_focus_mask(
        self,
        chunk_pos: torch.Tensor,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """Focus on tokens immediately following registered source cues."""
        if (
            not self._method_d_source_cue_focus
            or self._source_token_ids is None
            or not self._source_cue_token_ids
            or chunk_pos.numel() == 0
        ):
            return None
        ids = self._source_token_ids
        total = int(ids.numel())
        pos_cpu = chunk_pos.detach().reshape(-1).cpu().long()
        chunk_ids: List[Optional[int]] = []
        for pos in pos_cpu.tolist():
            pos = int(pos)
            chunk_ids.append(int(ids[pos].item()) if 0 <= pos < total else None)
        if not chunk_ids:
            return None
        mask = torch.zeros(len(chunk_ids), dtype=torch.bool, device=device)
        answer_tokens = max(1, int(self._method_d_source_cue_answer_tokens))
        for cue in self._source_cue_token_ids:
            cue_len = len(cue)
            if cue_len == 0 or len(chunk_ids) < cue_len:
                continue
            for idx in range(0, len(chunk_ids) - cue_len + 1):
                if chunk_ids[idx : idx + cue_len] == cue:
                    start = idx + cue_len
                    end = min(len(chunk_ids), start + answer_tokens)
                    if start < end:
                        mask[start:end] = True
        return mask if bool(mask.any().item()) else None

    def _build_method_d_source_cue_retrieval_mask(
        self,
        chunk_pos: torch.Tensor,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """Return a retrieval mask that may include cue/context tokens."""
        if (
            not self._method_d_source_cue_focus
            or self._source_token_ids is None
            or not self._source_cue_token_ids
            or chunk_pos.numel() == 0
        ):
            return None
        ids = self._source_token_ids
        total = int(ids.numel())
        pos_cpu = chunk_pos.detach().reshape(-1).cpu().long()
        chunk_ids: List[Optional[int]] = []
        for pos in pos_cpu.tolist():
            pos = int(pos)
            chunk_ids.append(int(ids[pos].item()) if 0 <= pos < total else None)
        if not chunk_ids:
            return None
        mask = torch.zeros(len(chunk_ids), dtype=torch.bool, device=device)
        answer_tokens = max(1, int(self._method_d_source_cue_answer_tokens))
        context_tokens = max(0, int(self._method_d_retrieve_focus_context_tokens))
        for cue in self._source_cue_token_ids:
            cue_len = len(cue)
            if cue_len == 0 or len(chunk_ids) < cue_len:
                continue
            for idx in range(0, len(chunk_ids) - cue_len + 1):
                if chunk_ids[idx : idx + cue_len] == cue:
                    answer_start = idx + cue_len
                    end = min(len(chunk_ids), answer_start + answer_tokens)
                    start = max(0, answer_start - context_tokens)
                    if start < end:
                        mask[start:end] = True
        return mask if bool(mask.any().item()) else None

    def set_method_d_oracle_range(self, token_range: Optional[Tuple[int, int]]) -> None:
        """Diagnostic only: force Method-D to retrieve chunks covering this range."""
        if token_range is None:
            self._method_d_oracle_range = None
            return
        start, end = int(token_range[0]), int(token_range[1])
        self._method_d_oracle_range = (min(start, end), max(start, end))

    def active_hbm_tokens(self, layer_idx: Optional[int] = None) -> int:
        """Count currently active physical HBM KV tokens."""
        caches = self._key_cache if layer_idx is None else [self._key_cache[layer_idx]]
        return sum(k.shape[-2] for k in caches if k is not None)

    def force_shrink_hbm_budget(self, new_hbm_budget_tokens: int) -> None:
        """Diagnostic: shrink active Tail budget after prefill and evict overflow to DRAM."""
        self.hbm_budget_tokens = int(new_hbm_budget_tokens)
        tail_budget = max(0, self.hbm_budget_tokens - self.sink_tokens)
        for layer_idx in range(len(self._tail_k)):
            tail_k = self._tail_k[layer_idx]
            tail_v = self._tail_v[layer_idx]
            tail_pos = self._tail_pos[layer_idx]
            if tail_k is None or tail_v is None or tail_pos is None:
                continue
            tail_len = tail_k.shape[-2]
            if tail_len <= tail_budget:
                continue
            evict_count = tail_len - tail_budget
            evict_k = tail_k[:, :, :evict_count, :]
            evict_v = tail_v[:, :, :evict_count, :]
            evict_pos = tail_pos[:evict_count]
            if self.enable_quant and evict_count > 0:
                self._evict_to_dram(layer_idx, evict_k, evict_v, positions=evict_pos)
            self._tail_k[layer_idx] = tail_k[:, :, evict_count:, :].clone()
            self._tail_v[layer_idx] = tail_v[:, :, evict_count:, :].clone()
            self._tail_pos[layer_idx] = tail_pos[evict_count:].clone()
            self._update_legacy_cache(layer_idx)
            self._log_memory_state(
                layer_idx,
                f"[Diagnostic] Shrunk HBM Tail budget to {tail_budget}, evicted {evict_count} tokens to DRAM.",
            )

    def predict_physical_length_after_update(self, layer_idx: int, query_len: int) -> int:
        """Best-effort mask-size prediction before Cache.update is called."""
        tail_budget = max(0, self.hbm_budget_tokens - self.sink_tokens)
        current_hh = 0
        if layer_idx < len(self._heavyhitter_k) and self._heavyhitter_k[layer_idx] is not None:
            current_hh = self._heavyhitter_k[layer_idx].shape[-2]
        if layer_idx >= len(self._sink_k) or self._sink_k[layer_idx] is None:
            return min(query_len, self.sink_tokens + tail_budget) + current_hh
        old_tail = self._tail_k[layer_idx].shape[-2] if self._tail_k[layer_idx] is not None else 0
        return self._sink_k[layer_idx].shape[-2] + min(tail_budget, old_tail + query_len) + current_hh

    def update_attention_scores(
        self,
        attention_weights: torch.Tensor,
        key_positions: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Phase D: Feed attention weights from the latest decode step to the
        HeavyHitterOracle for cumulative importance tracking.

        Also stores the weights for AdaptivePrefetchController to compute
        dynamic window w_t based on attention volatility σ(A_t).
        """
        if key_positions is not None and key_positions.numel() == attention_weights.numel():
            pos = key_positions.detach().to(attention_weights.device).long().reshape(-1)
            weights = attention_weights.detach().reshape(-1).float()
            max_pos = int(pos.max().item()) + 1 if pos.numel() else 0
            if self._oracle.token_scores is None or self._oracle.token_scores.shape[0] < max_pos:
                new_scores = torch.zeros(max_pos, dtype=torch.float32, device=weights.device)
                if self._oracle.token_scores is not None:
                    old = self._oracle.token_scores.to(weights.device)
                    new_scores[: old.shape[0]] = old
                self._oracle.token_scores = new_scores
            self._oracle.token_scores.index_add_(0, pos, weights)
        else:
            self._oracle.update(attention_weights)
        # Store for adaptive controller (copy to avoid detachment issues)
        self._last_attention_weights = attention_weights.detach().clone()

    def get_oracle_scores(self) -> Optional[torch.Tensor]:
        """Return current cumulative attention scores (for debugging/testing)."""
        return self._oracle.token_scores

    def schedule_prefetch(self, chunk_key: str) -> None:
        """Submit an asynchronous DRAM -> HBM prefetch task."""
        if self._prefetcher is None or not self._dram.contains(chunk_key):
            return
        entry = self._dram.retrieve(chunk_key)
        prefetch_entry = {
            "k_data": entry["k_data"],
            "k_scales": entry["k_scales"],
            "k_zps": entry["k_zps"],
            "v_data": entry["v_data"],
            "v_scales": entry["v_scales"],
            "v_zps": entry["v_zps"],
        }
        self._prefetcher.submit_prefetch_task(chunk_key, prefetch_entry, self._compressor)

    def predictive_prefetch_step(
        self,
        current_chunk: Optional[str] = None,
        attention_weights: Optional[torch.Tensor] = None,
        cache_miss: bool = False,
    ) -> List[str]:
        """
        Proactive prefetch: predict which DRAM chunks are needed next
        and submit async H2D transfers ahead of time.

        Integrates AdaptivePrefetchController for dynamic window sizing.
        """
        if self._prefetcher is None:
            return []

        # Lazy-init the predictive scheduler
        if self._predictive_scheduler is None and self._dram.num_entries > 0:
            self._predictive_scheduler = PredictivePrefetchScheduler(
                prefetcher=self._prefetcher,
                dram_table=self._dram.table,
                compressor=self._compressor,
            )

        if self._predictive_scheduler is not None:
            # AdaptivePrefetchController: dynamically adjust lookahead window
            new_window = self._adaptive_controller.compute_window(
                attention_weights=attention_weights,
                cache_miss=cache_miss,
            )
            self._predictive_scheduler.lookahead_window = new_window

            return self._predictive_scheduler.schedule_step(
                current_chunk=current_chunk,
                attention_weights=attention_weights,
            )
        return []

    def memory_summary(self) -> Dict[str, Any]:
        """Return a snapshot of HBM and DRAM consumption."""
        hbm_tokens = 0
        for k in self._key_cache:
            if k is not None:
                hbm_tokens += k.shape[-2]

        dram_summary = self._dram.memory_summary()

        return {
            "hbm_tokens": hbm_tokens,
            "dram_entries": dram_summary["num_entries"],
            "dram_bytes": dram_summary["total_bytes"],
            "max_hbm_tokens": self.max_hbm_tokens(),
        }

    def max_hbm_tokens(self) -> int:
        """
        返回HBM中能存储的最大token数（三区域架构）

        总HBM预算 = Sink + Tail + HeavyHitter
        = sink_tokens + hbm_budget_tokens + heavyhitter_budget
        """
        return self.sink_tokens + self.hbm_budget_tokens + self._heavyhitter_budget

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prefill_update(
        self,
        layer_idx: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        seq_offset: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        三区域架构 prefill 更新逻辑（分段预填充）

        流程：
        1. 提取Sink（开头64个tokens）→ Sink HBM分区
        2. 提取Tail（结尾2048个tokens）→ Tail HBM分区
        3. 中间tokens → 压缩到DRAM
        4. 初始化HeavyHitter分区（prefill阶段为空，后续通过竞争队列填充）
        5. 返回 Sink + Tail + HeavyHitter（初始为空）

        注意：Prefill阶段没有注意力分数，HeavyHitter分区在decode阶段动态填充
        """
        new_len = key_states.shape[-2]
        positions = torch.arange(
            seq_offset,
            seq_offset + new_len,
            dtype=torch.long,
            device=key_states.device,
        )

        # 初始化三个分区（如果需要）
        while len(self._sink_k) <= layer_idx:
            self._sink_k.append(None)
            self._sink_v.append(None)
            self._tail_k.append(None)
            self._tail_v.append(None)
            self._heavyhitter_k.append(None)
            self._heavyhitter_v.append(None)
            self._heavyhitter_scores.append(None)
            self._sink_pos.append(None)
            self._tail_pos.append(None)
            self._heavyhitter_pos.append(None)
            self._key_positions.append(None)
            self._key_cache.append(None)
            self._value_cache.append(None)
            self._seq_offsets.append(0)

        # ════════════════════════════════════════════════════════════════
        # 增量预填充：如果Sink已有数据，说明这是后续chunk
        # 保留Sink，将新chunk追加到Tail，超量部分驱逐到DRAM
        # ════════════════════════════════════════════════════════════════
        if self._sink_k[layer_idx] is not None:
            return self._incremental_prefill_update(
                layer_idx, key_states, value_states, seq_offset
            )

        # ════════════════════════════════════════════════════════════════
        # Step 1: 提取Sink（开头固定tokens）
        # ════════════════════════════════════════════════════════════════
        sink_amt = min(new_len, self.sink_tokens)

        if sink_amt > 0:
            self._sink_k[layer_idx] = key_states[..., :sink_amt, :].clone()
            self._sink_v[layer_idx] = value_states[..., :sink_amt, :].clone()
            self._sink_pos[layer_idx] = positions[:sink_amt].clone()
        else:
            self._sink_k[layer_idx] = torch.empty(
                key_states.shape[0], key_states.shape[1], 0, key_states.shape[3],
                device=key_states.device, dtype=key_states.dtype
            )
            self._sink_v[layer_idx] = torch.empty(
                value_states.shape[0], value_states.shape[1], 0, value_states.shape[3],
                device=value_states.device, dtype=value_states.dtype
            )
            self._sink_pos[layer_idx] = torch.empty(0, device=key_states.device, dtype=torch.long)

        # ════════════════════════════════════════════════════════════════
        # Step 2: 提取Tail（结尾固定tokens，滑动窗口）
        # ════════════════════════════════════════════════════════════════
        tail_budget = self.hbm_budget_tokens - self.sink_tokens
        tail_amt = min(new_len - sink_amt, tail_budget)

        if tail_amt > 0:
            self._tail_k[layer_idx] = key_states[..., -tail_amt:, :].clone()
            self._tail_v[layer_idx] = value_states[..., -tail_amt:, :].clone()
            self._tail_pos[layer_idx] = positions[-tail_amt:].clone()
        else:
            self._tail_k[layer_idx] = torch.empty(
                key_states.shape[0], key_states.shape[1], 0, key_states.shape[3],
                device=key_states.device, dtype=key_states.dtype
            )
            self._tail_v[layer_idx] = torch.empty(
                value_states.shape[0], value_states.shape[1], 0, value_states.shape[3],
                device=value_states.device, dtype=value_states.dtype
            )
            self._tail_pos[layer_idx] = torch.empty(0, device=key_states.device, dtype=torch.long)

        # ════════════════════════════════════════════════════════════════
        # Step 3: 中间tokens → 压缩到DRAM
        # ════════════════════════════════════════════════════════════════
        body_start = sink_amt
        body_end = new_len - tail_amt

        if self.enable_quant and body_end > body_start:
            # 提取中间tokens
            body_k = key_states[..., body_start:body_end, :]
            body_v = value_states[..., body_start:body_end, :]
            body_pos = positions[body_start:body_end]

            # 压缩并存储到DRAM
            self._evict_to_dram(layer_idx, body_k, body_v, positions=body_pos)

        # ════════════════════════════════════════════════════════════════
        # Step 4: 初始化HeavyHitter分区（初始为空）
        # ════════════════════════════════════════════════════════════════
        # Prefill阶段没有注意力分数，HeavyHitter初始为空
        # 在decode阶段通过竞争队列动态填充
        self._heavyhitter_k[layer_idx] = torch.empty(
            key_states.shape[0], key_states.shape[1], 0, key_states.shape[3],
            device=key_states.device, dtype=key_states.dtype
        )
        self._heavyhitter_v[layer_idx] = torch.empty(
            value_states.shape[0], value_states.shape[1], 0, value_states.shape[3],
            device=value_states.device, dtype=value_states.dtype
        )
        self._heavyhitter_scores[layer_idx] = torch.empty(
            0, device=key_states.device, dtype=torch.float32
        )
        self._heavyhitter_pos[layer_idx] = torch.empty(
            0, device=key_states.device, dtype=torch.long
        )

        # ════════════════════════════════════════════════════════════════
        # Step 5: 更新legacy cache
        # ════════════════════════════════════════════════════════════════
        self._update_legacy_cache(layer_idx)
        self._seq_offsets[layer_idx] = seq_offset + new_len

        self._log_memory_state(
            layer_idx,
            f"Processed chunk [{seq_offset}:{seq_offset + new_len}], returned truncated K/V, Memory sustained.",
        )
        return self._key_cache[layer_idx], self._value_cache[layer_idx]

    def _incremental_prefill_update(
        self,
        layer_idx: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        seq_offset: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        增量预填充：处理后续chunk。

        保留Sink（首个chunk的开头tokens），将新chunk追加到Tail。
        如果Tail超过预算，驱逐旧Tail开头tokens到DRAM。
        返回 Sink + old_tail + new_chunk 以保证注意力计算正确。
        """
        new_len = key_states.shape[-2]
        tail_budget = self.hbm_budget_tokens - self.sink_tokens
        positions = torch.arange(
            seq_offset,
            seq_offset + new_len,
            dtype=torch.long,
            device=key_states.device,
        )

        old_tail_k = self._tail_k[layer_idx]
        old_tail_v = self._tail_v[layer_idx]
        old_tail_pos = self._tail_pos[layer_idx]

        # Combine old Tail + current chunk
        combined_k = torch.cat([old_tail_k, key_states], dim=-2)
        combined_v = torch.cat([old_tail_v, value_states], dim=-2)
        combined_pos = torch.cat([old_tail_pos, positions], dim=0)

        # Evict excess from the beginning of combined Tail → DRAM
        combined_len = combined_k.shape[-2]
        if combined_len > tail_budget:
            evict_count = combined_len - tail_budget
            evicted_k = combined_k[:, :, :evict_count, :]
            evicted_v = combined_v[:, :, :evict_count, :]
            evicted_pos = combined_pos[:evict_count]

            if self.enable_quant and evict_count > 0:
                self._evict_to_dram(layer_idx, evicted_k, evicted_v, positions=evicted_pos)

            # Keep last tail_budget tokens
            self._tail_k[layer_idx] = combined_k[:, :, evict_count:, :].clone()
            self._tail_v[layer_idx] = combined_v[:, :, evict_count:, :].clone()
            self._tail_pos[layer_idx] = combined_pos[evict_count:].clone()
        else:
            self._tail_k[layer_idx] = combined_k
            self._tail_v[layer_idx] = combined_v
            self._tail_pos[layer_idx] = combined_pos

        # Update legacy cache and return
        self._update_legacy_cache(layer_idx)
        self._seq_offsets[layer_idx] = seq_offset + new_len

        self._log_memory_state(
            layer_idx,
            f"Processed chunk [{seq_offset}:{seq_offset + new_len}], returned truncated K/V, Memory sustained.",
        )
        return self._key_cache[layer_idx], self._value_cache[layer_idx]

    def _decode_update(
        self,
        layer_idx: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        seq_offset: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        三区域架构 decode 更新逻辑

        流程：
        1. 新token → Tail
        2. 如果Tail满 → 驱逐Tail开头tokens → 加入竞争队列
        3. 处理竞争队列 → top-K → HeavyHitter HBM分区
        4. 如果HeavyHitter满 → 驱逐低分数tokens → DRAM
        5. 返回 Sink + Tail + HeavyHitter
        """
        # 初始化三个分区（如果需要）
        while len(self._sink_k) <= layer_idx:
            self._sink_k.append(None)
            self._sink_v.append(None)
            self._tail_k.append(None)
            self._tail_v.append(None)
            self._heavyhitter_k.append(None)
            self._heavyhitter_v.append(None)
            self._heavyhitter_scores.append(None)
            self._sink_pos.append(None)
            self._tail_pos.append(None)
            self._heavyhitter_pos.append(None)
            self._key_positions.append(None)
            self._key_cache.append(None)
            self._value_cache.append(None)
            self._seq_offsets.append(0)

        # Step 1: 新token添加到Tail
        tail_budget = self.hbm_budget_tokens - self.sink_tokens
        new_pos = torch.tensor([seq_offset], dtype=torch.long, device=key_states.device)

        if self._tail_k[layer_idx] is None:
            # 第一次写入：初始化Tail
            self._tail_k[layer_idx] = key_states.clone()
            self._tail_v[layer_idx] = value_states.clone()
            self._tail_pos[layer_idx] = new_pos.clone()
        else:
            tail_len = self._tail_k[layer_idx].shape[-2]

            if tail_len < tail_budget:
                # Tail未满：直接添加
                self._tail_k[layer_idx] = torch.cat([self._tail_k[layer_idx], key_states], dim=-2)
                self._tail_v[layer_idx] = torch.cat([self._tail_v[layer_idx], value_states], dim=-2)
                self._tail_pos[layer_idx] = torch.cat([self._tail_pos[layer_idx], new_pos], dim=0)
            else:
                # ═══════════════════════════════════════════════════════════
                # Tail满：驱逐Tail开头tokens → 竞争队列
                # ═══════════════════════════════════════════════════════════
                evicted_k = self._tail_k[layer_idx][:, :, :1, :]
                evicted_v = self._tail_v[layer_idx][:, :, :1, :]
                evicted_pos = self._tail_pos[layer_idx][:1]

                # 获取驱逐tokens的注意力分数
                if self._oracle.token_scores is not None:
                    pos_idx = evicted_pos.to(self._oracle.token_scores.device).long()
                    if (pos_idx < self._oracle.token_scores.shape[0]).any():
                        evicted_score = self._oracle.token_scores[
                            pos_idx.clamp_max(self._oracle.token_scores.shape[0] - 1)
                        ]
                    else:
                        evicted_score = torch.tensor([1.0], device=self.device)
                else:
                    evicted_score = torch.tensor([1.0], device=self.device)

                # 压缩并加入竞争队列
                if self.enable_quant:
                    k_data, k_scales, k_zps = self._compressor.compress(evicted_k)
                    v_data, v_scales, v_zps = self._compressor.compress(evicted_v)
                    self._competition_queue.enqueue(
                        k=evicted_k, v=evicted_v, scores=evicted_score,
                        compressed={'k_data': k_data, 'k_scales': k_scales, 'k_zps': k_zps,
                                'v_data': v_data, 'v_scales': v_scales, 'v_zps': v_zps},
                        layer_idx=layer_idx, prefix=f"tail_evict", positions=evicted_pos
                    )
                else:
                    self._competition_queue.enqueue(
                        k=evicted_k, v=evicted_v, scores=evicted_score,
                        compressed=None, layer_idx=layer_idx, prefix=f"tail_evict", positions=evicted_pos
                    )

                # 滑动Tail窗口：移除开头，添加新token到末尾
                self._tail_k[layer_idx] = torch.cat([self._tail_k[layer_idx][:, :, 1:, :], key_states], dim=-2)
                self._tail_v[layer_idx] = torch.cat([self._tail_v[layer_idx][:, :, 1:, :], value_states], dim=-2)
                self._tail_pos[layer_idx] = torch.cat([self._tail_pos[layer_idx][1:], new_pos], dim=0)

        # Step 2: 处理竞争队列
        self._process_competition_queue(layer_idx)

        # Step 3: 更新legacy _key_cache (Sink + Tail + HeavyHitter)
        self._update_legacy_cache(layer_idx)
        self._seq_offsets[layer_idx] = seq_offset + 1

        # 返回用于attention的KV (Sink + Tail + HeavyHitter)
        return self._key_cache[layer_idx], self._value_cache[layer_idx]

    def _get_current_seq_length(self) -> int:
        """获取当前序列长度（用于oracle分数索引）"""
        total = 0
        for layer in range(self.num_layers):
            if self._sink_k[layer] is not None:
                total += self._sink_k[layer].shape[-2]
            if self._tail_k[layer] is not None:
                total += self._tail_k[layer].shape[-2]
        return total

    def _process_competition_queue(self, layer_idx: int):
        """
        处理注意力竞争队列

        逻辑：
        1. 从队列取top-K tokens
        2. 尝试加入HeavyHitter HBM分区
        3. 如果超过预算，驱逐低分数tokens
        """
        # 计算HeavyHitter当前占用
        current_hh_len = 0
        if self._heavyhitter_k[layer_idx] is not None:
            current_hh_len = self._heavyhitter_k[layer_idx].shape[-2]

        available_budget = self._heavyhitter_budget - current_hh_len

        if available_budget > 0:
            # 从竞争队列取top-K tokens
            top_k, top_v, top_scores, top_pos = self._competition_queue.dequeue_top_k_with_positions(available_budget)

            if top_k is not None:
                # 加入HeavyHitter分区
                if self._heavyhitter_k[layer_idx] is None:
                    self._heavyhitter_k[layer_idx] = top_k
                    self._heavyhitter_v[layer_idx] = top_v
                    self._heavyhitter_scores[layer_idx] = top_scores
                    self._heavyhitter_pos[layer_idx] = top_pos
                else:
                    self._heavyhitter_k[layer_idx] = torch.cat([self._heavyhitter_k[layer_idx], top_k], dim=-2)
                    self._heavyhitter_v[layer_idx] = torch.cat([self._heavyhitter_v[layer_idx], top_v], dim=-2)
                    self._heavyhitter_scores[layer_idx] = torch.cat([self._heavyhitter_scores[layer_idx], top_scores], dim=-1)
                    self._heavyhitter_pos[layer_idx] = torch.cat([self._heavyhitter_pos[layer_idx], top_pos], dim=0)

        # 如果HeavyHitter仍超过预算，驱逐低分数tokens到DRAM
        if self._heavyhitter_k[layer_idx] is not None:
            hh_len = self._heavyhitter_k[layer_idx].shape[-2]

            if hh_len > self._heavyhitter_budget:
                # 驱除多余的tokens
                num_evict = hh_len - self._heavyhitter_budget

                # 根据分数排序，驱逐最低分的tokens
                if self._heavyhitter_scores[layer_idx] is not None:
                    _, low_score_indices = torch.topk(
                        self._heavyhitter_scores[layer_idx],
                        k=num_evict,
                        largest=False
                    )
                else:
                    low_score_indices = torch.arange(num_evict, device=self.device)

                evicted_k = self._heavyhitter_k[layer_idx][:, :, low_score_indices, :]
                evicted_v = self._heavyhitter_v[layer_idx][:, :, low_score_indices, :]
                evicted_pos = self._heavyhitter_pos[layer_idx][low_score_indices]

                # 压缩并驱逐到DRAM
                if self.enable_quant:
                    self._evict_to_dram(layer_idx, evicted_k, evicted_v, positions=evicted_pos)

                # 保留剩余的高分数tokens
                keep_mask = torch.ones(hh_len, dtype=torch.bool, device=self.device)
                keep_mask[low_score_indices] = False

                self._heavyhitter_k[layer_idx] = self._heavyhitter_k[layer_idx][:, :, keep_mask, :]
                self._heavyhitter_v[layer_idx] = self._heavyhitter_v[layer_idx][:, :, keep_mask, :]
                self._heavyhitter_scores[layer_idx] = self._heavyhitter_scores[layer_idx][keep_mask]
                self._heavyhitter_pos[layer_idx] = self._heavyhitter_pos[layer_idx][keep_mask]

    def _update_legacy_cache(self, layer_idx: int):
        """更新legacy _key_cache 以保持兼容性"""
        # 用sink的shape作为参考，构造正确维度的空张量
        ref = self._sink_k[layer_idx]
        if ref is not None:
            s = ref.shape
            empty_k = torch.empty(s[0], s[1], 0, s[3], device=self.device, dtype=ref.dtype)
            empty_v = torch.empty(s[0], s[1], 0, s[3], device=self.device, dtype=ref.dtype)
        elif self._tail_k[layer_idx] is not None:
            s = self._tail_k[layer_idx].shape
            empty_k = torch.empty(s[0], s[1], 0, s[3], device=self.device, dtype=self._tail_k[layer_idx].dtype)
            empty_v = torch.empty(s[0], s[1], 0, s[3], device=self.device, dtype=self._tail_k[layer_idx].dtype)
        else:
            return

        empty_pos = torch.empty(0, device=empty_k.device, dtype=torch.long)
        sink_k = self._sink_k[layer_idx] if self._sink_k[layer_idx] is not None else empty_k
        tail_k = self._tail_k[layer_idx] if self._tail_k[layer_idx] is not None else empty_k
        hh_k = self._heavyhitter_k[layer_idx] if self._heavyhitter_k[layer_idx] is not None else empty_k

        self._key_cache[layer_idx] = torch.cat([sink_k, tail_k, hh_k], dim=-2)
        sink_pos = self._sink_pos[layer_idx] if self._sink_pos[layer_idx] is not None else empty_pos
        tail_pos = self._tail_pos[layer_idx] if self._tail_pos[layer_idx] is not None else empty_pos
        hh_pos = self._heavyhitter_pos[layer_idx] if self._heavyhitter_pos[layer_idx] is not None else empty_pos
        self._key_positions[layer_idx] = torch.cat([sink_pos, tail_pos, hh_pos], dim=0)

        sink_v = self._sink_v[layer_idx] if self._sink_v[layer_idx] is not None else empty_v
        tail_v = self._tail_v[layer_idx] if self._tail_v[layer_idx] is not None else empty_v
        hh_v = self._heavyhitter_v[layer_idx] if self._heavyhitter_v[layer_idx] is not None else empty_v

        self._value_cache[layer_idx] = torch.cat([sink_v, tail_v, hh_v], dim=-2)

    def decompress_dram_chunks(
        self, layer_idx: int
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], int]:
        """
        Self-healing: decompress ALL DRAM chunks for a specific layer to BF16.
        Non-destructive — entries remain in DRAM for future use.
        Only decompresses chunks whose key matches the given layer_idx.
        """
        prefix = f"l{layer_idx}_"
        dram_keys = [k for k in self._dram.table.keys() if k.startswith(prefix)]
        if not dram_keys:
            return None, None, 0

        dram_k_parts, dram_v_parts = [], []
        for chunk_key in dram_keys:
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
            except Exception:
                continue

        if not dram_k_parts:
            return None, None, 0

        dram_k = torch.cat(dram_k_parts, dim=-2)
        dram_v = torch.cat(dram_v_parts, dim=-2)
        token_count = dram_k.shape[-2]

        return dram_k, dram_v, token_count

    def decompress_dram_chunks_adaptive(
        self,
        layer_idx: int,
        window_size: Optional[int] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], int]:
        """
        TRUE adaptive self-healing: decompress only TOP-W DRAM chunks based on attention scores.

        This implements the dynamic window w_t described in the paper:
          w_t = w_min + (σ_t / σ_ref - 1) · α + β · miss_rate_t

        Args:
            layer_idx: Transformer layer index
            window_size: Number of chunks to retrieve. If None, uses AdaptivePrefetchController.

        Returns:
            (dram_k, dram_v, retrieved_count) or (None, None, 0) if no DRAM data

        Memory Impact:
          - Retrieves only top-w_t chunks (NOT all chunks)
          - HBM spike = O(w_t * chunk_size), NOT O(total_chunks)
          - This is the BOUNDED O(w_t) transient promised in the paper

        Recall Impact:
          - If needle token is outside top-w_t chunks: RETRIEVAL FAILURE
          - Expected NIAH recall: roughly w_t / total_chunks (NOT 100%)
        """
        prefix = f"l{layer_idx}_"
        dram_keys = [k for k in self._dram.table.keys() if k.startswith(prefix)]

        if not dram_keys:
            return None, None, 0

        # Compute adaptive window size using AdaptivePrefetchController
        if window_size is None:
            # Reuse the same controller for self-healing window sizing
            w_t = int(self._adaptive_controller.compute_window(attention_weights=self._last_attention_weights, cache_miss=False))
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

                restored_k = self._compressor.decompress(q_k, s_k, z_k, target_dtype=torch.bfloat16)
                restored_v = self._compressor.decompress(q_v, s_v, z_v, target_dtype=torch.bfloat16)

                dram_k_parts.append(restored_k)
                dram_v_parts.append(restored_v)
            except Exception as e:
                print(f"  [Warning] Failed to decompress {chunk_key}: {e}")
                continue

        if not dram_k_parts:
            return None, None, 0

        # Concatenate in original eviction order (preserve temporal sequence)
        selected_keys_sorted = sorted(selected_keys, key=lambda k: self._chunk_eviction_order.index(k))

        # Re-sort parts to match temporal order
        key_to_part_k = dict(zip(selected_keys, dram_k_parts))
        key_to_part_v = dict(zip(selected_keys, dram_v_parts))

        dram_k_parts_sorted = [key_to_part_k[k] for k in selected_keys_sorted if k in key_to_part_k]
        dram_v_parts_sorted = [key_to_part_v[k] for k in selected_keys_sorted if k in key_to_part_v]

        dram_k = torch.cat(dram_k_parts_sorted, dim=-2)
        dram_v = torch.cat(dram_v_parts_sorted, dim=-2)
        token_count = dram_k.shape[-2]

        # Calculate HBM memory spike
        spike_mb = dram_k.element_size() * dram_k.nelement() / 1024 / 1024
        spike_mb += dram_v.element_size() * dram_v.nelement() / 1024 / 1024

        print(
            f"  [Adaptive Self-Healing] Retrieved {token_count} tokens | "
            f"HBM_spike={spike_mb:.1f}MB (vs {len(dram_keys)*2048*16/1024/1024:.1f}MB full)"
        )

        return dram_k, dram_v, token_count

    def get_dram_chunks_quantized_adaptive(
        self,
        layer_idx: int,
        window_size: Optional[int] = None,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        TRUE dynamic window + Triton kernel: Return 4-bit quantized chunks (no decompression).

        This is the CORRECT implementation where dynamic window and Triton kernel
        work TOGETHER:
          1. AdaptivePrefetchController computes w_t based on σ(A_t)
          2. Select top-w_t chunks by attention score
          3. Return 4-bit quantized data (NO BF16 decompression)
          4. Triton fused kernel computes attention directly on 4-bit data

        Memory Impact:
          - HBM spike: O(w_t * chunk_size) for 4-bit data transfer
          - NO BF16 decompression spike (eliminated!)
          - This is the paper's "eliminates 512MB transient" promise

        Args:
            layer_idx: Transformer layer index
            window_size: Number of chunks to retrieve. If None, uses adaptive controller.

        Returns:
            Dict with k_data, k_scales, k_zps, v_data, v_scales, v_zps (all 4-bit)
            Or None if no DRAM data
        """
        prefix = f"l{layer_idx}_"
        dram_keys = [k for k in self._dram.table.keys() if k.startswith(prefix)]

        if not dram_keys:
            return None

        # Compute adaptive window size
        if window_size is None:
            w_t = int(self._adaptive_controller.compute_window(attention_weights=self._last_attention_weights, cache_miss=False))
            window_size = max(1, min(w_t, len(dram_keys)))

        # Rank chunks by attention score (heavy hitters first)
        ranked_chunks = sorted(
            dram_keys,
            key=lambda k: self._chunk_attention_scores.get(k, 0.0),
            reverse=True
        )

        # Select only top-w_t chunks
        selected_keys = ranked_chunks[:window_size]

        print(
            f"  [Triton-Optimized Adaptive Self-Healing] layer={layer_idx} | "
            f"total_chunks={len(dram_keys)} window={window_size} | "
            f"retrieving={len(selected_keys)} chunks (4-bit, NO decompression) | "
            f"coverage={100.0*len(selected_keys)/len(dram_keys):.1f}%"
        )

        # Collect 4-bit quantized data (NO decompression!)
        all_k_data, all_k_scales, all_k_zps = [], [], []
        all_v_data, all_v_scales, all_v_zps = [], [], []

        for chunk_key in selected_keys:
            entry = self._dram.retrieve(chunk_key)
            if entry is None:
                continue

            # Transfer to GPU but KEEP 4-bit format!
            all_k_data.append(entry["k_data"].to(self.device, non_blocking=True))
            all_k_scales.append(entry["k_scales"].to(self.device, non_blocking=True))
            all_k_zps.append(entry["k_zps"].to(self.device, non_blocking=True))
            all_v_data.append(entry["v_data"].to(self.device, non_blocking=True))
            all_v_scales.append(entry["v_scales"].to(self.device, non_blocking=True))
            all_v_zps.append(entry["v_zps"].to(self.device, non_blocking=True))

        if not all_k_data:
            return None

        # Concatenate in original eviction order
        selected_keys_sorted = sorted(selected_keys, key=lambda k: self._chunk_eviction_order.index(k))
        key_to_idx = {k: i for i, k in enumerate(dram_keys)}
        sorted_indices = [key_to_idx[k] for k in selected_keys_sorted if k in key_to_idx]

        # Handle concatenation based on tensor dimensions
        # For data tensors (3D: [batch, heads, seq] or 4D): concatenate along dim=-2 (sequence dim)
        k_data = torch.cat([all_k_data[i] for i in sorted_indices], dim=-2)
        v_data = torch.cat([all_v_data[i] for i in sorted_indices], dim=-2)

        # For scales/zp tensors: determine dimension based on shape
        if all_k_scales[0].ndim >= 2:
            # 2D+ tensors: concatenate along sequence dimension (dim=-2)
            k_scales = torch.cat([all_k_scales[i] for i in sorted_indices], dim=-2)
            k_zps = torch.cat([all_k_zps[i] for i in sorted_indices], dim=-2)
            v_scales = torch.cat([all_v_scales[i] for i in sorted_indices], dim=-2)
            v_zps = torch.cat([all_v_zps[i] for i in sorted_indices], dim=-2)
        else:
            # 1D tensors: concatenate along dim=0 (the only dimension)
            k_scales = torch.cat([all_k_scales[i] for i in sorted_indices], dim=0)
            k_zps = torch.cat([all_k_zps[i] for i in sorted_indices], dim=0)
            v_scales = torch.cat([all_v_scales[i] for i in sorted_indices], dim=0)
            v_zps = torch.cat([all_v_zps[i] for i in sorted_indices], dim=0)

        # Calculate memory savings
        tokens_4bit = k_data.shape[-2]
        bf16_spike_mb = tokens_4bit * 2 * 32 * 2 / 1024 / 1024  # Would be BF16 size
        bit4_mb = (
            k_data.element_size() * k_data.nelement() +
            k_scales.element_size() * k_scales.nelement() +
            k_zps.element_size() * k_zps.nelement() +
            v_data.element_size() * v_data.nelement() +
            v_scales.element_size() * v_scales.nelement() +
            v_zps.element_size() * v_zps.nelement()
        ) / 1024 / 1024

        print(
            f"  [Triton-Optimized] {tokens_4bit} tokens in 4-bit | "
            f"HMB={bit4_mb:.1f}MB (vs {bf16_spike_mb:.1f}MB if BF16) | "
            f"saved={bf16_spike_mb - bit4_mb:.1f}MB"
        )

        return {
            "k_data": k_data,
            "k_scales": k_scales,
            "k_zps": k_zps,
            "v_data": v_data,
            "v_scales": v_scales,
            "v_zps": v_zps,
        }

    def decompress_dram_chunks_method_d(
        self,
        layer_idx: int,
        query_key: torch.Tensor,
        top_k: Optional[int] = None,
        score_only: bool = False,
        use_last_selection: bool = False,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], int, str]:
        """
        Method D: Query-aware DRAM chunk retrieval.

        Uses token-level Query x Key dot-product scoring against quantized
        DRAM keys to select the most relevant chunks for retrieval.

        This is an INDEPENDENT method that does NOT affect the existing
        adaptive (Method C) retrieval logic.

        Args:
            layer_idx: Transformer layer index
            query_key: Current query K tensor [batch, heads, 1, head_dim]
            top_k: Number of chunks to retrieve. If None, uses adaptive window.

        Returns:
            (dram_k, dram_v, retrieved_count, method_used)
            method_used: "method_d" or "method_c_fallback"
        """
        if self._method_d_retriever is None:
            return None, None, 0, "disabled"
        self._last_retrieved_positions.pop(layer_idx, None)
        self._last_retrieved_focus_mask.pop(layer_idx, None)
        if not use_last_selection:
            self._last_method_d_selection.pop(layer_idx, None)
            self._last_source_token_scores = {}

        prefix = f"l{layer_idx}_"
        dram_keys = [k for k in self._dram.table.keys() if k.startswith(prefix)]

        if not dram_keys:
            return None, None, 0, "no_dram"

        # Compute adaptive window size
        if top_k is None:
            w_t = int(self._adaptive_controller.compute_window(
                attention_weights=self._last_attention_weights, cache_miss=False
            ))
            top_k = max(1, min(w_t, len(dram_keys)))

        oracle_range = self._method_d_oracle_range
        reuse_hit = False
        if use_last_selection:
            selection_items = list(self._last_method_d_selection.get(layer_idx, []))
            selected_keys = [
                str(item.get("chunk_key"))
                for item in selection_items
                if item.get("chunk_key") in dram_keys
            ]
            best_offsets = {
                str(item.get("chunk_key")): int(item.get("best_token_offset", -1))
                for item in selection_items
                if item.get("chunk_key") in dram_keys
            }
            reuse_hit = any(bool(item.get("reuse_hit")) for item in selection_items)
            method_base = (
                str(selection_items[0].get("method_used", "method_d"))
                if selection_items else "method_d"
            )
            method_used = f"{method_base}_materialize"
        elif oracle_range is not None:
            oracle_start, oracle_end = oracle_range
            selected_keys = []
            best_offsets = {}
            for key in dram_keys:
                start_pos, end_pos = self._chunk_position_ranges.get(key, (-1, -1))
                if start_pos < oracle_end and end_pos > oracle_start:
                    selected_keys.append(key)
                    offset = max(0, min(oracle_start - start_pos, max(end_pos - start_pos - 1, 0)))
                    best_offsets[key] = int(offset)
            selected_keys = sorted(
                selected_keys,
                key=lambda k: self._chunk_eviction_order.index(k)
                if k in self._chunk_eviction_order else 0,
            )[:top_k]
            method_used = "oracle_range"
            self._last_retrieval_scores = {key: 1.0e30 for key in selected_keys}
        else:
            reuse_state = self._method_d_reuse_cache.get(layer_idx)
            if (
                self._method_d_reuse_ttl_tokens > 0
                and reuse_state
                and int(reuse_state.get("ttl_remaining", 0)) > 0
            ):
                cached_keys = [
                    key for key in reuse_state.get("selected_keys", [])
                    if key in dram_keys
                ]
                if cached_keys:
                    selected_keys = cached_keys[:top_k]
                    scores = reuse_state.get("scores", {})
                    source_scores = reuse_state.get("source_scores", {})
                    best_offsets = {
                        key: int(reuse_state.get("best_offsets", {}).get(key, -1))
                        for key in selected_keys
                    }
                    self._last_retrieval_scores = {
                        key: float(scores.get(key, float("nan")))
                        for key in selected_keys
                    }
                    self._last_source_token_scores = {
                        key: float(source_scores.get(key, 0.0))
                        for key in selected_keys
                    }
                    reuse_state["ttl_remaining"] = int(reuse_state.get("ttl_remaining", 0)) - 1
                    method_used = f"{reuse_state.get('method_used', 'method_d')}_reuse"
                    reuse_hit = True
                else:
                    self._method_d_reuse_cache.pop(layer_idx, None)

            if not reuse_hit:
                # Use Method D to rank and select chunks.
                selected_keys, method_used = self._method_d_retriever.retrieve_chunks(
                    query_key=query_key,
                    candidate_keys=dram_keys,
                    top_k=top_k,
                    historical_scores=self._chunk_attention_scores,
                    dram_table=self._dram.table,
                    compressor=self._compressor,
                )
                self._last_retrieval_scores = dict(
                    getattr(self._method_d_retriever.query_aware_retriever, "last_scores", {})
                )
                best_offsets = dict(
                    getattr(
                        self._method_d_retriever.query_aware_retriever,
                        "last_best_token_offsets",
                        {},
                    )
                )
                if (
                    self._method_d_consensus_boost
                    or self._method_d_min_position > 0
                    or self._method_d_tail_guard_tokens > 0
                    or self._method_d_source_token_boost > 0.0
                ):
                    adjusted_scores = {}
                    source_positive_scores = {}
                    current_end = self._seq_offsets[layer_idx] if layer_idx < len(self._seq_offsets) else 0
                    tail_guard_start = None
                    if self._method_d_tail_guard_tokens > 0:
                        if current_end > 0:
                            tail_guard_start = current_end - self._method_d_tail_guard_tokens
                    for key in dram_keys:
                        if key not in self._last_retrieval_scores:
                            continue
                        start_pos, end_pos = self._chunk_position_ranges.get(key, (-1, -1))
                        source_score = self._method_d_source_token_score(key, current_end)
                        if self._method_d_allow_source_before_min_position:
                            source_score = max(
                                source_score,
                                self._method_d_chunk_source_cue_score(key),
                            )
                        source_bypasses_min = (
                            self._method_d_allow_source_before_min_position
                            and source_score > 0.0
                        )
                        if (
                            self._method_d_min_position > 0
                            and start_pos < self._method_d_min_position
                            and not source_bypasses_min
                        ):
                            continue
                        if tail_guard_start is not None and start_pos >= tail_guard_start:
                            continue
                        bin_key = self._method_d_range_bin(start_pos, end_pos)
                        vote_bonus = self._method_d_consensus_boost * math.log1p(
                            self._method_d_range_votes.get(bin_key, 0)
                        )
                        self._last_source_token_scores[key] = source_score
                        source_bonus = self._method_d_source_token_boost * source_score
                        adjusted_scores[key] = (
                            float(self._last_retrieval_scores[key]) + vote_bonus + source_bonus
                        )
                        if source_score > 0.0:
                            source_positive_scores[key] = adjusted_scores[key]
                    if (
                        self._method_d_require_source_overlap
                        and self._method_d_source_token_boost > 0.0
                        and source_positive_scores
                    ):
                        adjusted_scores = source_positive_scores
                        method_used = f"{method_used}_source_filtered"
                    selected_keys = [
                        key for key, _ in sorted(
                            adjusted_scores.items(), key=lambda item: item[1], reverse=True
                        )[:top_k]
                    ]
                    if selected_keys:
                        method_used = f"{method_used}_consensus"
                        self._last_retrieval_scores = adjusted_scores
            for key in selected_keys:
                start_pos, end_pos = self._chunk_position_ranges.get(key, (-1, -1))
                bin_key = self._method_d_range_bin(start_pos, end_pos)
                self._method_d_range_votes[bin_key] = self._method_d_range_votes.get(bin_key, 0) + 1
            if (
                not reuse_hit
                and self._method_d_reuse_ttl_tokens > 0
                and selected_keys
            ):
                best_source_score = max(
                    float(self._last_source_token_scores.get(key, 0.0))
                    for key in selected_keys
                )
                if best_source_score >= self._method_d_reuse_source_threshold:
                    self._method_d_reuse_cache[layer_idx] = {
                        "selected_keys": list(selected_keys),
                        "best_offsets": {
                            key: int(best_offsets.get(key, -1)) for key in selected_keys
                        },
                        "scores": {
                            key: float(self._last_retrieval_scores.get(key, float("nan")))
                            for key in selected_keys
                        },
                        "source_scores": {
                            key: float(self._last_source_token_scores.get(key, 0.0))
                            for key in selected_keys
                        },
                        "method_used": method_used,
                        "ttl_remaining": self._method_d_reuse_ttl_tokens,
                    }
        self._last_method_d_selection[layer_idx] = [
            {
                "chunk_key": key,
                "range": list(self._chunk_position_ranges.get(key, (-1, -1))),
                "score": float(self._last_retrieval_scores.get(key, float("nan"))),
                "source_token_score": float(self._last_source_token_scores.get(key, 0.0)),
                "best_token_offset": int(best_offsets.get(key, -1)),
                "scoring_backend": getattr(
                    self._method_d_retriever.query_aware_retriever,
                    "last_scoring_backend",
                    "unknown",
                ),
                "reuse_hit": bool(reuse_hit),
                "reuse_ttl_remaining": int(
                    self._method_d_reuse_cache.get(layer_idx, {}).get("ttl_remaining", 0)
                ),
                "range_vote": int(
                    self._method_d_range_votes.get(
                        self._method_d_range_bin(*self._chunk_position_ranges.get(key, (-1, -1))),
                        0,
                    )
                ),
                "method_used": method_used,
            }
            for key in selected_keys
        ] if not use_last_selection else self._last_method_d_selection.get(layer_idx, [])

        if not selected_keys:
            return None, None, 0, method_used

        if score_only:
            for item in self._last_method_d_selection.get(layer_idx, []):
                item["deferred_dequant"] = True
            return None, None, len(selected_keys), method_used

        target_dtype = query_key.dtype if query_key.is_floating_point() else torch.bfloat16
        reuse_state = self._method_d_reuse_cache.get(layer_idx)
        if self._method_d_reuse_kv_cache and reuse_hit and reuse_state:
            cached = reuse_state.get("retrieved_kv_cache")
            if (
                cached
                and cached.get("selected_keys") == list(selected_keys)
                and cached.get("dtype") == str(target_dtype)
                and cached.get("device") == str(self.device)
            ):
                self._last_retrieved_positions[layer_idx] = cached["positions"]
                if cached.get("focus_mask") is not None:
                    self._last_retrieved_focus_mask[layer_idx] = cached["focus_mask"]
                token_count = int(cached["k"].shape[-2])
                for item in self._last_method_d_selection.get(layer_idx, []):
                    item["retrieved_kv_cache_hit"] = True
                return cached["k"], cached["v"], token_count, f"{method_used}_kv_cache"

        # Decompress selected chunks to BF16
        dram_k_parts, dram_v_parts, dram_pos_parts, dram_focus_parts = [], [], [], []
        for chunk_key in selected_keys:
            entry = self._dram.retrieve(chunk_key)
            if entry is None:
                continue
            try:
                if self._diagnostic_bf16_dram and "k_fp" in entry and "v_fp" in entry:
                    restored_k = entry["k_fp"].to(
                        self.device, dtype=target_dtype, non_blocking=True
                    )
                    restored_v = entry["v_fp"].to(
                        self.device, dtype=target_dtype, non_blocking=True
                    )
                else:
                    q_k = entry["k_data"].to(self.device, non_blocking=True)
                    s_k = entry["k_scales"].to(self.device, non_blocking=True)
                    z_k = entry["k_zps"].to(self.device, non_blocking=True)
                    q_v = entry["v_data"].to(self.device, non_blocking=True)
                    s_v = entry["v_scales"].to(self.device, non_blocking=True)
                    z_v = entry["v_zps"].to(self.device, non_blocking=True)
                    restored_k = self._compressor.decompress(
                        q_k, s_k, z_k, target_dtype=target_dtype
                    )
                    restored_v = self._compressor.decompress(
                        q_v, s_v, z_v, target_dtype=target_dtype
                    )

                if "positions" in entry:
                    chunk_pos = entry["positions"].to(self.device, non_blocking=True).long()
                else:
                    start_pos, end_pos = self._chunk_position_ranges.get(
                        chunk_key, (0, restored_k.shape[-2])
                    )
                    chunk_pos = torch.arange(start_pos, end_pos, dtype=torch.long, device=self.device)

                focus_mask = self._build_method_d_focus_mask(
                    chunk_len=restored_k.shape[-2],
                    best_offset=int(best_offsets.get(chunk_key, -1)),
                    device=restored_k.device,
                )
                focus_source = "dot_best"
                cue_focus_mask = self._build_method_d_source_cue_focus_mask(
                    chunk_pos=chunk_pos,
                    device=restored_k.device,
                )
                if cue_focus_mask is not None and cue_focus_mask.any():
                    focus_mask = cue_focus_mask
                    focus_source = "source_cue"
                if (
                    self._method_d_retrieve_focus_only
                    and focus_source == "source_cue"
                    and focus_mask.numel() == restored_k.shape[-2]
                    and bool(focus_mask.any().item())
                ):
                    retrieval_mask = self._build_method_d_source_cue_retrieval_mask(
                        chunk_pos=chunk_pos,
                        device=restored_k.device,
                    )
                    if retrieval_mask is None or retrieval_mask.numel() != focus_mask.numel():
                        retrieval_mask = focus_mask
                    retrieval_indices = torch.nonzero(
                        retrieval_mask, as_tuple=False
                    ).reshape(-1)
                    restored_k = restored_k.index_select(-2, retrieval_indices)
                    restored_v = restored_v.index_select(-2, retrieval_indices)
                    chunk_pos = chunk_pos.index_select(0, retrieval_indices)
                    focus_mask = focus_mask.index_select(0, retrieval_indices)
                    for item in self._last_method_d_selection.get(layer_idx, []):
                        if item.get("chunk_key") == chunk_key:
                            item["focus_only_retrieval"] = True
                            item["focus_context_tokens"] = int(
                                self._method_d_retrieve_focus_context_tokens
                            )
                            item["focus_source"] = focus_source
                            item["focus_token_count"] = int(focus_mask.sum().item())
                            item["retrieved_focus_token_count"] = int(
                                restored_k.shape[-2]
                            )
                            if chunk_pos.numel() > 0:
                                item["retrieved_range"] = [
                                    int(chunk_pos[0].item()),
                                    int(chunk_pos[-1].item()) + 1,
                                ]
                token_window = self._method_d_token_window
                if token_window > 0 and restored_k.shape[-2] > token_window:
                    focus_indices = torch.nonzero(focus_mask, as_tuple=False).reshape(-1)
                    if focus_indices.numel() > 0:
                        best_offset = int(focus_indices[focus_indices.numel() // 2].item())
                    else:
                        best_offset = int(best_offsets.get(chunk_key, 0))
                    best_offset = max(0, min(best_offset, restored_k.shape[-2] - 1))
                    half_window = max(1, token_window // 2)
                    start = max(0, best_offset - half_window)
                    end = min(restored_k.shape[-2], start + token_window)
                    start = max(0, end - token_window)
                    restored_k = restored_k[:, :, start:end, :]
                    restored_v = restored_v[:, :, start:end, :]
                    chunk_pos = chunk_pos[start:end]
                    focus_mask = focus_mask[start:end]
                    for item in self._last_method_d_selection.get(layer_idx, []):
                        if item.get("chunk_key") == chunk_key:
                            item["token_window"] = [int(start), int(end)]
                            item["focus_source"] = focus_source
                            if focus_mask.numel() > 0:
                                item["focus_token_count"] = int(focus_mask.sum().item())
                            if chunk_pos.numel() > 0:
                                item["retrieved_range"] = [
                                    int(chunk_pos[0].item()),
                                    int(chunk_pos[-1].item()) + 1,
                                ]
                else:
                    for item in self._last_method_d_selection.get(layer_idx, []):
                        if item.get("chunk_key") == chunk_key:
                            item["focus_source"] = focus_source
                            if focus_mask.numel() > 0:
                                item["focus_token_count"] = int(focus_mask.sum().item())

                dram_k_parts.append(restored_k)
                dram_v_parts.append(restored_v)
                dram_pos_parts.append(chunk_pos)
                dram_focus_parts.append(focus_mask)
            except Exception:
                continue

        if not dram_k_parts:
            return None, None, 0, method_used

        # Sort by temporal order for correct attention computation
        selected_keys_sorted = sorted(
            selected_keys,
            key=lambda k: self._chunk_eviction_order.index(k) if k in self._chunk_eviction_order else 0
        )
        key_to_part_k = dict(zip(selected_keys, dram_k_parts))
        key_to_part_v = dict(zip(selected_keys, dram_v_parts))
        key_to_part_pos = dict(zip(selected_keys, dram_pos_parts))
        key_to_part_focus = dict(zip(selected_keys, dram_focus_parts))

        dram_k_parts_sorted = [key_to_part_k[k] for k in selected_keys_sorted if k in key_to_part_k]
        dram_v_parts_sorted = [key_to_part_v[k] for k in selected_keys_sorted if k in key_to_part_v]
        dram_pos_parts_sorted = [key_to_part_pos[k] for k in selected_keys_sorted if k in key_to_part_pos]
        dram_focus_parts_sorted = [
            key_to_part_focus[k] for k in selected_keys_sorted if k in key_to_part_focus
        ]

        dram_k = torch.cat(dram_k_parts_sorted, dim=-2)
        dram_v = torch.cat(dram_v_parts_sorted, dim=-2)
        self._last_retrieved_positions[layer_idx] = torch.cat(dram_pos_parts_sorted, dim=0)
        if dram_focus_parts_sorted:
            self._last_retrieved_focus_mask[layer_idx] = torch.cat(
                dram_focus_parts_sorted, dim=0
            )
        else:
            self._last_retrieved_focus_mask.pop(layer_idx, None)
        token_count = dram_k.shape[-2]

        reuse_state = self._method_d_reuse_cache.get(layer_idx)
        if (
            self._method_d_reuse_kv_cache
            and
            reuse_state is not None
            and self._method_d_reuse_ttl_tokens > 0
            and selected_keys
        ):
            reuse_state["retrieved_kv_cache"] = {
                "selected_keys": list(selected_keys),
                "dtype": str(target_dtype),
                "device": str(self.device),
                "k": dram_k.detach(),
                "v": dram_v.detach(),
                "positions": self._last_retrieved_positions[layer_idx].detach(),
                "focus_mask": self._last_retrieved_focus_mask.get(layer_idx),
            }
            if reuse_state["retrieved_kv_cache"]["focus_mask"] is not None:
                reuse_state["retrieved_kv_cache"]["focus_mask"] = (
                    reuse_state["retrieved_kv_cache"]["focus_mask"].detach()
                )
            for item in self._last_method_d_selection.get(layer_idx, []):
                item["retrieved_kv_cache_hit"] = False

        spike_mb = dram_k.element_size() * dram_k.nelement() / 1024 / 1024
        spike_mb += dram_v.element_size() * dram_v.nelement() / 1024 / 1024

        print(
            f"  [Method D] layer={layer_idx} | method={method_used} | "
            f"selected={len(selected_keys)}/{len(dram_keys)} chunks | "
            f"tokens={token_count} | HBM_spike={spike_mb:.1f}MB"
        )
        if selected_keys_sorted:
            best_key = selected_keys[0]
            best_score = self._last_retrieval_scores.get(best_key, float("nan"))
            best_range = self._chunk_position_ranges.get(best_key, (-1, -1))
            print(
                f"  Query matched with DRAM chunk {best_key} via 4-bit Dot Product, "
                f"Score: {best_score:.6f}"
            )
            print(
                "  [Self-Healing] Query triggered retrieval of chunks from DRAM "
                "to SRAM via Dot-Product Scoring."
            )
            print(f"  Retrieved chunk range: [{best_range[0]}:{best_range[1]}]")

        return dram_k, dram_v, token_count, method_used

    def _get_dram_kv_quantized(
        self, layer_idx: int
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        Get DRAM KV data in 4-bit quantized format (no decompression).

        Used by Triton fused kernel to avoid BF16 intermediate allocation.

        Returns:
            Dict with keys: k_data, k_scales, k_zps, v_data, v_scales, v_zps
            Or None if no DRAM data
        """
        prefix = f"l{layer_idx}_"
        dram_keys = [k for k in self._dram.table.keys() if k.startswith(prefix)]

        if not dram_keys:
            return None

        all_k_data, all_k_scales, all_k_zps = [], [], []
        all_v_data, all_v_scales, all_v_zps = [], [], []

        for chunk_key in dram_keys:
            entry = self._dram.retrieve(chunk_key)
            if entry is None:
                continue

            all_k_data.append(entry["k_data"])
            all_k_scales.append(entry["k_scales"])
            all_k_zps.append(entry["k_zps"])
            all_v_data.append(entry["v_data"])
            all_v_scales.append(entry["v_scales"])
            all_v_zps.append(entry["v_zps"])

        if not all_k_data:
            return None

        # Concatenate in eviction order
        sorted_keys = sorted(dram_keys, key=lambda k: self._chunk_eviction_order.index(k))

        key_to_idx = {k: i for i, k in enumerate(dram_keys)}
        sorted_indices = [key_to_idx[k] for k in sorted_keys if k in key_to_idx]

        k_data = torch.cat([all_k_data[i] for i in sorted_indices], dim=-2)
        k_scales = torch.cat([all_k_scales[i] for i in sorted_indices], dim=-2)
        k_zps = torch.cat([all_k_zps[i] for i in sorted_indices], dim=-2)
        v_data = torch.cat([all_v_data[i] for i in sorted_indices], dim=-2)
        v_scales = torch.cat([all_v_scales[i] for i in sorted_indices], dim=-2)
        v_zps = torch.cat([all_v_zps[i] for i in sorted_indices], dim=-2)

        return {
            "k_data": k_data,
            "k_scales": k_scales,
            "k_zps": k_zps,
            "v_data": v_data,
            "v_scales": v_scales,
            "v_zps": v_zps,
        }

    def count_dram_tokens(self, layer_idx: int = 0) -> int:
        """Count total tokens in DRAM for a specific layer."""
        prefix = f"l{layer_idx}_"
        dram_keys = [k for k in self._dram.table.keys() if k.startswith(prefix)]
        if not dram_keys:
            return 0
        entry = self._dram.retrieve(dram_keys[0])
        if entry is None:
            return 0
        k = entry["k_data"]
        tokens_per_chunk = k.shape[-2] if k.dim() >= 2 else 1
        return len(dram_keys) * tokens_per_chunk

    def _current_process_nvidia_smi_mb(self) -> Optional[int]:
        """Return current process GPU memory from nvidia-smi when available."""
        try:
            proc = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,used_memory",
                    "--format=csv,noheader,nounits",
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=2,
            )
        except Exception:
            return None
        pid = str(os.getpid())
        for raw in proc.stdout.splitlines():
            parts = [p.strip() for p in raw.split(",")]
            if len(parts) >= 2 and parts[0] == pid:
                try:
                    return int(parts[1])
                except ValueError:
                    return None
        return None

    def _log_memory_state(self, layer_idx: int, message: str) -> None:
        """Emit mechanism and memory evidence for the first layer."""
        if layer_idx != 0:
            return
        active_len = 0
        if layer_idx < len(self._key_cache) and self._key_cache[layer_idx] is not None:
            active_len = self._key_cache[layer_idx].shape[-2]
        dram_tokens = self.count_dram_tokens(layer_idx)
        allocated = reserved = None
        if torch.cuda.is_available() and str(self.device).startswith("cuda"):
            try:
                allocated = torch.cuda.max_memory_allocated(self.device) / 1024 ** 3
                reserved = torch.cuda.max_memory_reserved(self.device) / 1024 ** 3
            except Exception:
                allocated = reserved = None
        smi_mb = self._current_process_nvidia_smi_mb()
        print(f"  {message}")
        print(f"  Active HBM KV length: {active_len}")
        print(f"  DRAM compressed KV length: {dram_tokens}")
        if allocated is not None:
            print(f"  torch.cuda.max_memory_allocated: {allocated:.2f} GB")
            print(f"  torch.cuda.max_memory_reserved: {reserved:.2f} GB")
        if smi_mb is not None:
            print(f"  nvidia-smi process memory: {smi_mb / 1024:.2f} GB")

    def _evict_to_dram(
        self,
        layer_idx: int,
        k_chunk: torch.Tensor,
        v_chunk: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Compress a KV chunk and move it to DRAM via DRAMStorageManager (pinned CPU memory).

        Enhanced: tracks chunk metadata for adaptive self-healing (attention scores, eviction order).
        """
        chunk_key = f"l{layer_idx}_e{self._eviction_counter}"

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
        if positions is not None:
            entry["positions"] = positions.detach().to("cpu")
        if self._diagnostic_bf16_dram:
            entry["k_fp"] = k_chunk.detach().to("cpu", dtype=torch.bfloat16)
            entry["v_fp"] = v_chunk.detach().to("cpu", dtype=torch.bfloat16)

        # DRAMStorageManager handles .cpu().pin_memory() internally
        self._dram.store_entry(chunk_key, entry)

        # Track metadata for adaptive self-healing
        self._chunk_eviction_order.append(chunk_key)
        if positions is not None and positions.numel() > 0:
            start_pos = int(positions[0].item())
            end_pos = int(positions[-1].item()) + 1
        else:
            start_pos = self._eviction_counter * k_chunk.shape[-2]
            end_pos = start_pos + k_chunk.shape[-2]
        self._chunk_position_ranges[chunk_key] = (start_pos, end_pos)

        # Compute and store chunk attention score (average of oracle scores for tokens in this chunk)
        if self._oracle.token_scores is not None:
            start_token_idx = self._eviction_counter * k_chunk.shape[-2]
            end_token_idx = start_token_idx + k_chunk.shape[-2]
            chunk_scores = self._oracle.token_scores[start_token_idx:end_token_idx]
            chunk_avg_score = float(chunk_scores.mean().item()) if len(chunk_scores) > 0 else 0.0
            self._chunk_attention_scores[chunk_key] = chunk_avg_score
        else:
            # Fallback: use FIFO order as score (older chunks = lower score)
            chunk_avg_score = 1.0 / (self._eviction_counter + 1)
            self._chunk_attention_scores[chunk_key] = chunk_avg_score

        # Method D: Register chunk embedding for query-aware retrieval
        if self._method_d_retriever is not None:
            self._method_d_retriever.register_chunk(
                chunk_key=chunk_key,
                start_pos=start_pos,
                end_pos=end_pos,
                historical_attention=chunk_avg_score,
            )

        if self._bandwidth_limiter is not None:
            stored = self._dram.retrieve(chunk_key)
            if stored is not None:
                for t in stored.values():
                    self._bandwidth_limiter.simulate_transfer(t)

        if layer_idx == 0:
            tokens = k_chunk.shape[-2]
            self._eviction_counter += 1
            print(
                f"  [Evict->DRAM] layer=0 chunk={chunk_key} "
                f"range=[{start_pos}:{end_pos}] tokens={tokens} "
                f"score={self._chunk_attention_scores[chunk_key]:.4f} "
                f"DRAM_entries={self._dram.num_entries}"
            )


if __name__ == "__main__":
    manager = HeteroKVManager(num_layers=4, sink_tokens=64, hbm_budget_tokens=256)
    for layer in range(4):
        manager.allocate(layer, budget=512)

    # Prefill simulation
    k = torch.randn(1, 8, 512, 128, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(1, 8, 512, 128, dtype=torch.bfloat16, device="cuda")
    out_k, out_v = manager.update(0, k, v, mode="prefill")
    print("Prefill output shape:", out_k.shape, out_v.shape)
    print("HBM resident shape:", manager.get_hbm_kv(0)[0].shape)
    print("Memory summary:", manager.memory_summary())

    # Decode simulation
    k1 = torch.randn(1, 8, 1, 128, dtype=torch.bfloat16, device="cuda")
    v1 = torch.randn(1, 8, 1, 128, dtype=torch.bfloat16, device="cuda")
    out_k, out_v = manager.update(0, k1, v1, mode="decode")
    print("Decode output shape:", out_k.shape)
