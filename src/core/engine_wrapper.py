"""
src/core/engine_wrapper.py
==========================
Hetero-KVCache Engine Wrapper.

Presents a Tiered Storage System abstraction for long-context MLLM inference.
The FusedHeteroCache is a thin HuggingFace DynamicCache adapter backed by
HeteroKVManager, which handles transient interception, 4-bit quantization,
and zero-fragmentation in-place rolling.
"""

import gc
import os
import torch
from typing import Any, Dict, Optional, Tuple
from transformers.cache_utils import DynamicCache

from src.memory.manager import HeteroKVManager

# Optional Triton fused operator (falls back to standard matmul if unavailable)
try:
    from src.quantization.kernels.fused_dequant_attn import (
        fused_dequant_attention,
        fused_dequant_attn_forward,
        fused_dequant_attn_decode,
    )
    _TRITON_AVAILABLE = True
except Exception:
    _TRITON_AVAILABLE = False


class FusedHeteroCache(DynamicCache):
    """
    HF-compatible DynamicCache subclass backed by HeteroKVManager.

    Short-KV Architecture:
      - During chunked prefill, the manager physically truncates returned KV to
        bounded Sink + Tail + Heavy-Hitter state.
      - During decode, DRAM chunks can be selected by token-level Query x Key
        scoring without materializing the full 128K KV in HBM.
    """

    def __init__(
        self,
        num_layers: Optional[int] = None,
        sink_tokens: int = 64,
        keep_tail: int = 8192,
        chunk_size: int = 2048,
        device: str = "cuda:0",
        group_size: int = 128,
        enable_quant: bool = True,
        enable_prefetch: bool = True,
        enable_triton: bool = False,
        bandwidth_limiter=None,
        self_healing: bool = True,
        adaptive_self_healing: bool = False,
        enable_method_d: bool = False,  # Query-aware retrieval
        method_d_alpha: float = 1.0,  # 1.0 = pure query-aware, 0.0 = pure historical
        method_d_gate_margin: float = 1.10,
        method_d_token_window: int = 0,
        method_d_layer_min: int = 0,
        method_d_layer_max: Optional[int] = None,
        method_d_top_k: Optional[int] = None,
        method_d_retrieval_bias: float = 0.0,
        method_d_score_reduce: str = "max",
        method_d_top_r: int = 8,
        method_d_query_history_tokens: int = 1,
        method_d_consensus_boost: float = 0.0,
        method_d_min_position: int = 0,
        method_d_tail_guard_tokens: int = 0,
        method_d_focus_radius: int = 0,
        method_d_source_token_boost: float = 0.0,
        method_d_source_query_tokens: int = 64,
        method_d_require_source_overlap: bool = False,
        method_d_allow_source_before_min_position: bool = False,
        method_d_focus_bias: float = 0.0,
        method_d_nonfocus_penalty: float = 0.0,
        method_d_source_fusion_alpha: float = 0.0,
        method_d_source_fusion_low_alpha: float = 0.0,
        method_d_source_fusion_source_threshold: float = 0.0,
        method_d_source_fusion_focus_only: bool = False,
        method_d_source_cue_focus: bool = False,
        method_d_source_cue_answer_tokens: int = 8,
        method_d_retrieve_focus_only: bool = False,
        method_d_retrieve_focus_context_tokens: int = 0,
        method_d_reuse_ttl_tokens: int = 0,
        method_d_reuse_source_threshold: float = 0.0,
        method_d_source_gate_bypass_threshold: float = 0.0,
        method_d_reuse_gate_bypass: bool = False,
        method_d_reuse_kv_cache: bool = False,
        method_d_triton_scoring: bool = False,
        method_d_triton_scoring_batch_chunks: int = 8,
        diagnostic_bf16_dram: bool = False,
    ):
        super().__init__()

        self.sink_tokens = sink_tokens
        self.keep_tail = keep_tail
        self.chunk_size = chunk_size
        self.device = device
        self.enable_quant = enable_quant
        self.enable_triton = enable_triton and _TRITON_AVAILABLE
        self.enable_prefetch = enable_prefetch
        self.group_size = group_size
        self._bandwidth_limiter = bandwidth_limiter
        self.self_healing = self_healing
        self.adaptive_self_healing = adaptive_self_healing
        self.enable_method_d = enable_method_d
        self.method_d_alpha = method_d_alpha
        self.method_d_gate_margin = method_d_gate_margin
        self.method_d_token_window = int(method_d_token_window)
        self.method_d_layer_min = int(method_d_layer_min)
        self.method_d_layer_max = None if method_d_layer_max is None else int(method_d_layer_max)
        self.method_d_top_k = None if method_d_top_k is None else int(method_d_top_k)
        self.method_d_retrieval_bias = float(method_d_retrieval_bias)
        self.method_d_score_reduce = str(method_d_score_reduce)
        self.method_d_top_r = int(method_d_top_r)
        self.method_d_query_history_tokens = max(1, int(method_d_query_history_tokens))
        self.method_d_consensus_boost = float(method_d_consensus_boost)
        self.method_d_min_position = int(method_d_min_position)
        self.method_d_tail_guard_tokens = int(method_d_tail_guard_tokens)
        self.method_d_focus_radius = max(0, int(method_d_focus_radius))
        self.method_d_source_token_boost = float(method_d_source_token_boost)
        self.method_d_source_query_tokens = max(1, int(method_d_source_query_tokens))
        self.method_d_require_source_overlap = bool(method_d_require_source_overlap)
        self.method_d_allow_source_before_min_position = bool(
            method_d_allow_source_before_min_position
        )
        self.method_d_focus_bias = float(method_d_focus_bias)
        self.method_d_nonfocus_penalty = float(method_d_nonfocus_penalty)
        self.method_d_source_fusion_alpha = max(
            0.0, min(1.0, float(method_d_source_fusion_alpha))
        )
        self.method_d_source_fusion_low_alpha = max(
            0.0, min(self.method_d_source_fusion_alpha, float(method_d_source_fusion_low_alpha))
        )
        self.method_d_source_fusion_source_threshold = max(
            0.0, float(method_d_source_fusion_source_threshold)
        )
        self.method_d_source_fusion_focus_only = bool(method_d_source_fusion_focus_only)
        self.method_d_source_cue_focus = bool(method_d_source_cue_focus)
        self.method_d_source_cue_answer_tokens = max(1, int(method_d_source_cue_answer_tokens))
        self.method_d_retrieve_focus_only = bool(method_d_retrieve_focus_only)
        self.method_d_retrieve_focus_context_tokens = max(
            0, int(method_d_retrieve_focus_context_tokens)
        )
        self.method_d_reuse_ttl_tokens = max(0, int(method_d_reuse_ttl_tokens))
        self.method_d_reuse_source_threshold = max(0.0, float(method_d_reuse_source_threshold))
        self.method_d_source_gate_bypass_threshold = max(
            0.0, float(method_d_source_gate_bypass_threshold)
        )
        self.method_d_reuse_gate_bypass = bool(method_d_reuse_gate_bypass)
        self.method_d_reuse_kv_cache = bool(method_d_reuse_kv_cache)
        self.method_d_triton_scoring = bool(method_d_triton_scoring)
        self.method_d_triton_scoring_batch_chunks = max(
            1, int(method_d_triton_scoring_batch_chunks)
        )
        self.diagnostic_bf16_dram = bool(diagnostic_bf16_dram)

        # Deferred initialization: num_layers is set on first update
        self._num_layers = num_layers
        self._manager: Optional[HeteroKVManager] = None

        # Global sequence length tracker for RoPE alignment
        self.real_seq_len: int = 0

        # Self-healing: pre-computed DRAM swap-in token count
        self._swap_in_tokens: int = 0

        # Method D: store the last decode query K for query-aware retrieval
        self._last_decode_query_k: Optional[torch.Tensor] = None
        self._last_returned_key_positions: Dict[int, torch.Tensor] = {}
        self._last_retrieved_counts: Dict[int, int] = {}
        self._last_retrieved_focus_masks: Dict[int, torch.Tensor] = {}
        self._last_retrieved_source_fusion_alpha: Dict[int, float] = {}
        self._method_d_oracle_range: Optional[Tuple[int, int]] = None
        self._method_d_events = []
        self._method_d_query_history: Dict[int, torch.Tensor] = {}
        self._source_token_ids: Optional[torch.Tensor] = None
        self._source_cue_token_ids = []
        self._attention_probe_events = []

        # Triton-optimized path: 4-bit DRAM data for fused kernel (no BF16 decompression)
        self._dram_quant_kv: Optional[Dict[str, torch.Tensor]] = None
        self._dram_quant_layer: int = -1

        # ──────────────────────────────────────────────────────────────
        # Oracle 集成：存储待处理的注意力权重
        # 用途：在每个 decode step 后捕获注意力权重，传递给 manager.update_attention_scores()
        # 机制：fused_attention_patch 在计算 attention 时将权重存储在这里
        #       cache.update() 在最后一层检测到权重时，更新 oracle
        # ──────────────────────────────────────────────────────────────
        self._pending_attention_weights: Optional[torch.Tensor] = None
        self._pending_key_positions: Optional[torch.Tensor] = None

        print(
            f"[FusedHeteroCache] Initialized | "
            f"sink={sink_tokens} tail={keep_tail} chunk={chunk_size} "
            f"quant={'ON' if enable_quant else 'OFF'} "
            f"prefetch={'ON' if enable_prefetch else 'OFF'} "
            f"triton={'ON' if self.enable_triton else 'OFF'} "
            f"self_healing={'ON' if self_healing else 'OFF'}"
            f"{f' adaptive={adaptive_self_healing}' if adaptive_self_healing else ''}"
            f"{f' retrieval_bias={self.method_d_retrieval_bias:.3f}' if self.method_d_retrieval_bias else ''}"
            f"{f' score_reduce={self.method_d_score_reduce}' if enable_method_d else ''}"
            f"{f' query_history={self.method_d_query_history_tokens}' if enable_method_d else ''}"
            f"{f' consensus={self.method_d_consensus_boost:.3f}' if self.method_d_consensus_boost else ''}"
            f"{f' source_token_boost={self.method_d_source_token_boost:.3f}' if self.method_d_source_token_boost else ''}"
            f"{' source_overlap_required' if self.method_d_require_source_overlap else ''}"
            f"{f' focus_bias={self.method_d_focus_bias:.3f}' if self.method_d_focus_bias else ''}"
            f"{f' source_fusion={self.method_d_source_fusion_alpha:.3f}' if self.method_d_source_fusion_alpha else ''}"
            f"{' retrieve_focus_only' if self.method_d_retrieve_focus_only else ''}"
            f"{f' reuse_ttl={self.method_d_reuse_ttl_tokens}' if self.method_d_reuse_ttl_tokens else ''}"
            f"{f' source_gate_bypass>={self.method_d_source_gate_bypass_threshold:.3f}' if self.method_d_source_gate_bypass_threshold else ''}"
            f"{' reuse_gate_bypass=ON' if self.method_d_reuse_gate_bypass else ''}"
            f"{' reuse_kv_cache=ON' if self.method_d_reuse_kv_cache else ''}"
            f"{' triton_scoring=ON' if self.method_d_triton_scoring else ''}"
            f"{f' triton_batch={self.method_d_triton_scoring_batch_chunks}' if self.method_d_triton_scoring else ''}"
        )

    def _ensure_manager(self, layer_idx: int) -> HeteroKVManager:
        """Lazy initialization of the tiered storage manager."""
        if self._manager is not None:
            return self._manager

        # Heuristic: if num_layers not provided, infer from layer_idx + 1
        num_layers = self._num_layers if self._num_layers is not None else (layer_idx + 1)
        self._manager = HeteroKVManager(
            num_layers=num_layers,
            sink_tokens=self.sink_tokens,
            hbm_budget_tokens=self.keep_tail,
            device=self.device,
            enable_quant=self.enable_quant,
            enable_prefetch=self.enable_prefetch,
            group_size=self.group_size,
            bandwidth_limiter=self._bandwidth_limiter,
            enable_method_d=self.enable_method_d,
            method_d_alpha=self.method_d_alpha,
            method_d_token_window=self.method_d_token_window,
            method_d_score_reduce=self.method_d_score_reduce,
            method_d_top_r=self.method_d_top_r,
            method_d_consensus_boost=self.method_d_consensus_boost,
            method_d_min_position=self.method_d_min_position,
            method_d_tail_guard_tokens=self.method_d_tail_guard_tokens,
            method_d_focus_radius=self.method_d_focus_radius,
            method_d_source_token_boost=self.method_d_source_token_boost,
            method_d_source_query_tokens=self.method_d_source_query_tokens,
            method_d_require_source_overlap=self.method_d_require_source_overlap,
            method_d_allow_source_before_min_position=(
                self.method_d_allow_source_before_min_position
            ),
            method_d_source_cue_focus=self.method_d_source_cue_focus,
            method_d_source_cue_answer_tokens=self.method_d_source_cue_answer_tokens,
            method_d_retrieve_focus_only=self.method_d_retrieve_focus_only,
            method_d_retrieve_focus_context_tokens=(
                self.method_d_retrieve_focus_context_tokens
            ),
            method_d_reuse_ttl_tokens=self.method_d_reuse_ttl_tokens,
            method_d_reuse_source_threshold=self.method_d_reuse_source_threshold,
            method_d_reuse_kv_cache=self.method_d_reuse_kv_cache,
            method_d_triton_scoring=self.method_d_triton_scoring,
            method_d_triton_scoring_batch_chunks=(
                self.method_d_triton_scoring_batch_chunks
            ),
            diagnostic_bf16_dram=self.diagnostic_bf16_dram,
        )
        if self._source_token_ids is not None:
            self._manager.set_source_token_ids(self._source_token_ids)
        if self._source_cue_token_ids:
            self._manager.set_source_cue_token_ids(
                self._source_cue_token_ids,
                answer_tokens=self.method_d_source_cue_answer_tokens,
            )
        if self._method_d_oracle_range is not None:
            self._manager.set_method_d_oracle_range(self._method_d_oracle_range)
        return self._manager

    # ------------------------------------------------------------------
    # DynamicCache interface
    # ------------------------------------------------------------------

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Main HF entry point. Automatically distinguishes prefill (seq_len > 1)
        from decode (seq_len == 1).
        """
        manager = self._ensure_manager(layer_idx)
        new_len = key_states.shape[-2]
        mode = "prefill" if new_len > 1 else "decode"
        cache_kwargs = cache_kwargs or {}
        cache_position = cache_kwargs.get("cache_position")
        if cache_position is not None and cache_position.numel() > 0:
            logical_start = int(cache_position.reshape(-1)[0].item())
        else:
            logical_start = self.real_seq_len
        query_states = cache_kwargs.get("query_states")

        out_k, out_v = manager.update(
            layer_idx=layer_idx,
            key_states=key_states,
            value_states=value_states,
            mode=mode,
            seq_offset=logical_start,
        )
        base_positions = manager.get_key_positions(layer_idx)
        if base_positions is not None:
            self._last_returned_key_positions[layer_idx] = base_positions
        self._last_retrieved_counts[layer_idx] = 0
        self._last_retrieved_focus_masks.pop(layer_idx, None)

        # ──────────────────────────────────────────────────────────────
        # Oracle 集成：在最后一层更新 HeavyHitterOracle
        #
        # 关键修复：之前 oracle.update() 从未被调用，导致 token_scores 始终为 None
        # 现在在每个 decode step 的最后一层，将捕获的注意力权重传递给 oracle
        #
        # 数据流：
        #   1. fused_attention_patch 在计算 attention 时捕获 attn_weights
        #   2. 存储到 cache._pending_attention_weights
        #   3. 在这里（最后一层）传递给 manager.update_attention_scores()
        #   4. manager 调用 oracle.update() 更新累积注意力分数
        #   5. 下一次驱逐决策时使用这些分数（不再是 FIFO）
        # ──────────────────────────────────────────────────────────────
        is_last_layer = (self._num_layers is not None and layer_idx == self._num_layers - 1)
        if mode == "decode" and is_last_layer:
            if hasattr(self, '_pending_attention_weights') and \
               self._pending_attention_weights is not None:
                # 将注意力权重传递给 manager，更新 oracle
                manager.update_attention_scores(
                    self._pending_attention_weights,
                    key_positions=self._pending_key_positions,
                )
                # 清理，为下一个 token 准备（避免内存泄漏）
                self._pending_attention_weights = None
                self._pending_key_positions = None

        # Self-healing: swap in DRAM tokens during decode.
        has_dram_for_layer = manager.count_dram_tokens(layer_idx) > 0
        if mode == "decode" and self.self_healing and (self._swap_in_tokens > 0 or has_dram_for_layer):
            self._last_retrieved_counts.pop(layer_idx, None)
            self._last_retrieved_focus_masks.pop(layer_idx, None)
            self._last_retrieved_source_fusion_alpha.pop(layer_idx, None)
            if self.enable_method_d and self._method_d_layer_enabled(layer_idx):
                # ──────────────────────────────────────────────────────
                # Method D: Query-Aware Retrieval (INDEPENDENT EXPERIMENT)
                # 1. Use current token's K as query to compute similarity
                # 2. Select top-k chunks by token-level Query x Key dot product
                # 3. Retrieve and decompress only selected chunks
                # ──────────────────────────────────────────────────────
                # Prefer RoPE-applied query_states captured by the attention
                # wrapper.  key_states is only a compatibility fallback.
                query_k = query_states if query_states is not None else key_states
                if query_states is None and layer_idx == 0:
                    print("  [Method D][WARN] query_states missing; falling back to key_states")
                retrieval_query = self._method_d_update_query_history(layer_idx, query_k)
                _, _, selected_count, method_used = manager.decompress_dram_chunks_method_d(
                    layer_idx,
                    retrieval_query,
                    top_k=self.method_d_top_k,
                    score_only=True,
                )
                if selected_count > 0:
                    selected_chunks = manager.get_last_method_d_selection(layer_idx)
                    gate_info = self._method_d_source_gate_shortcut(selected_chunks)
                    if gate_info is None:
                        use_retrieval, gate_info = self._method_d_gate_retrieval(
                            query_k, out_k, manager
                        )
                    else:
                        use_retrieval = True
                    effective_alpha = (
                        self._method_d_effective_source_fusion_alpha(selected_chunks)
                        if use_retrieval else 0.0
                    )
                    dram_k, dram_v, count = None, None, 0
                    if use_retrieval:
                        dram_k, dram_v, count, method_used = manager.decompress_dram_chunks_method_d(
                            layer_idx,
                            retrieval_query,
                            top_k=self.method_d_top_k,
                            use_last_selection=True,
                        )
                        if dram_k is None or count <= 0:
                            use_retrieval = False
                            self._last_retrieved_source_fusion_alpha.pop(layer_idx, None)
                    if use_retrieval:
                        self._last_retrieved_source_fusion_alpha[layer_idx] = effective_alpha
                    selected_chunks = manager.get_last_method_d_selection(layer_idx)
                    self._record_method_d_event(
                        layer_idx=layer_idx,
                        method_used=method_used,
                        selected_chunks=selected_chunks,
                        retrieved_count=count if use_retrieval else 0,
                        gate_allowed=use_retrieval,
                        gate_info=gate_info,
                        query_history_len=retrieval_query.shape[-2],
                    )

                    if use_retrieval:
                        out_k = torch.cat([dram_k, out_k], dim=-2)
                        out_v = torch.cat([dram_v, out_v], dim=-2)
                        retrieved_pos = manager.get_last_retrieved_positions(layer_idx)
                        if retrieved_pos is not None and base_positions is not None:
                            self._last_retrieved_counts[layer_idx] = int(retrieved_pos.numel())
                            self._last_returned_key_positions[layer_idx] = torch.cat(
                                [retrieved_pos.to(base_positions.device), base_positions], dim=0
                            )
                            focus_mask = manager.get_last_retrieved_focus_mask(layer_idx)
                            if focus_mask is not None:
                                self._last_retrieved_focus_masks[layer_idx] = focus_mask.to(
                                    base_positions.device, non_blocking=True
                                )
                        self._swap_in_tokens = count  # CRITICAL FIX: Use actual retrieved count, not full DRAM count
                        print(
                            f"  [Method D] Retrieved {count} tokens using {method_used} | "
                            f"dram_best={gate_info.get('dram_best', float('nan')):.6f} "
                            f"hbm_best={gate_info.get('hbm_best', float('nan')):.6f} "
                            f"margin={self.method_d_gate_margin:.3f}"
                        )
                    else:
                        self._swap_in_tokens = 0
                        if hasattr(manager, "clear_method_d_reuse"):
                            manager.clear_method_d_reuse(layer_idx)
                        print(
                            "  [Method D] skipped DRAM retrieval by HBM gate | "
                            f"dram_best={gate_info.get('dram_best', float('nan')):.6f} "
                            f"hbm_best={gate_info.get('hbm_best', float('nan')):.6f} "
                            f"margin={self.method_d_gate_margin:.3f}"
                        )
                else:
                    self._swap_in_tokens = 0  # No DRAM data retrieved
                self._dram_quant_kv = None
            elif self.enable_method_d:
                self._swap_in_tokens = 0
                self._dram_quant_kv = None

            elif self.adaptive_self_healing and self.enable_triton:
                # ──────────────────────────────────────────────────────
                # Path A: Dynamic Window + Triton Fused Kernel (TOGETHER)
                # 1. AdaptivePrefetchController computes w_t from σ(A_t)
                # 2. Select top-w_t chunks by attention score
                # 3. Transfer 4-bit data to GPU (NO BF16 decompression)
                # 4. Store in _dram_quant_kv for Triton kernel to consume
                # ──────────────────────────────────────────────────────
                quant_kv = manager.get_dram_chunks_quantized_adaptive(
                    layer_idx, window_size=None
                )
                if quant_kv is not None:
                    # Store 4-bit data for Triton kernel to pick up during attention
                    self._dram_quant_kv = quant_kv
                    self._dram_quant_layer = layer_idx
                else:
                    self._dram_quant_kv = None

            elif self.adaptive_self_healing:
                # ──────────────────────────────────────────────────────
                # Path B: Dynamic Window only (no Triton, fallback to BF16)
                # Retrieve top-w_t chunks, decompress to BF16, concat
                # ──────────────────────────────────────────────────────
                dram_k, dram_v, count = manager.decompress_dram_chunks_adaptive(
                    layer_idx, window_size=None
                )
                if dram_k is not None and count > 0:
                    out_k = torch.cat([dram_k, out_k], dim=-2)
                    out_v = torch.cat([dram_v, out_v], dim=-2)
                self._dram_quant_kv = None

            else:
                # ──────────────────────────────────────────────────────
                # Path C: Full retrieval (100% recall, O(N) memory spike)
                # Decompress ALL DRAM chunks to BF16, concat
                # ──────────────────────────────────────────────────────
                dram_k, dram_v, count = manager.decompress_dram_chunks(layer_idx)
                if dram_k is not None and count > 0:
                    out_k = torch.cat([dram_k, out_k], dim=-2)
                    out_v = torch.cat([dram_v, out_v], dim=-2)
                self._dram_quant_kv = None

        if layer_idx == 0:
            self.real_seq_len = max(self.real_seq_len, logical_start + new_len)
            # After layer 0 decode, refresh DRAM token count for next step
            # Skip for Method D since we already set _swap_in_tokens with actual retrieved count
            if mode == "decode" and self.self_healing and not self.enable_method_d:
                self._refresh_swap_count()

        return out_k, out_v

    def _method_d_layer_enabled(self, layer_idx: int) -> bool:
        """Return whether Method-D retrieval may run on this layer."""
        if layer_idx < self.method_d_layer_min:
            return False
        if self.method_d_layer_max is not None and layer_idx > self.method_d_layer_max:
            return False
        return True

    def _record_method_d_event(
        self,
        layer_idx: int,
        method_used: str,
        selected_chunks,
        retrieved_count: int,
        gate_allowed: bool,
        gate_info: Dict[str, float],
        query_history_len: int = 1,
    ) -> None:
        event = {
            "layer": int(layer_idx),
            "method": method_used,
            "gate_allowed": bool(gate_allowed),
            "retrieved_count": int(retrieved_count),
            "selected_chunks": selected_chunks,
            "dram_best": float(gate_info.get("dram_best", float("nan"))),
            "hbm_best": float(gate_info.get("hbm_best", float("nan"))),
            "margin": float(self.method_d_gate_margin),
            "retrieval_bias": float(self.method_d_retrieval_bias),
            "score_reduce": self.method_d_score_reduce,
            "query_history_len": int(query_history_len),
            "consensus_boost": float(self.method_d_consensus_boost),
            "min_position": int(self.method_d_min_position),
            "tail_guard_tokens": int(self.method_d_tail_guard_tokens),
            "focus_radius": int(self.method_d_focus_radius),
            "source_token_boost": float(self.method_d_source_token_boost),
            "require_source_overlap": bool(self.method_d_require_source_overlap),
            "allow_source_before_min_position": bool(
                self.method_d_allow_source_before_min_position
            ),
            "focus_bias": float(self.method_d_focus_bias),
            "nonfocus_penalty": float(self.method_d_nonfocus_penalty),
            "source_fusion_alpha": float(self.method_d_source_fusion_alpha),
            "source_fusion_focus_only": bool(self.method_d_source_fusion_focus_only),
            "source_cue_focus": bool(self.method_d_source_cue_focus),
            "retrieve_focus_only": bool(self.method_d_retrieve_focus_only),
            "retrieve_focus_context_tokens": int(
                self.method_d_retrieve_focus_context_tokens
            ),
            "effective_source_fusion_alpha": float(
                self._last_retrieved_source_fusion_alpha.get(layer_idx, 0.0)
            ),
            "source_fusion_low_alpha": float(self.method_d_source_fusion_low_alpha),
            "source_fusion_source_threshold": float(
                self.method_d_source_fusion_source_threshold
            ),
            "reuse_ttl_tokens": int(self.method_d_reuse_ttl_tokens),
            "reuse_source_threshold": float(self.method_d_reuse_source_threshold),
            "source_gate_bypass_threshold": float(
                self.method_d_source_gate_bypass_threshold
            ),
            "source_gate_bypass": bool(gate_info.get("source_gate_bypass", False)),
            "source_gate_best": float(gate_info.get("source_gate_best", 0.0)),
            "reuse_gate_bypass": bool(gate_info.get("reuse_gate_bypass", False)),
            "reuse_gate_bypass_enabled": bool(self.method_d_reuse_gate_bypass),
            "reuse_kv_cache": bool(self.method_d_reuse_kv_cache),
            "triton_scoring": bool(self.method_d_triton_scoring),
            "triton_scoring_batch_chunks": int(self.method_d_triton_scoring_batch_chunks),
            "scoring_backend": (
                selected_chunks[0].get("scoring_backend")
                if selected_chunks else "unknown"
            ),
        }
        self._method_d_events.append(event)
        if len(self._method_d_events) > 512:
            self._method_d_events = self._method_d_events[-512:]

    def _method_d_update_query_history(
        self,
        layer_idx: int,
        query_states: torch.Tensor,
    ) -> torch.Tensor:
        """Keep a bounded recent-query matrix for multi-token Q x K retrieval."""
        if self.method_d_query_history_tokens <= 1:
            return query_states
        q = query_states.detach()
        if q.shape[-2] > self.method_d_query_history_tokens:
            q = q[..., -self.method_d_query_history_tokens :, :]
        prev = self._method_d_query_history.get(layer_idx)
        if prev is None:
            hist = q
        else:
            hist = torch.cat([prev.to(q.device, dtype=q.dtype), q], dim=-2)
            hist = hist[..., -self.method_d_query_history_tokens :, :]
        self._method_d_query_history[layer_idx] = hist.detach()
        return hist

    def get_method_d_events(self):
        return list(self._method_d_events)

    def set_method_d_oracle_range(self, token_range: Optional[Tuple[int, int]]) -> None:
        """Diagnostic only: force Method-D to retrieve chunks covering a token range."""
        self._method_d_oracle_range = None if token_range is None else (
            int(token_range[0]),
            int(token_range[1]),
        )
        if self._manager is not None:
            self._manager.set_method_d_oracle_range(self._method_d_oracle_range)

    def set_source_token_ids(self, token_ids: torch.Tensor) -> None:
        """Register source token ids for optional source-aware reranking."""
        self._source_token_ids = token_ids.detach().reshape(-1).cpu().long()
        if self._manager is not None:
            self._manager.set_source_token_ids(self._source_token_ids)

    def set_source_cue_token_ids(self, cue_token_ids, answer_tokens: Optional[int] = None) -> None:
        """Register non-oracle source cue token sequences for answer-span focus."""
        self._source_cue_token_ids = [
            [int(token) for token in cue] for cue in (cue_token_ids or []) if cue
        ]
        if answer_tokens is not None:
            self.method_d_source_cue_answer_tokens = max(1, int(answer_tokens))
        if self._manager is not None:
            self._manager.set_source_cue_token_ids(
                self._source_cue_token_ids,
                answer_tokens=self.method_d_source_cue_answer_tokens,
            )

    def _method_d_effective_source_fusion_alpha(self, selected_chunks) -> float:
        base = float(self.method_d_source_fusion_alpha)
        threshold = float(self.method_d_source_fusion_source_threshold)
        if base <= 0.0 or threshold <= 0.0:
            return base
        best_source_score = 0.0
        for chunk in selected_chunks or []:
            try:
                best_source_score = max(best_source_score, float(chunk.get("source_token_score", 0.0)))
            except Exception:
                continue
        if best_source_score < threshold:
            return float(self.method_d_source_fusion_low_alpha)
        return base

    def record_attention_probe(
        self,
        layer_idx: int,
        attn_weights: torch.Tensor,
        key_positions: Optional[torch.Tensor],
        cache_position: Optional[torch.Tensor],
    ) -> None:
        """Record first-token attribution for retrieved vs active HBM KV."""
        if key_positions is None or attn_weights is None:
            return
        try:
            weights = attn_weights.detach().float()
            if weights.dim() != 4:
                return
            weights = weights[0, :, -1, :].mean(dim=0)
            positions = key_positions.reshape(-1).to(weights.device).long()
            if positions.numel() != weights.numel():
                return
            retrieved_count = min(
                int(self._last_retrieved_counts.get(layer_idx, 0)),
                int(weights.numel()),
            )
            retrieved_mask = torch.zeros_like(weights, dtype=torch.bool)
            if retrieved_count > 0:
                retrieved_mask[:retrieved_count] = True
            hbm_mask = ~retrieved_mask
            event = {
                "layer": int(layer_idx),
                "kv_len": int(weights.numel()),
                "retrieved_count": int(retrieved_count),
                "retrieved_mass": float(weights[retrieved_mask].sum().item()) if retrieved_count else 0.0,
                "hbm_mass": float(weights[hbm_mask].sum().item()) if hbm_mask.any() else 0.0,
            }
            if cache_position is not None and cache_position.numel() > 0:
                event["query_position"] = int(cache_position.reshape(-1)[-1].item())
            if self._method_d_oracle_range is not None:
                start, end = self._method_d_oracle_range
                needle_mask = (positions >= start) & (positions < end)
                event["needle_range"] = [int(start), int(end)]
                event["needle_mass"] = float(weights[needle_mask].sum().item()) if needle_mask.any() else 0.0
                event["retrieved_needle_mass"] = (
                    float(weights[needle_mask & retrieved_mask].sum().item())
                    if needle_mask.any() and retrieved_count else 0.0
                )
                event["needle_positions_present"] = int(needle_mask.sum().item())
            top_k = min(5, int(weights.numel()))
            if top_k:
                values, indices = torch.topk(weights, k=top_k)
                event["top_positions"] = [
                    int(positions[int(idx)].item()) for idx in indices.detach().cpu()
                ]
                event["top_weights"] = [float(v) for v in values.detach().cpu()]
            self._attention_probe_events.append(event)
            if len(self._attention_probe_events) > 2048:
                self._attention_probe_events = self._attention_probe_events[-2048:]
        except Exception as exc:
            print(f"[HeteroKV AttentionProbe][WARN] {exc}")

    def get_attention_probe_events(self):
        return list(self._attention_probe_events)

    def force_shrink_hbm_budget(self, new_hbm_budget_tokens: int) -> None:
        """Diagnostic: shrink HBM cache after prefill to isolate prefill damage."""
        if self._manager is None:
            return
        self.keep_tail = int(new_hbm_budget_tokens)
        self._manager.force_shrink_hbm_budget(new_hbm_budget_tokens)
        self._last_returned_key_positions.clear()
        self._last_retrieved_counts.clear()
        self._last_retrieved_focus_masks.clear()
        self._swap_in_tokens = 0

    def _method_d_source_gate_shortcut(self, selected_chunks) -> Optional[Dict[str, float]]:
        """Optional diagnostic shortcut: trust very strong source-cue evidence.

        This is disabled by default.  When enabled, it skips the HBM-vs-DRAM
        dot-product gate only if the selected chunk already has a source-cue
        score above the configured threshold.  It must be reported separately
        from the default Method-D result because it changes the gate policy.
        """
        threshold = float(self.method_d_source_gate_bypass_threshold)
        if not selected_chunks:
            return None
        best_source = 0.0
        finite_scores = []
        for chunk in selected_chunks:
            try:
                best_source = max(best_source, float(chunk.get("source_token_score", 0.0)))
            except Exception:
                pass
            try:
                score = float(chunk.get("score", float("nan")))
                if score == score:
                    finite_scores.append(score)
            except Exception:
                pass
        dram_best = max(finite_scores) if finite_scores else float("nan")
        if self.method_d_reuse_gate_bypass and any(
            bool(chunk.get("reuse_hit", False)) for chunk in selected_chunks
        ):
            return {
                "dram_best": dram_best,
                "hbm_best": float("nan"),
                "reuse_gate_bypass": 1.0,
                "source_gate_best": best_source,
            }
        if threshold <= 0.0 or best_source < threshold:
            return None
        return {
            "dram_best": dram_best,
            "hbm_best": float("nan"),
            "source_gate_bypass": 1.0,
            "source_gate_best": best_source,
            "source_gate_threshold": threshold,
        }

    def _method_d_gate_retrieval(
        self,
        query_states: torch.Tensor,
        hbm_key_states: torch.Tensor,
        manager: HeteroKVManager,
    ) -> Tuple[bool, Dict[str, float]]:
        """Use retrieved DRAM chunks only when they beat active HBM QK evidence."""
        try:
            from src.memory.query_aware_retriever import QueryAwareRetriever

            dram_scores = getattr(manager, "_last_retrieval_scores", {})
            finite_scores = [
                float(v) for v in dram_scores.values()
                if isinstance(v, (int, float)) and v == v
            ]
            if not finite_scores:
                return True, {"dram_best": float("nan"), "hbm_best": float("nan")}
            dram_best = max(finite_scores)

            q = query_states.detach().to(hbm_key_states.device, non_blocking=True).float()
            if q.dim() == 3:
                q = q.unsqueeze(2)
            if q.shape[-2] != 1:
                q = q[..., -1:, :]
            hbm_k = hbm_key_states.detach().float()
            hbm_scores = QueryAwareRetriever._token_dot_scores(q, hbm_k)
            hbm_best = float(hbm_scores.reshape(-1).max().item())
            return dram_best >= hbm_best * self.method_d_gate_margin, {
                "dram_best": dram_best,
                "hbm_best": hbm_best,
            }
        except Exception as exc:
            print(f"  [Method D][WARN] retrieval gate failed open: {exc}")
            return True, {"dram_best": float("nan"), "hbm_best": float("nan")}

    def get_seq_length(self, layer_idx: int = 0) -> int:
        # Report HBM pool + DRAM swap-in tokens so attention mask dimensions match.
        if self._manager is None:
            return 0
        k_cache = self._manager._key_cache
        if layer_idx >= len(k_cache) or k_cache[layer_idx] is None:
            return 0
        hbm_size = k_cache[layer_idx].shape[-2]
        return hbm_size + self._swap_in_tokens

    def get_key_positions(self, layer_idx: int = 0) -> Optional[torch.Tensor]:
        """Return logical positions for the K/V tensor returned by last update."""
        if layer_idx in self._last_returned_key_positions:
            return self._last_returned_key_positions[layer_idx]
        if self._manager is None:
            return None
        return self._manager.get_key_positions(layer_idx)

    def get_retrieved_count(self, layer_idx: int = 0) -> int:
        """Return number of DRAM tokens prepended to the current layer output."""
        return int(self._last_retrieved_counts.get(layer_idx, 0))

    def get_retrieval_focus_mask(self, layer_idx: int = 0) -> Optional[torch.Tensor]:
        """Return source-aware focus mask aligned with prepended DRAM tokens."""
        return self._last_retrieved_focus_masks.get(layer_idx)

    def get_retrieval_source_fusion_alpha(self, layer_idx: int = 0) -> float:
        """Return dynamic source-fusion alpha for the current retrieved block."""
        return float(self._last_retrieved_source_fusion_alpha.get(layer_idx, 0.0))

    @property
    def seen_tokens(self) -> int:
        return self.real_seq_len

    # ------------------------------------------------------------------
    # transformers 4.57+ Cache protocol: get_mask_sizes for mask dimension
    # ------------------------------------------------------------------

    def get_mask_sizes(
        self, cache_position: torch.Tensor, layer_idx: int = 0
    ) -> Tuple[int, int]:
        """
        Return (kv_length, kv_offset) for correct mask dimensions.
        Returns the actual physical KV length so mask matches what update() returns.
        """
        if self._manager is None:
            return cache_position.shape[0], 0
        query_len = cache_position.shape[0]
        physical_size = self._manager.predict_physical_length_after_update(layer_idx, query_len)
        return physical_size + self._swap_in_tokens, 0

    def _refresh_swap_count(self) -> None:
        """Pre-compute DRAM swap-in token count for the next decode step."""
        if self._manager is not None:
            self._swap_in_tokens = self._manager.count_dram_tokens(layer_idx=0)

    # ------------------------------------------------------------------
    # Tiered storage passthrough API
    # ------------------------------------------------------------------

    def allocate(self, layer_idx: int, seq_len: int, budget_tokens: Optional[int] = None) -> bool:
        """Pre-allocate physical HBM pool for a layer."""
        return self._ensure_manager(layer_idx).allocate(layer_idx, seq_len, budget_tokens)

    def compress(self, layer_idx: int, device: str = "DRAM") -> int:
        """Force-compress overflow tokens to the specified tier."""
        return self._ensure_manager(layer_idx).compress(layer_idx, device)

    def memory_summary(self) -> Dict[str, Any]:
        """Return HBM/DRAM usage snapshot."""
        if self._manager is None:
            return {"hbm_tokens": 0, "dram_entries": 0, "dram_bytes": 0, "max_hbm_tokens": self.sink_tokens + self.keep_tail}
        return self._manager.memory_summary()

    @property
    def dram_table(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """Expose DRAM table for legacy benchmark compatibility."""
        if self._manager is None:
            return {}
        return self._manager._dram_table

    def schedule_prefetch(self, chunk_key: str) -> None:
        """Schedule an asynchronous DRAM -> HBM prefetch."""
        if self._manager is not None:
            self._manager.schedule_prefetch(chunk_key)

    def swap_in_chunk(
        self, chunk_key: str
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Swap a compressed chunk back into HBM (legacy: decompresses to BF16)."""
        if self._manager is None:
            return None
        return self._manager.swap_in(0, chunk_key)

    def swap_in_quantized(
        self, chunk_key: str
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        Zero-copy swap-in: returns quantized tensors without decompressing.
        Use with fused_dequant_attn_* to avoid BF16 memory spikes.
        """
        if self._manager is None:
            return None
        return self._manager.swap_in_quantized(0, chunk_key)

    def fused_attn_on_swapped(
        self,
        q: torch.Tensor,
        chunk_key: str,
        sm_scale: Optional[float] = None,
    ) -> Optional[torch.Tensor]:
        """
        End-to-end fused attention on a swapped-in quantized chunk.
        Computes attention output without ever materializing BF16 K/V.

        Args:
            q: Query tensor [batch, heads, 1, head_dim] (decode) or [batch, heads, seq_q, head_dim]
            chunk_key: DRAM chunk to attend to
            sm_scale: softmax scaling (default 1/sqrt(head_dim))

        Returns:
            Attention output [batch, heads, seq_q, head_dim] or None
        """
        quant_data = self.swap_in_quantized(chunk_key)
        if quant_data is None:
            return None

        try:
            if q.shape[2] == 1:
                return fused_dequant_attn_decode(
                    q,
                    quant_data["k_data"], quant_data["k_scales"], quant_data["k_zps"],
                    quant_data["v_data"], quant_data["v_scales"], quant_data["v_zps"],
                    sm_scale=sm_scale,
                )
            else:
                return fused_dequant_attn_forward(
                    q,
                    quant_data["k_data"], quant_data["k_scales"], quant_data["k_zps"],
                    quant_data["v_data"], quant_data["v_scales"], quant_data["v_zps"],
                    sm_scale=sm_scale,
                )
        except Exception:
            # Fallback to standard swap-in + decompress path
            result = self.swap_in_chunk(chunk_key)
            if result is None:
                return None
            restored_k, restored_v = result
            scores = torch.matmul(q.float(), restored_k.float().transpose(-2, -1))
            if sm_scale is not None:
                scores = scores * sm_scale
            attn = torch.softmax(scores, dim=-1)
            return torch.matmul(attn, restored_v.float())


# ---------------------------------------------------------------------------
# Chunked Prefill Engine
# ---------------------------------------------------------------------------

class ChunkedPrefillEngine:
    """
    Drives long-sequence prefill by splitting inputs into chunks and feeding
    them through the model with the Hetero cache.
    """

    def __init__(
        self,
        model,
        cache: FusedHeteroCache,
        chunk_size: int = 2048,
    ):
        self.model = model
        self.cache = cache
        self.chunk_size = chunk_size

    @torch.inference_mode()
    def prefill(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Chunked prefill loop. Ensures transient peak memory stays bounded.
        """
        total_len = input_ids.shape[-1]
        device = input_ids.device

        print(f"[ChunkedPrefill] total_tokens={total_len} chunk_size={self.chunk_size}")

        chunk_idx = 0
        for start in range(0, total_len, self.chunk_size):
            end = min(start + self.chunk_size, total_len)
            chunk_ids = input_ids[:, start:end]
            chunk_len = end - start

            if position_ids is not None:
                chunk_pos = position_ids[:, start:end]
            else:
                chunk_pos = torch.arange(
                    start, end, dtype=torch.long, device=device
                ).unsqueeze(0)

            # cache_position uses true absolute positions.  The short-KV
            # attention wrapper builds a causal mask from retained key
            # positions instead of pretending the cache is contiguous.
            chunk_cache_pos = torch.arange(
                start, end, dtype=torch.long, device=device
            )

            chunk_mask = None
            if attention_mask is not None:
                chunk_mask = attention_mask[:, :end]

            self.model(
                input_ids=chunk_ids,
                past_key_values=self.cache,
                use_cache=True,
                position_ids=chunk_pos,
                attention_mask=chunk_mask,
                cache_position=chunk_cache_pos,
            )

            mem_gb = torch.cuda.memory_allocated(device) / 1024 ** 3
            peak_gb = torch.cuda.max_memory_allocated(device) / 1024 ** 3
            print(
                f"  chunk [{start:>6}:{end:>6}] "
                f"current={mem_gb:.2f}GB peak={peak_gb:.2f}GB "
                f"dram_entries={len(self.cache.dram_table)}"
            )

            # Overlap next chunk with prefetch of any DRAM-resident data
            if self.cache._manager is not None:
                for key in list(self.cache.dram_table.keys()):
                    self.cache.schedule_prefetch(key)

            del chunk_ids, chunk_pos
            if chunk_mask is not None:
                del chunk_mask
            # Trigger GC every 4 chunks to reclaim transient tensor memory
            chunk_idx += 1
            if chunk_idx % 4 == 0 or end == total_len:
                gc.collect()

        print(f"[ChunkedPrefill] Done. real_seq_len={self.cache.real_seq_len}")


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def build_fused_cache(
    num_layers: Optional[int] = None,
    device: str = "cuda:0",
    sink_tokens: int = 64,
    keep_tail: int = 8192,
    chunk_size: int = 2048,
    group_size: int = 128,
    enable_quant: bool = True,
    enable_prefetch: bool = True,
    enable_triton: bool = False,
    bandwidth_limiter=None,
    self_healing: bool = True,
    adaptive_self_healing: bool = False,
    enable_method_d: bool = True,
    method_d_alpha: float = 1.0,
    method_d_gate_margin: float = 1.10,
    method_d_token_window: int = 0,
    method_d_layer_min: int = 0,
    method_d_layer_max: Optional[int] = None,
    method_d_top_k: Optional[int] = None,
    method_d_retrieval_bias: float = 0.0,
    method_d_score_reduce: str = "max",
    method_d_top_r: int = 8,
    method_d_query_history_tokens: int = 1,
    method_d_consensus_boost: float = 0.0,
    method_d_min_position: int = 0,
    method_d_tail_guard_tokens: int = 0,
    method_d_focus_radius: int = 0,
    method_d_source_token_boost: float = 0.0,
    method_d_source_query_tokens: int = 64,
    method_d_require_source_overlap: bool = False,
    method_d_allow_source_before_min_position: bool = False,
    method_d_focus_bias: float = 0.0,
    method_d_nonfocus_penalty: float = 0.0,
    method_d_source_fusion_alpha: float = 0.0,
    method_d_source_fusion_low_alpha: float = 0.0,
    method_d_source_fusion_source_threshold: float = 0.0,
    method_d_source_fusion_focus_only: bool = False,
    method_d_source_cue_focus: bool = False,
    method_d_source_cue_answer_tokens: int = 8,
    method_d_retrieve_focus_only: bool = False,
    method_d_retrieve_focus_context_tokens: int = 0,
    method_d_reuse_ttl_tokens: int = 0,
    method_d_reuse_source_threshold: float = 0.0,
    method_d_source_gate_bypass_threshold: float = 0.0,
    method_d_reuse_gate_bypass: bool = False,
    method_d_reuse_kv_cache: bool = False,
    method_d_triton_scoring: bool = False,
    method_d_triton_scoring_batch_chunks: int = 8,
    diagnostic_bf16_dram: bool = False,
) -> FusedHeteroCache:
    """
    Factory: create a fully-configured FusedHeteroCache instance.

    Args:
        adaptive_self_healing: If True, use TRUE dynamic window self-healing
            (retrieves only top-w_t chunks based on attention scores).
            If False, use full retrieval (100% recall, O(N) memory spike).
            Default: False.
    """
    return FusedHeteroCache(
        num_layers=num_layers,
        sink_tokens=sink_tokens,
        keep_tail=keep_tail,
        chunk_size=chunk_size,
        device=device,
        group_size=group_size,
        enable_quant=enable_quant,
        enable_prefetch=enable_prefetch,
        enable_triton=enable_triton,
        bandwidth_limiter=bandwidth_limiter,
        self_healing=self_healing,
        adaptive_self_healing=adaptive_self_healing,
        enable_method_d=enable_method_d,
        method_d_alpha=method_d_alpha,
        method_d_gate_margin=method_d_gate_margin,
        method_d_token_window=method_d_token_window,
        method_d_layer_min=method_d_layer_min,
        method_d_layer_max=method_d_layer_max,
        method_d_top_k=method_d_top_k,
        method_d_retrieval_bias=method_d_retrieval_bias,
        method_d_score_reduce=method_d_score_reduce,
        method_d_top_r=method_d_top_r,
        method_d_query_history_tokens=method_d_query_history_tokens,
        method_d_consensus_boost=method_d_consensus_boost,
        method_d_min_position=method_d_min_position,
        method_d_tail_guard_tokens=method_d_tail_guard_tokens,
        method_d_focus_radius=method_d_focus_radius,
        method_d_source_token_boost=method_d_source_token_boost,
        method_d_source_query_tokens=method_d_source_query_tokens,
        method_d_require_source_overlap=method_d_require_source_overlap,
        method_d_allow_source_before_min_position=method_d_allow_source_before_min_position,
        method_d_focus_bias=method_d_focus_bias,
        method_d_nonfocus_penalty=method_d_nonfocus_penalty,
        method_d_source_fusion_alpha=method_d_source_fusion_alpha,
        method_d_source_fusion_low_alpha=method_d_source_fusion_low_alpha,
        method_d_source_fusion_source_threshold=method_d_source_fusion_source_threshold,
        method_d_source_fusion_focus_only=method_d_source_fusion_focus_only,
        method_d_source_cue_focus=method_d_source_cue_focus,
        method_d_source_cue_answer_tokens=method_d_source_cue_answer_tokens,
        method_d_retrieve_focus_only=method_d_retrieve_focus_only,
        method_d_retrieve_focus_context_tokens=method_d_retrieve_focus_context_tokens,
        method_d_reuse_ttl_tokens=method_d_reuse_ttl_tokens,
        method_d_reuse_source_threshold=method_d_reuse_source_threshold,
        method_d_source_gate_bypass_threshold=method_d_source_gate_bypass_threshold,
        method_d_reuse_gate_bypass=method_d_reuse_gate_bypass,
        method_d_reuse_kv_cache=method_d_reuse_kv_cache,
        method_d_triton_scoring=method_d_triton_scoring,
        method_d_triton_scoring_batch_chunks=method_d_triton_scoring_batch_chunks,
        diagnostic_bf16_dram=diagnostic_bf16_dram,
    )
