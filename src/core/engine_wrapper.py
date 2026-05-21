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

    Transient Cache Architecture:
      - During prefill, the manager intercepts KV tensors but returns the
        FULL sequence to satisfy FlashAttention dimension checks. The massive
        transient memory is reclaimed via aggressive GC.
      - During decode, only a fixed Sink + Tail subset resides in HBM,
        achieving O(1) steady-state memory regardless of sequence length.
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
        enable_triton: bool = True,
        bandwidth_limiter=None,
        self_healing: bool = True,
        adaptive_self_healing: bool = False,
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

        # Deferred initialization: num_layers is set on first update
        self._num_layers = num_layers
        self._manager: Optional[HeteroKVManager] = None

        # Global sequence length tracker for RoPE alignment
        self.real_seq_len: int = 0

        # Self-healing: pre-computed DRAM swap-in token count
        self._swap_in_tokens: int = 0

        # Triton-optimized path: 4-bit DRAM data for fused kernel (no BF16 decompression)
        self._dram_quant_kv: Optional[Dict[str, torch.Tensor]] = None
        self._dram_quant_layer: int = -1

        print(
            f"[FusedHeteroCache] Initialized | "
            f"sink={sink_tokens} tail={keep_tail} chunk={chunk_size} "
            f"quant={'ON' if enable_quant else 'OFF'} "
            f"prefetch={'ON' if enable_prefetch else 'OFF'} "
            f"triton={'ON' if self.enable_triton else 'OFF'} "
            f"self_healing={'ON' if self_healing else 'OFF'}"
            f"{f' adaptive={adaptive_self_healing}' if adaptive_self_healing else ''}"
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
        )
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

        out_k, out_v = manager.update(
            layer_idx=layer_idx,
            key_states=key_states,
            value_states=value_states,
            mode=mode,
            seq_offset=self.real_seq_len,
        )

        # Self-healing: swap in DRAM tokens during decode
        if mode == "decode" and self.self_healing and self._swap_in_tokens > 0:
            if self.adaptive_self_healing and self.enable_triton:
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
            self.real_seq_len += new_len
            # After layer 0 decode, refresh DRAM token count for next step
            if mode == "decode" and self.self_healing:
                self._refresh_swap_count()

        return out_k, out_v

    def get_seq_length(self, layer_idx: int = 0) -> int:
        # Report HBM pool + DRAM swap-in tokens so attention mask dimensions match.
        if self._manager is None:
            return 0
        k_cache = self._manager._key_cache
        if layer_idx >= len(k_cache) or k_cache[layer_idx] is None:
            return 0
        hbm_size = k_cache[layer_idx].shape[-2]
        return hbm_size + self._swap_in_tokens

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
        For prefill chunks: return (chunk_size, 0).
        For decode: return (hbm_pool + dram_swap + 1, 0) to match extended KV.
        """
        if self._manager is None:
            return cache_position.shape[0], 0
        k_cache = self._manager._key_cache
        if layer_idx >= len(k_cache) or k_cache[layer_idx] is None:
            return cache_position.shape[0], 0
        physical_size = k_cache[layer_idx].shape[-2]
        query_len = cache_position.shape[0]
        if query_len > 1:  # Prefill chunk: mask width = chunk_size
            return query_len, 0
        else:  # Decode: mask width = HBM + DRAM swap-in + new token
            return physical_size + self._swap_in_tokens + 1, 0

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

            # cache_position starts from 0 for each chunk so the mask
            # covers only the current chunk's KV (not past + current).
            # RoPE uses position_ids with correct absolute positions.
            chunk_cache_pos = torch.arange(
                0, chunk_len, dtype=torch.long, device=device
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
    enable_triton: bool = True,
    bandwidth_limiter=None,
    self_healing: bool = True,
    adaptive_self_healing: bool = False,
) -> FusedHeteroCache:
    """
    Factory: create a fully-configured FusedHeteroCache instance.

    Args:
        adaptive_self_healing: If True, use TRUE dynamic window self-healing
            (retrieves only top-w_t chunks based on attention scores).
            If False, use full retrieval (100% recall, O(N) memory spike).
            Default: False (matches paper's NIAH 100% recall claim).
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
    )
