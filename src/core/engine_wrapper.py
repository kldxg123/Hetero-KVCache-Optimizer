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
    from src.quantization.fused_dequant_attn import fused_dequant_attention
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

        # Deferred initialization: num_layers is set on first update
        self._num_layers = num_layers
        self._manager: Optional[HeteroKVManager] = None

        # Global sequence length tracker for RoPE alignment
        self.real_seq_len: int = 0

        print(
            f"[FusedHeteroCache] Initialized | "
            f"sink={sink_tokens} tail={keep_tail} chunk={chunk_size} "
            f"quant={'ON' if enable_quant else 'OFF'} "
            f"prefetch={'ON' if enable_prefetch else 'OFF'} "
            f"triton={'ON' if self.enable_triton else 'OFF'}"
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

        if layer_idx == 0:
            self.real_seq_len += new_len

        return out_k, out_v

    def get_seq_length(self, layer_idx: int = 0) -> int:
        # Report the physical cache size, not total seen tokens.
        # This is critical for correct attention mask dimensions in transformers 4.57+
        if self._manager is None:
            return 0
        k_cache = self._manager._key_cache
        if layer_idx >= len(k_cache) or k_cache[layer_idx] is None:
            return 0
        return k_cache[layer_idx].shape[-2]

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
        For decode: return (physical_pool_size + 1, 0) to match actual KV after update.
        """
        if self._manager is None:
            return cache_position.shape[0], 0
        k_cache = self._manager._key_cache
        if layer_idx >= len(k_cache) or k_cache[layer_idx] is None:
            return cache_position.shape[0], 0
        physical_size = k_cache[layer_idx].shape[-2]
        # Return the size AFTER the update (includes new token for decode)
        # The +1 accounts for the token being added during this forward pass
        query_len = cache_position.shape[0]
        if query_len > 1:  # Prefill chunk: mask width = chunk_size
            return query_len, 0
        else:  # Decode: mask width = current pool size + new token
            return physical_size + 1, 0

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
        """Swap a compressed chunk back into HBM."""
        if self._manager is None:
            return None
        return self._manager.swap_in(0, chunk_key)


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
) -> FusedHeteroCache:
    """Factory: create a fully-configured FusedHeteroCache instance."""
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
    )
