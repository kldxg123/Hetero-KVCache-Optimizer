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
from typing import Any, Dict, List, Optional, Tuple

import torch

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.quantization.kv_compressor import KVCompressor
from src.memory.dram_storage import DRAMStorageManager
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

        # HBM-resident physical pools: one slot per layer
        self._key_cache: List[Optional[torch.Tensor]] = [None] * num_layers
        self._value_cache: List[Optional[torch.Tensor]] = [None] * num_layers
        self._seq_offsets: List[int] = [0] * num_layers  # logical start offset for each layer

        # Compression engine
        self._compressor = KVCompressor(group_size=group_size, bits=bits)

        # Heavy Hitter Oracle for attention-driven eviction (Phase D/E)
        self._oracle = HeavyHitterOracle(
            block_size=16,
            sink_tokens=sink_tokens,
            local_window=hbm_budget_tokens,
        )

        # Adaptive prefetch controller (Phase F)
        self._adaptive_controller = AdaptivePrefetchController()

        # Async prefetcher for DRAM -> HBM overlap
        self._prefetcher: Optional[AsyncPrefetcher] = None
        if enable_prefetch and torch.cuda.is_available():
            self._prefetcher = AsyncPrefetcher(device=torch.device(device))

        # Predictive prefetch scheduler (initialized lazily when DRAM entries exist)
        self._predictive_scheduler: Optional[PredictivePrefetchScheduler] = None

        # DRAM tier-2 storage: managed by DRAMStorageManager (pinned CPU memory)
        self._dram = DRAMStorageManager()
        self._eviction_counter = 0

        # Adaptive self-healing: track chunk metadata for dynamic window retrieval
        self._chunk_eviction_order: List[str] = []  # Track eviction order
        self._chunk_attention_scores: Dict[str, float] = {}  # chunk_key -> avg score

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
            In prefill mode, this returns the *full* original tensors to maintain
            FlashAttention compatibility (transient architecture).
            In decode mode, this returns the *pruned* HBM-resident tensors.
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

    def update_attention_scores(self, attention_weights: torch.Tensor) -> None:
        """
        Phase D: Feed attention weights from the latest decode step to the
        HeavyHitterOracle for cumulative importance tracking.
        """
        self._oracle.update(attention_weights)

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
        return self.sink_tokens + self.hbm_budget_tokens

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
        Transient interception architecture:
          1. Internally retain only Sink + Tail in the HBM physical pool.
          2. Evict overflow tail to DRAM via 4-bit quantization.
          3. Return the FULL original tensors to preserve FlashAttention compatibility.
        """
        new_len = key_states.shape[-2]
        max_hbm = self.max_hbm_tokens()

        # Robustness: auto-expand internal lists if num_layers was under-estimated
        while len(self._key_cache) <= layer_idx:
            self._key_cache.append(None)
            self._value_cache.append(None)
            self._seq_offsets.append(0)

        if self._key_cache[layer_idx] is None:
            # First write: crop to sink + tail, evict body to DRAM
            sink_amt = min(new_len, self.sink_tokens)
            tail_amt = min(new_len - sink_amt, self.hbm_budget_tokens)

            k_sink = key_states[..., :sink_amt, :]
            v_sink = value_states[..., :sink_amt, :]

            if tail_amt > 0:
                k_tail = key_states[..., -tail_amt:, :]
                v_tail = value_states[..., -tail_amt:, :]
                self._key_cache[layer_idx] = torch.cat([k_sink, k_tail], dim=-2)
                self._value_cache[layer_idx] = torch.cat([v_sink, v_tail], dim=-2)
            else:
                self._key_cache[layer_idx] = k_sink
                self._value_cache[layer_idx] = v_sink

            # Evict body (middle tokens between sink and tail) to DRAM
            if self.enable_quant and new_len > max_hbm:
                body_start = sink_amt
                body_end = new_len - tail_amt
                if body_end > body_start:
                    self._evict_to_dram(
                        layer_idx,
                        key_states[..., body_start:body_end, :],
                        value_states[..., body_start:body_end, :],
                    )
        else:
            # Append and maintain rolling window
            new_k = torch.cat([self._key_cache[layer_idx], key_states], dim=-2)
            new_v = torch.cat([self._value_cache[layer_idx], value_states], dim=-2)
            cur_len = new_k.shape[-2]

            if cur_len > max_hbm:
                overflow = cur_len - max_hbm
                evict_start = self.sink_tokens
                evict_end = evict_start + overflow

                if self.enable_quant:
                    self._evict_to_dram(
                        layer_idx,
                        new_k[..., evict_start:evict_end, :],
                        new_v[..., evict_start:evict_end, :],
                    )

                self._key_cache[layer_idx] = torch.cat(
                    [new_k[..., : self.sink_tokens, :], new_k[..., evict_end:, :]],
                    dim=-2,
                )
                self._value_cache[layer_idx] = torch.cat(
                    [new_v[..., : self.sink_tokens, :], new_v[..., evict_end:, :]],
                    dim=-2,
                )
            else:
                self._key_cache[layer_idx] = new_k
                self._value_cache[layer_idx] = new_v

            del new_k, new_v

        self._seq_offsets[layer_idx] = seq_offset + new_len

        # Return full tensors to keep FlashAttention kernels happy
        return key_states, value_states

    def _decode_update(
        self,
        layer_idx: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        seq_offset: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode residency: append new token to the physical pool and maintain
        the rolling window. When HBM is full, consult HeavyHitterOracle for
        attention-driven eviction instead of simple FIFO.
        """
        max_hbm = self.max_hbm_tokens()

        # Robustness: auto-expand internal lists if num_layers was under-estimated
        while len(self._key_cache) <= layer_idx:
            self._key_cache.append(None)
            self._value_cache.append(None)
            self._seq_offsets.append(0)

        k_cache = self._key_cache[layer_idx]
        v_cache = self._value_cache[layer_idx]

        if k_cache is None:
            self._key_cache[layer_idx] = key_states
            self._value_cache[layer_idx] = value_states
            self._seq_offsets[layer_idx] = seq_offset + 1
            return key_states, value_states

        cur_len = k_cache.shape[-2]
        if cur_len < max_hbm:
            new_k = torch.cat([k_cache, key_states], dim=-2)
            new_v = torch.cat([v_cache, value_states], dim=-2)
        else:
            # Oracle-driven eviction: find least important tokens to evict
            new_k = torch.cat([k_cache, key_states], dim=-2)
            new_v = torch.cat([v_cache, value_states], dim=-2)

            tokens_to_evict = 1  # One token per decode step

            if self._oracle.token_scores is not None and self.enable_quant:
                # Phase D+E: Use Triton-accelerated oracle for eviction decision
                current_seq_len = new_k.shape[-2]
                evict_candidates = self._oracle.get_eviction_candidates(
                    current_seq_len=current_seq_len,
                    evict_num_blocks=max(1, tokens_to_evict // self._oracle.block_size),
                )

                if evict_candidates.numel() > 0:
                    # Convert block indices to token ranges and build eviction mask
                    evict_mask = torch.zeros(current_seq_len, dtype=torch.bool, device=new_k.device)
                    for block_idx in evict_candidates:
                        start_tok = block_idx.item() * self._oracle.block_size
                        end_tok = min(start_tok + self._oracle.block_size, current_seq_len)
                        # Protect sink zone
                        if start_tok < self.sink_tokens:
                            continue
                        evict_mask[start_tok:end_tok] = True

                    if evict_mask.any():
                        keep_mask = ~evict_mask
                        evicted_k = new_k[..., evict_mask, :]
                        evicted_v = new_v[..., evict_mask, :]
                        self._evict_to_dram(layer_idx, evicted_k, evicted_v)

                        new_k = new_k[..., keep_mask, :]
                        new_v = new_v[..., keep_mask, :]
                    else:
                        # Fallback: simple rolling window
                        new_k = torch.cat(
                            [k_cache[..., :self.sink_tokens, :],
                             k_cache[..., self.sink_tokens + 1:, :],
                             key_states], dim=-2)
                        new_v = torch.cat(
                            [v_cache[..., :self.sink_tokens, :],
                             v_cache[..., self.sink_tokens + 1:, :],
                             value_states], dim=-2)
                        evicted_k = k_cache[..., self.sink_tokens:self.sink_tokens + 1, :]
                        evicted_v = v_cache[..., self.sink_tokens:self.sink_tokens + 1, :]
                        self._evict_to_dram(layer_idx, evicted_k, evicted_v)
                else:
                    # No candidates from oracle (all protected), fallback to FIFO
                    new_k = torch.cat(
                        [k_cache[..., :self.sink_tokens, :],
                         k_cache[..., self.sink_tokens + 1:, :],
                         key_states], dim=-2)
                    new_v = torch.cat(
                        [v_cache[..., :self.sink_tokens, :],
                         v_cache[..., self.sink_tokens + 1:, :],
                         value_states], dim=-2)
                    evicted_k = k_cache[..., self.sink_tokens:self.sink_tokens + 1, :]
                    evicted_v = v_cache[..., self.sink_tokens:self.sink_tokens + 1, :]
                    self._evict_to_dram(layer_idx, evicted_k, evicted_v)
            else:
                # No attention scores yet: FIFO eviction of oldest tail token
                new_k = torch.cat(
                    [k_cache[..., :self.sink_tokens, :],
                     k_cache[..., self.sink_tokens + 1:, :],
                     key_states], dim=-2)
                new_v = torch.cat(
                    [v_cache[..., :self.sink_tokens, :],
                     v_cache[..., self.sink_tokens + 1:, :],
                     value_states], dim=-2)
                evicted_k = k_cache[..., self.sink_tokens:self.sink_tokens + 1, :]
                evicted_v = v_cache[..., self.sink_tokens:self.sink_tokens + 1, :]
                self._evict_to_dram(layer_idx, evicted_k, evicted_v)

        self._key_cache[layer_idx] = new_k
        self._value_cache[layer_idx] = new_v
        self._seq_offsets[layer_idx] = seq_offset + 1

        return self._key_cache[layer_idx], self._value_cache[layer_idx]

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

    def _evict_to_dram(
        self,
        layer_idx: int,
        k_chunk: torch.Tensor,
        v_chunk: torch.Tensor,
    ) -> None:
        """Compress a KV chunk and move it to DRAM via DRAMStorageManager (pinned CPU memory)."""
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

        # DRAMStorageManager handles .cpu().pin_memory() internally
        self._dram.store_entry(chunk_key, entry)

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
                f"tokens={tokens} DRAM_entries={self._dram.num_entries}"
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
