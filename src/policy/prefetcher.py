"""
src/policy/prefetcher.py
========================
Async Prefetcher: DRAM -> HBM computation-communication overlap.

Protocol:
  dram_data dict contains the following keys (consistent with FusedHeteroCache._evict_to_dram):
    "k_data", "k_scales", "k_zps"   <- Key quantized data
    "v_data", "v_scales", "v_zps"   <- Value quantized data

The main thread calls submit_prefetch_task(chunk_{i+1}) while computing chunk_i,
then calls fetch_if_ready(chunk_{i+1}) after chunk_i finishes to obtain
the decompressed BF16 (K, V) pair.
"""

import torch
from typing import Dict, Any, Optional, Tuple
from contextlib import nullcontext


class AsyncPrefetcher:
    """
    Background async prefetcher: uses an independent CUDA Stream to move and decompress data
    concurrently with the main compute stream.

    Supports two storage protocols:
      A) Split K/V protocol (from FusedHeteroCache):
         keys: k_data, k_scales, k_zps, v_data, v_scales, v_zps
      B) Merged protocol (from legacy HeteroKVManager):
         keys: q_data, scales, zps  (only K, V handled externally)
    """

    def __init__(self, device: torch.device):
        self.device = device
        # Dedicated background CUDA stream, concurrent with main compute stream for H2D copy and decompression
        if device.type == "cuda":
            self.prefetch_stream = torch.cuda.Stream(device=device)
        else:
            self.prefetch_stream = None  # CPU mode: no async stream
        # Prefetch buffer: chunk_key -> (restored_k, restored_v) BF16
        self.buffer: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

    # ------------------------------------------------------------------
    # Submit task (non-blocking)
    # ------------------------------------------------------------------

    def submit_prefetch_task(
        self,
        chunk_key: str,
        dram_data: Dict[str, Any],
        compressor,
    ) -> None:
        """
        Submit a prefetch task to the background, returns immediately without blocking the main thread.

        Args:
            chunk_key: unique identifier (e.g. "l0_e3")
            dram_data: quantized data dict in DRAM
            compressor: KVCompressor instance for decompression
        """
        if chunk_key in self.buffer:
            return  # Already prefetched, skip

        # Determine protocol type
        has_split_kv = "k_data" in dram_data and "v_data" in dram_data
        has_merged = "q_data" in dram_data

        if not has_split_kv and not has_merged:
            return  # Unknown protocol, skip

        ctx = torch.cuda.stream(self.prefetch_stream) if self.prefetch_stream else nullcontext()
        with ctx:
            if has_split_kv:
                # Protocol A: K/V split
                q_k = dram_data["k_data"].to(self.device, non_blocking=True)
                s_k = dram_data["k_scales"].to(self.device, non_blocking=True)
                z_k = dram_data["k_zps"].to(self.device, non_blocking=True)
                q_v = dram_data["v_data"].to(self.device, non_blocking=True)
                s_v = dram_data["v_scales"].to(self.device, non_blocking=True)
                z_v = dram_data["v_zps"].to(self.device, non_blocking=True)

                restored_k = compressor.decompress(q_k, s_k, z_k).to(torch.bfloat16)
                restored_v = compressor.decompress(q_v, s_v, z_v).to(torch.bfloat16)
                self.buffer[chunk_key] = (restored_k, restored_v)

            else:
                # Protocol B: legacy merged (only K)
                q_data = dram_data["q_data"].to(self.device, non_blocking=True)
                scales = dram_data["scales"].to(self.device, non_blocking=True)
                zps = dram_data["zps"].to(self.device, non_blocking=True)

                restored_kv = compressor.decompress(q_data, scales, zps).to(torch.bfloat16)
                # Old protocol lacks V, use K as placeholder (caller handles V externally)
                self.buffer[chunk_key] = (restored_kv, restored_kv)

    # ------------------------------------------------------------------
    # Fetch result (called by main thread)
    # ------------------------------------------------------------------

    def fetch_if_ready(self, chunk_key: str) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Called by main thread to retrieve the prefetched (K, V) BF16 tensor pair.
        Returns None if not ready (or not submitted).
        """
        if chunk_key not in self.buffer:
            return None

        # Wait for background stream H2D copy and decompression to complete
        if self.prefetch_stream is not None:
            torch.cuda.current_stream().wait_stream(self.prefetch_stream)
        return self.buffer.pop(chunk_key)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Clear buffer (call after inference ends)."""
        self.buffer.clear()

    @property
    def pending_keys(self):
        return list(self.buffer.keys())
