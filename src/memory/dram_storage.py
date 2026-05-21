"""
src/memory/dram_storage.py
==========================
DRAMStorageManager: Tier 2 storage for compressed KV cache blocks.

Manages 4-bit quantized KV entries in CPU pinned (page-locked) memory.
Pinned memory enables zero-copy PCIe DMA transfers between DRAM and HBM,
overlapping H2D communication with GPU computation.

Each entry is a dict of tensors:
  {"k_data", "k_scales", "k_zps", "v_data", "v_scales", "v_zps"}

All tensors are stored on CPU with pin_memory() for async DMA.
"""

import torch
from typing import Any, Dict, List, Optional, Tuple


class DRAMStorageManager:
    """
    Tier 2 storage manager: holds compressed KV blocks in CPU pinned memory.

    Provides a dict-like interface (store / retrieve / remove / contains)
    while tracking memory consumption and enforcing optional capacity limits.
    """

    def __init__(
        self,
        max_entries: Optional[int] = None,
        max_bytes: Optional[int] = None,
    ):
        """
        Args:
            max_entries: optional cap on number of stored chunks (None = unlimited).
            max_bytes: optional cap on total pinned memory in bytes (None = unlimited).
        """
        self.max_entries = max_entries
        self.max_bytes = max_bytes

        # Primary storage: chunk_key -> {k_data, k_scales, k_zps, v_data, v_scales, v_zps}
        self._table: Dict[str, Dict[str, torch.Tensor]] = {}

        # Running memory counter (avoids recomputing from all tensors each time)
        self._total_bytes: int = 0

        print(
            f"[DRAM] Storage Manager initialized | "
            f"max_entries={max_entries or 'inf'} "
            f"max_bytes={max_bytes or 'inf'}"
        )

    # ------------------------------------------------------------------
    # Core CRUD operations
    # ------------------------------------------------------------------

    def store(
        self,
        chunk_key: str,
        q_k: torch.Tensor,
        k_scales: torch.Tensor,
        k_zps: torch.Tensor,
        q_v: torch.Tensor,
        v_scales: torch.Tensor,
        v_zps: torch.Tensor,
    ) -> bool:
        """
        Compress-then-store: move quantized KV data to CPU pinned memory.

        Args:
            chunk_key: unique identifier (e.g. "l0_e3").
            q_k, k_scales, k_zps: compressed key tensors.
            q_v, v_scales, v_zps: compressed value tensors.

        Returns:
            True if stored successfully, False if capacity exceeded.
        """
        # Capacity checks
        if self.max_entries is not None and len(self._table) >= self.max_entries:
            print(f"[DRAM] Capacity exceeded: {len(self._table)}/{self.max_entries} entries")
            return False

        entry_bytes = self._estimate_entry_size(q_k, k_scales, k_zps, q_v, v_scales, v_zps)
        if self.max_bytes is not None and self._total_bytes + entry_bytes > self.max_bytes:
            print(f"[DRAM] Memory cap exceeded: {self._total_bytes + entry_bytes}/{self.max_bytes} bytes")
            return False

        # If key already exists, subtract old entry's size
        if chunk_key in self._table:
            self._total_bytes -= self._entry_bytes(chunk_key)

        # Transfer to CPU pinned memory for zero-copy DMA
        entry = {
            "k_data": q_k.cpu().pin_memory(),
            "k_scales": k_scales.cpu().pin_memory(),
            "k_zps": k_zps.cpu().pin_memory(),
            "v_data": q_v.cpu().pin_memory(),
            "v_scales": v_scales.cpu().pin_memory(),
            "v_zps": v_zps.cpu().pin_memory(),
        }
        self._table[chunk_key] = entry
        self._total_bytes += entry_bytes

        return True

    def store_entry(self, chunk_key: str, entry: Dict[str, torch.Tensor]) -> bool:
        """
        Store a pre-built entry dict (convenience wrapper for _evict_to_dram compat).

        The entry must contain all six keys: k_data, k_scales, k_zps,
        v_data, v_scales, v_zps.
        """
        required = ("k_data", "k_scales", "k_zps", "v_data", "v_scales", "v_zps")
        if not all(k in entry for k in required):
            raise ValueError(f"Entry missing required keys. Need: {required}")

        # Ensure all tensors are on pinned CPU memory
        pinned_entry = {}
        entry_bytes = 0
        for key in required:
            t = entry[key]
            if t.device.type != "cpu" or not t.is_pinned():
                t = t.cpu().pin_memory()
            pinned_entry[key] = t
            entry_bytes += t.element_size() * t.nelement()

        # Capacity checks
        if self.max_entries is not None and len(self._table) >= self.max_entries:
            return False
        if self.max_bytes is not None and self._total_bytes + entry_bytes > self.max_bytes:
            return False

        if chunk_key in self._table:
            self._total_bytes -= self._entry_bytes(chunk_key)

        self._table[chunk_key] = pinned_entry
        self._total_bytes += entry_bytes
        return True

    def retrieve(self, chunk_key: str) -> Optional[Dict[str, torch.Tensor]]:
        """
        Retrieve a stored entry without removing it.

        Returns:
            Entry dict with pinned CPU tensors, or None if not found.
        """
        return self._table.get(chunk_key)

    def remove(self, chunk_key: str) -> Optional[Dict[str, torch.Tensor]]:
        """
        Remove and return an entry (pop semantics).

        Returns:
            The removed entry dict, or None if not found.
        """
        entry = self._table.pop(chunk_key, None)
        if entry is not None:
            self._total_bytes -= self._entry_bytes_from(entry)
        return entry

    def contains(self, chunk_key: str) -> bool:
        return chunk_key in self._table

    # ------------------------------------------------------------------
    # Transfer helpers
    # ------------------------------------------------------------------

    def transfer_to_device(
        self,
        chunk_key: str,
        device: torch.device,
        non_blocking: bool = True,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        Transfer a stored entry from pinned DRAM to GPU HBM.

        Uses non-blocking H2D copy enabled by pinned memory for
        compute-communication overlap.

        Returns:
            Entry dict with tensors on `device`, or None if not found.
        """
        entry = self._table.get(chunk_key)
        if entry is None:
            return None

        result = {}
        for key in ("k_data", "k_scales", "k_zps", "v_data", "v_scales", "v_zps"):
            t = entry.get(key)
            if t is None:
                return None
            result[key] = t.to(device, non_blocking=non_blocking)

        return result

    # ------------------------------------------------------------------
    # Query / stats
    # ------------------------------------------------------------------

    @property
    def table(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """Direct access to the storage dict for compatibility."""
        return self._table

    @property
    def total_bytes(self) -> int:
        return self._total_bytes

    @property
    def num_entries(self) -> int:
        return len(self._table)

    @property
    def keys(self) -> List[str]:
        return list(self._table.keys())

    def memory_summary(self) -> Dict[str, Any]:
        return {
            "num_entries": len(self._table),
            "total_bytes": self._total_bytes,
            "total_mb": self._total_bytes / (1024 * 1024),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_entry_size(
        q_k: torch.Tensor,
        k_scales: torch.Tensor,
        k_zps: torch.Tensor,
        q_v: torch.Tensor,
        v_scales: torch.Tensor,
        v_zps: torch.Tensor,
    ) -> int:
        total = 0
        for t in (q_k, k_scales, k_zps, q_v, v_scales, v_zps):
            total += t.element_size() * t.nelement()
        return total

    @staticmethod
    def _entry_bytes_from(entry: Dict[str, torch.Tensor]) -> int:
        total = 0
        for t in entry.values():
            total += t.element_size() * t.nelement()
        return total

    def _entry_bytes(self, chunk_key: str) -> int:
        entry = self._table.get(chunk_key)
        if entry is None:
            return 0
        return self._entry_bytes_from(entry)

    # ------------------------------------------------------------------
    # Standalone test
    # ------------------------------------------------------------------


if __name__ == "__main__":
    print("=== DRAMStorageManager Unit Test ===\n")

    dram = DRAMStorageManager()

    # Simulate quantized KV block
    k_data = torch.randint(0, 15, (1, 2, 16, 128), dtype=torch.uint8)
    k_scales = torch.randn(4, dtype=torch.float32)
    k_zps = torch.randint(0, 15, (4,), dtype=torch.uint8)
    v_data = torch.randint(0, 15, (1, 2, 16, 128), dtype=torch.uint8)
    v_scales = torch.randn(4, dtype=torch.float32)
    v_zps = torch.randint(0, 15, (4,), dtype=torch.uint8)

    # Store
    assert dram.store("l0_e0", k_data, k_scales, k_zps, v_data, v_scales, v_zps)
    assert dram.num_entries == 1
    print(f"  Stored l0_e0: {dram.memory_summary()}")

    # Retrieve
    entry = dram.retrieve("l0_e0")
    assert entry is not None
    assert entry["k_data"].is_pinned()
    print(f"  Retrieved l0_e0: pinned={entry['k_data'].is_pinned()}")

    # Contains
    assert dram.contains("l0_e0")
    assert not dram.contains("l0_e99")

    # Remove
    removed = dram.remove("l0_e0")
    assert removed is not None
    assert dram.num_entries == 0
    assert dram.total_bytes == 0

    # store_entry compat
    entry_dict = {
        "k_data": k_data, "k_scales": k_scales, "k_zps": k_zps,
        "v_data": v_data, "v_scales": v_scales, "v_zps": v_zps,
    }
    assert dram.store_entry("l1_e0", entry_dict)
    assert dram.num_entries == 1

    print("\n  [PASS] All DRAMStorageManager tests passed.")
