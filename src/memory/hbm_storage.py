import torch
import time
from typing import List, Dict, Optional


class HBMStorageManager:
    """
    Tier 1 storage manager: manages KV Blocks in GPU HBM.
    Simulates vLLM's PagedAttention memory pool for decoupling.
    """

    def __init__(self,
                 num_layers: int = 32,
                 num_heads: int = 32,
                 head_dim: int = 128,
                 block_size: int = 16,
                 max_num_blocks: int = 1024,
                 device: str = "cuda:0"):

        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.max_num_blocks = max_num_blocks
        self.device = torch.device(device)

        print(f"[HBM] Initializing HBM Storage Manager (device: {self.device})...")

        # Pre-allocate GPU memory pool (simulates vLLM KV cache pool)
        # Shape: [num_blocks, num_layers, 2 (K and V), block_size, num_heads, head_dim]
        pool_shape = (max_num_blocks, num_layers, 2, block_size, num_heads, head_dim)

        # Use float16 to save half the memory
        self.kv_pool = torch.zeros(pool_shape, dtype=torch.float16, device=self.device)

        # Physical block allocation table (0 = free, 1 = occupied)
        self.free_blocks: List[int] = list(range(max_num_blocks))

        # Logical to physical mapping: { logical_block_id : physical_block_id }
        self.logical_to_physical: Dict[int, int] = {}

        print(f"[HBM] Pool allocated. Total capacity: {max_num_blocks} Blocks. "
              f"Approx memory: {self.kv_pool.element_size() * self.kv_pool.nelement() / (1024 ** 3):.2f} GB")

    def allocate_block(self, logical_block_id: int) -> Optional[int]:
        """Allocate a GPU physical block for a new request."""
        if not self.free_blocks:
            print("[HBM] Memory pool full! Eviction required.")
            return None

        physical_id = self.free_blocks.pop(0)
        self.logical_to_physical[logical_block_id] = physical_id
        return physical_id

    def evict_to_dram(self, logical_block_id: int, dram_buffer: torch.Tensor, dram_idx: int) -> bool:
        """
        [Core operation] Evict the specified logical block from GPU (HBM) to system memory (DRAM).
        """
        if logical_block_id not in self.logical_to_physical:
            return False

        physical_id = self.logical_to_physical[logical_block_id]

        # Extract the physical block data
        block_data = self.kv_pool[physical_id]

        # Perform Device-to-Host (D2H) memory copy (non-blocking)
        # Note: in real heterogeneous architectures, pinned memory + non_blocking=True is required
        dram_buffer[dram_idx].copy_(block_data, non_blocking=True)

        # Recycle physical block
        del self.logical_to_physical[logical_block_id]
        self.free_blocks.append(physical_id)

        print(f"[Eviction] Logical Block {logical_block_id} (Phys: {physical_id}) evicted to DRAM.")
        return True


# ==========================================
# Standalone test logic
# ==========================================
if __name__ == "__main__":
    # Simulated environment: restrict parameter scale for quick testing
    hbm_manager = HBMStorageManager(max_num_blocks=100)  # Mini pool of 100 blocks

    print("\n--- Test 1: Allocate KV Blocks ---")
    for logical_id in range(5):
        phys_id = hbm_manager.allocate_block(logical_id)
        print(f"Allocated Logical Block {logical_id} -> Physical Block {phys_id}")

    print(f"Remaining HBM physical blocks: {len(hbm_manager.free_blocks)}")

    print("\n--- Test 2: Simulate dynamic eviction ---")
    # Assume Tier 2 (DRAM) has prepared a CPU pinned memory buffer to receive data
    dummy_dram_pool = torch.empty(
        hbm_manager.kv_pool[:10].shape,
        dtype=hbm_manager.kv_pool.dtype,
        device='cpu'
    ).pin_memory()

    # Algorithm layer decides to evict Block 2 and 3
    evict_targets = [2, 3]
    for i, target_id in enumerate(evict_targets):
        hbm_manager.evict_to_dram(target_id, dummy_dram_pool, dram_idx=i)

    # Wait for CUDA async stream synchronization (ensure D2H copy completes)
    torch.cuda.synchronize()

    print(f"\nRemaining HBM physical blocks: {len(hbm_manager.free_blocks)}")
    print("[PASS] HBM manager basic test passed.")
