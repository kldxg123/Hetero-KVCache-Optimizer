import torch
import math
from typing import List

from src.kernels.oracle_triton import compute_block_scores


class HeavyHitterOracle:
    """
    The "brain" of the dynamic eviction algorithm.
    Tracks attention scores to decide which Logical Blocks to evict when HBM capacity is tight.

    Phase-1 Kernelization: all eviction-candidate selection logic is now executed
    entirely on the GPU via a fused Triton kernel, eliminating Python for-loops
    and CPU-GPU synchronization points.
    """

    def __init__(self, block_size: int = 16, sink_tokens: int = 32, local_window: int = 128):
        self.block_size = block_size
        self.sink_tokens = sink_tokens  # Protected zone 1: initial system prompts etc (Sink)
        self.local_window = local_window  # Protected zone 2: most recently generated context (Local)
        self.token_scores = None  # Cumulative attention score per token

        print(f"[HeavyHitter] Initialized (Block={block_size}, Sink={sink_tokens}, Local={local_window})")

    def update(self, recent_attention: torch.Tensor):
        """
        Called at each decode step to update cumulative weights of historical tokens.
        recent_attention: shape [seq_len], attention weights from the newest token to all historical tokens.
        """
        seq_len = recent_attention.shape[0]

        # 1. Dynamically expand score board on first run or when sequence length increases
        if self.token_scores is None or self.token_scores.shape[0] < seq_len:
            new_scores = torch.zeros(seq_len, dtype=torch.float32, device=recent_attention.device)
            if self.token_scores is not None:
                new_scores[:self.token_scores.shape[0]] = self.token_scores
            self.token_scores = new_scores

        # 2. Accumulate importance scores (this represents S_i computation)
        self.token_scores[:seq_len] += recent_attention

    def get_eviction_candidates(self, current_seq_len: int, evict_num_blocks: int) -> torch.Tensor:
        """
        Core decision logic: find the least important Block IDs.
        Returns a GPU-resident LongTensor of block indices to evict.
        All operations are fused on the GPU; no CPU-GPU synchronization occurs.
        """
        # No eviction needed if current length is within protected zones
        if current_seq_len <= self.sink_tokens + self.local_window:
            device = self.token_scores.device if self.token_scores is not None else (
                recent_attention.device if 'recent_attention' in dir() else torch.device('cpu')
            )
            return torch.empty(0, dtype=torch.long, device=device)

        num_blocks = math.ceil(current_seq_len / self.block_size)

        # Compute protected zone boundaries (Masking)
        sink_blocks = math.ceil(self.sink_tokens / self.block_size)
        local_blocks = math.ceil(self.local_window / self.block_size)

        # ---------------------------------------------------------
        # Cold-start protection
        # If attention scores have not been received yet, default to evicting
        # from the oldest block outside the Sink protection (FIFO policy).
        # ---------------------------------------------------------
        if self.token_scores is None:
            first_evictable = sink_blocks
            last_evictable = num_blocks - local_blocks
            count = max(0, min(evict_num_blocks, last_evictable - first_evictable))
            if count <= 0:
                return torch.empty(0, dtype=torch.long, device='cpu')
            return torch.arange(first_evictable, first_evictable + count, dtype=torch.long, device='cpu')

        # ---------------------------------------------------------
        # Triton-accelerated path: block-score aggregation on GPU
        # ---------------------------------------------------------
        block_scores = compute_block_scores(self.token_scores, current_seq_len, self.block_size)

        # Build safe mask on GPU (no sync)
        safe_mask = torch.zeros(num_blocks, dtype=torch.bool, device=block_scores.device)
        if sink_blocks > 0:
            safe_mask[:sink_blocks] = True
        if local_blocks > 0:
            safe_mask[max(0, num_blocks - local_blocks):] = True

        # Protected blocks get +inf so they are never selected by bottom-k
        block_scores[safe_mask] = float('inf')

        # Number of blocks we are allowed to evict is bounded by unsafe blocks
        unsafe_count = max(0, num_blocks - sink_blocks - local_blocks)
        k = min(evict_num_blocks, unsafe_count)

        if k <= 0:
            return torch.empty(0, dtype=torch.long, device=block_scores.device)

        # GPU-native top-k (largest=False => lowest scores)
        # sorted=True guarantees deterministic output for easier testing
        _, candidates = torch.topk(block_scores, k=k, largest=False, sorted=True)

        return candidates.to(torch.long)


# ==========================================
# Standalone test logic
# ==========================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Simulated environment: Block size 16, protect first 32 tokens, protect last 64 tokens
    oracle = HeavyHitterOracle(block_size=16, sink_tokens=32, local_window=64)

    current_length = 256  # Assume current context length is 256 (16 Blocks total)

    # Simulate attention accumulation during generation
    print("\n--- Simulating attention accumulation during inference ---")
    for step in range(5):
        simulated_attn = torch.rand(current_length, device=device) * 0.1
        simulated_attn[64:96] = 0.001  # Corresponds to Block 4 and Block 5
        oracle.update(simulated_attn)

    print("\n--- Triggering Eviction Decision ---")
    targets = oracle.get_eviction_candidates(current_seq_len=current_length, evict_num_blocks=2)

    print(f"Total Blocks: {math.ceil(current_length / 16)}")
    print(f"Decision: Logical Block IDs to evict: {targets.tolist()}")
