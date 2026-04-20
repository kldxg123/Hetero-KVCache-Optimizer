"""
src/kernels/oracle_triton.py
===========================
Fused Attention-Aware Scheduler Kernel.

Eliminates CPU-GPU synchronization and Python loops from the Heavy Hitter
Oracle's eviction candidate selection by fusing block-level score aggregation
into a single Triton kernel.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _block_mean_kernel(
    token_scores_ptr,
    block_scores_ptr,
    seq_len,
    block_size: tl.constexpr,
):
    """
    Each CUDA block computes the mean attention score for one logical block.
    `block_size` is promoted to a compile-time constant so Triton can unroll
    the inner loop without dynamic break statements.
    """
    pid = tl.program_id(0)
    start_idx = pid * block_size
    end_idx = tl.minimum(start_idx + block_size, seq_len)

    if start_idx >= seq_len:
        tl.store(block_scores_ptr + pid, 0.0)
        return

    BLOCK_SZ: tl.constexpr = 256
    acc = 0.0
    # Fixed iteration count derived from constexpr block_size
    num_iters = tl.cdiv(block_size, BLOCK_SZ)
    for i in range(num_iters):
        curr_start = start_idx + i * BLOCK_SZ
        offs = curr_start + tl.arange(0, BLOCK_SZ)
        # mask guards both against seq_len boundary and unrolled loop overrun
        mask = offs < end_idx
        vals = tl.load(token_scores_ptr + offs, mask=mask, other=0.0)
        acc += tl.sum(vals, axis=0)

    count = end_idx - start_idx
    mean_val = acc / tl.maximum(count, 1)
    tl.store(block_scores_ptr + pid, mean_val)


def compute_block_scores(token_scores: torch.Tensor, current_seq_len: int, block_size: int) -> torch.Tensor:
    """
    Compute per-block mean attention scores entirely on the GPU.

    Args:
        token_scores: [max_seq_len] cumulative attention scores (float32).
        current_seq_len: active sequence length.
        block_size: number of tokens per logical block.

    Returns:
        block_scores: [num_blocks] mean score per block.
    """
    num_blocks = (current_seq_len + block_size - 1) // block_size
    block_scores = torch.empty(num_blocks, dtype=token_scores.dtype, device=token_scores.device)

    # Fast-path for token-level granularity avoids kernel launch overhead.
    if block_size == 1:
        block_scores.copy_(token_scores[:current_seq_len])
        return block_scores

    grid = (num_blocks,)
    # Pass block_size as a keyword argument so Triton treats it as constexpr.
    _block_mean_kernel[grid](
        token_scores, block_scores,
        current_seq_len, block_size=block_size,
    )
    return block_scores
