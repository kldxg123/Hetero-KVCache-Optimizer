"""
Triton scoring kernel for DRAM-resident uint8 INT4 Key chunks.

The current KVCompressor stores 4-bit values in uint8 tensors, with group-wise
scale/zero-point vectors over the flattened tensor.  This kernel intentionally
does only retrieval scoring: it dequantizes K inside the kernel and reduces
QK scores to per-block maxima, avoiding materializing the full FP16/BF16 Key
chunk or a full QK score matrix in HBM.
"""

from __future__ import annotations

from typing import Dict

import torch

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except Exception:  # pragma: no cover - exercised on CPU-only environments
    triton = None
    tl = None
    _TRITON_AVAILABLE = False


SUPPORTED_REDUCERS = {"max", "query_top_r_mean", "query_mean_max"}


def is_available() -> bool:
    return bool(_TRITON_AVAILABLE and torch.cuda.is_available())


if _TRITON_AVAILABLE:

    @triton.jit
    def _int4_score_blocks_kernel(
        q_ptr,
        k_ptr,
        scale_ptr,
        zp_ptr,
        block_score_ptr,
        block_offset_ptr,
        total_programs: tl.constexpr,
        qh_total: tl.constexpr,
        kvh_total: tl.constexpr,
        q_len: tl.constexpr,
        kv_len: tl.constexpr,
        head_dim: tl.constexpr,
        num_blocks: tl.constexpr,
        group_size: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_q = tl.program_id(0)
        pid_block = tl.program_id(1)

        q_pos = pid_q % q_len
        tmp = pid_q // q_len
        qh = tmp % qh_total
        batch = tmp // qh_total

        kv_group = qh_total // kvh_total
        kvh = qh // kv_group

        offs_d = tl.arange(0, BLOCK_D)
        q_base = ((batch * qh_total + qh) * q_len + q_pos) * head_dim
        q_vals = tl.load(
            q_ptr + q_base + offs_d,
            mask=offs_d < head_dim,
            other=0.0,
        ).to(tl.float32)

        offs_k = pid_block * BLOCK_K + tl.arange(0, BLOCK_K)
        k_offsets = (
            ((batch * kvh_total + kvh) * kv_len + offs_k[:, None]) * head_dim
            + offs_d[None, :]
        )
        mask = (offs_k[:, None] < kv_len) & (offs_d[None, :] < head_dim)
        q_int = tl.load(k_ptr + k_offsets, mask=mask, other=0).to(tl.float32)
        group_offsets = k_offsets // group_size
        scales = tl.load(scale_ptr + group_offsets, mask=mask, other=1.0).to(tl.float32)
        zps = tl.load(zp_ptr + group_offsets, mask=mask, other=0.0).to(tl.float32)
        k_vals = ((q_int - zps) * scales).to(tl.float16).to(tl.float32)

        scores = tl.sum(k_vals * q_vals[None, :], axis=1)
        scores = tl.where(offs_k < kv_len, scores, -3.4028234663852886e38)
        max_score = tl.max(scores, axis=0)
        winner = tl.max(
            tl.where(scores == max_score, offs_k.to(tl.float32), -1.0),
            axis=0,
        ).to(tl.int32)

        out_offset = pid_q * num_blocks + pid_block
        tl.store(block_score_ptr + out_offset, max_score)
        tl.store(block_offset_ptr + out_offset, winner)

    @triton.jit
    def _int4_score_chunk_batch_kernel(
        q_ptr,
        k_ptr,
        scale_ptr,
        zp_ptr,
        block_score_ptr,
        block_offset_ptr,
        batch_total: tl.constexpr,
        qh_total: tl.constexpr,
        kvh_total: tl.constexpr,
        q_len: tl.constexpr,
        kv_len: tl.constexpr,
        head_dim: tl.constexpr,
        num_blocks: tl.constexpr,
        group_size: tl.constexpr,
        num_groups_per_chunk: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid = tl.program_id(0)
        pid_block = tl.program_id(1)

        q_units = batch_total * qh_total * q_len
        chunk = pid // q_units
        inner = pid - chunk * q_units
        q_pos = inner % q_len
        tmp = inner // q_len
        qh = tmp % qh_total
        batch = tmp // qh_total

        kv_group = qh_total // kvh_total
        kvh = qh // kv_group

        offs_d = tl.arange(0, BLOCK_D)
        q_base = ((batch * qh_total + qh) * q_len + q_pos) * head_dim
        q_vals = tl.load(
            q_ptr + q_base + offs_d,
            mask=offs_d < head_dim,
            other=0.0,
        ).to(tl.float32)

        offs_k = pid_block * BLOCK_K + tl.arange(0, BLOCK_K)
        inner_k_offsets = (
            ((batch * kvh_total + kvh) * kv_len + offs_k[:, None]) * head_dim
            + offs_d[None, :]
        )
        chunk_elements = batch_total * kvh_total * kv_len * head_dim
        k_offsets = chunk * chunk_elements + inner_k_offsets
        mask = (offs_k[:, None] < kv_len) & (offs_d[None, :] < head_dim)
        q_int = tl.load(k_ptr + k_offsets, mask=mask, other=0).to(tl.float32)
        group_offsets = chunk * num_groups_per_chunk + inner_k_offsets // group_size
        scales = tl.load(scale_ptr + group_offsets, mask=mask, other=1.0).to(tl.float32)
        zps = tl.load(zp_ptr + group_offsets, mask=mask, other=0.0).to(tl.float32)
        k_vals = ((q_int - zps) * scales).to(tl.float16).to(tl.float32)

        scores = tl.sum(k_vals * q_vals[None, :], axis=1)
        scores = tl.where(offs_k < kv_len, scores, -3.4028234663852886e38)
        max_score = tl.max(scores, axis=0)
        winner = tl.max(
            tl.where(scores == max_score, offs_k.to(tl.float32), -1.0),
            axis=0,
        ).to(tl.int32)

        out_offset = pid * num_blocks + pid_block
        tl.store(block_score_ptr + out_offset, max_score)
        tl.store(block_offset_ptr + out_offset, winner)


def _reduce_block_scores(
    block_scores: torch.Tensor,
    block_offsets: torch.Tensor,
    score_reduce: str,
    top_r: int,
) -> Dict[str, float | int]:
    if score_reduce == "query_top_r_mean":
        per_query = block_scores.float().amax(dim=(0, 1, 3))
        if per_query.numel() == 0:
            score = float("-inf")
        else:
            k = min(max(1, int(top_r)), per_query.numel())
            score = float(torch.topk(per_query, k=k, largest=True).values.mean().item())
    elif score_reduce == "query_mean_max":
        per_query = block_scores.float().amax(dim=(0, 1, 3))
        score = float(per_query.mean().item()) if per_query.numel() else float("-inf")
    else:
        score = float(block_scores.float().max().item())

    if block_scores.numel() == 0:
        best_offset = 0
    else:
        best_offset = int(block_offsets.reshape(-1)[int(block_scores.reshape(-1).argmax().item())].item())
        best_offset = max(0, best_offset)
    return {"score": score, "best_token_offset": best_offset}


def score_int4_key_chunk(
    query: torch.Tensor,
    k_data: torch.Tensor,
    k_scales: torch.Tensor,
    k_zps: torch.Tensor,
    *,
    group_size: int,
    score_reduce: str = "max",
    top_r: int = 8,
    block_k: int = 32,
) -> Dict[str, float | int]:
    """Return a chunk score and best token offset without materializing FP K."""
    if not is_available():
        raise RuntimeError("Triton/CUDA is not available")
    if score_reduce not in SUPPORTED_REDUCERS:
        raise NotImplementedError(f"Triton scoring does not support reducer: {score_reduce}")
    if query.dim() == 3:
        query = query.unsqueeze(2)
    if query.dim() != 4 or k_data.dim() != 4:
        raise RuntimeError(f"Expected 4D query/key tensors, got {query.shape} and {k_data.shape}")

    batch_q, q_heads, q_len, head_dim = query.shape
    batch_k, kv_heads, kv_len, key_dim = k_data.shape
    if batch_q != batch_k or head_dim != key_dim:
        raise RuntimeError(f"Q/K shape mismatch: query={tuple(query.shape)} key={tuple(k_data.shape)}")
    if q_heads % kv_heads != 0:
        raise NotImplementedError(
            f"Triton scoring supports q_heads % kv_heads == 0, got {q_heads}/{kv_heads}"
        )

    q = query.contiguous()
    k = k_data.contiguous()
    scales = k_scales.contiguous()
    zps = k_zps.contiguous()
    if not (q.is_cuda and k.is_cuda and scales.is_cuda and zps.is_cuda):
        raise RuntimeError("Triton scoring inputs must be CUDA tensors")

    block_d = triton.next_power_of_2(head_dim)
    num_blocks = triton.cdiv(kv_len, block_k)
    block_scores = torch.empty(
        (batch_q, q_heads, q_len, num_blocks),
        device=q.device,
        dtype=torch.float32,
    )
    block_offsets = torch.empty(
        (batch_q, q_heads, q_len, num_blocks),
        device=q.device,
        dtype=torch.int32,
    )

    grid = (batch_q * q_heads * q_len, num_blocks)
    _int4_score_blocks_kernel[grid](
        q,
        k,
        scales,
        zps,
        block_scores,
        block_offsets,
        batch_q * q_heads * q_len,
        q_heads,
        kv_heads,
        q_len,
        kv_len,
        head_dim,
        num_blocks,
        int(group_size),
        block_k,
        block_d,
        num_warps=4,
    )
    return _reduce_block_scores(block_scores, block_offsets, score_reduce, top_r)


def score_int4_key_chunks_batch(
    query: torch.Tensor,
    k_data: torch.Tensor,
    k_scales: torch.Tensor,
    k_zps: torch.Tensor,
    *,
    group_size: int,
    score_reduce: str = "max",
    top_r: int = 8,
    block_k: int = 32,
) -> Dict[str, list[float] | list[int]]:
    """Score several same-shaped candidate chunks with one Triton launch grid."""
    if not is_available():
        raise RuntimeError("Triton/CUDA is not available")
    if score_reduce not in SUPPORTED_REDUCERS:
        raise NotImplementedError(f"Triton scoring does not support reducer: {score_reduce}")
    if query.dim() == 3:
        query = query.unsqueeze(2)
    if query.dim() != 4 or k_data.dim() != 5:
        raise RuntimeError(f"Expected query [B,H,Q,D] and key [C,B,H,K,D], got {query.shape} and {k_data.shape}")

    chunk_count, batch_k, kv_heads, kv_len, key_dim = k_data.shape
    batch_q, q_heads, q_len, head_dim = query.shape
    if batch_q != batch_k or head_dim != key_dim:
        raise RuntimeError(f"Q/K shape mismatch: query={tuple(query.shape)} key={tuple(k_data.shape)}")
    if q_heads % kv_heads != 0:
        raise NotImplementedError(
            f"Triton scoring supports q_heads % kv_heads == 0, got {q_heads}/{kv_heads}"
        )

    q = query.contiguous()
    k = k_data.contiguous()
    scales = k_scales.contiguous()
    zps = k_zps.contiguous()
    if not (q.is_cuda and k.is_cuda and scales.is_cuda and zps.is_cuda):
        raise RuntimeError("Triton scoring inputs must be CUDA tensors")
    if scales.dim() != 2 or zps.dim() != 2 or scales.shape != zps.shape:
        raise RuntimeError(
            f"Expected stacked scale/zp tensors [chunks, groups], got {scales.shape} and {zps.shape}"
        )

    block_d = triton.next_power_of_2(head_dim)
    num_blocks = triton.cdiv(kv_len, block_k)
    block_scores = torch.empty(
        (chunk_count, batch_q, q_heads, q_len, num_blocks),
        device=q.device,
        dtype=torch.float32,
    )
    block_offsets = torch.empty(
        (chunk_count, batch_q, q_heads, q_len, num_blocks),
        device=q.device,
        dtype=torch.int32,
    )

    grid = (chunk_count * batch_q * q_heads * q_len, num_blocks)
    _int4_score_chunk_batch_kernel[grid](
        q,
        k,
        scales,
        zps,
        block_scores,
        block_offsets,
        batch_q,
        q_heads,
        kv_heads,
        q_len,
        kv_len,
        head_dim,
        num_blocks,
        int(group_size),
        int(scales.shape[-1]),
        block_k,
        block_d,
        num_warps=4,
    )

    scores: list[float] = []
    offsets: list[int] = []
    for idx in range(chunk_count):
        reduced = _reduce_block_scores(
            block_scores[idx],
            block_offsets[idx],
            score_reduce,
            top_r,
        )
        scores.append(float(reduced["score"]))
        offsets.append(int(reduced["best_token_offset"]))
    return {"scores": scores, "best_token_offsets": offsets}
