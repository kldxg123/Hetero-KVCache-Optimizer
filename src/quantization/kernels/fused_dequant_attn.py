"""
src/quantization/kernels/fused_dequant_attn.py
===============================================
Fused Dequantize-Attention Triton kernels.

Eliminates the BF16 intermediate spike by performing 4-bit -> BF16 dequantization
in GPU registers and directly computing the attention dot-product, followed by
an online Softmax (two-pass: max + sum) so the full attention matrix is never
materialized.

Exported API:
  fused_dequant_attention   — single-head utility (legacy)
  fused_dequant_attn_decode — decode path: [B, H, 1, D] query
  fused_dequant_attn_forward — prefill path: [B, H, S, D] query
"""

import torch
import triton
import triton.language as tl
import math
from typing import Optional


# =====================================================================
# Kernel 1: Decode — single query token against quantized KV cache
# =====================================================================

@triton.jit
def _fused_dequant_attn_decode_kernel(
    Q_ptr, K_quant_ptr, K_scale_ptr, K_zp_ptr,
    V_quant_ptr, V_scale_ptr, V_zp_ptr,
    Out_ptr,
    stride_qb, stride_qh, stride_qd,
    stride_kb, stride_ks, stride_kd,
    stride_vb, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_od,
    kv_seq_len,
    head_dim: tl.constexpr,
    BLOCK_KV: tl.constexpr,
):
    """
    One program instance handles one (batch, head) pair.
    Iterates over kv_seq_len in blocks of BLOCK_KV, performing:
      1. Load 4-bit quantized K, dequantize in registers, compute Q·K^T
      2. Online softmax: track running max and running sum
      3. Load 4-bit quantized V, dequantize in registers, accumulate weighted V
    """
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    # ---- Load query [head_dim] ----
    offs_d = tl.arange(0, head_dim)
    q_base = Q_ptr + pid_b * stride_qb + pid_h * stride_qh
    q = tl.load(q_base + offs_d * stride_qd)  # [head_dim]

    # ---- Online softmax accumulators ----
    running_max = float('-inf')
    running_sum = 0.0
    acc_out = tl.zeros([head_dim], dtype=tl.float32)

    num_blocks = tl.cdiv(kv_seq_len, BLOCK_KV)

    for blk in range(num_blocks):
        kv_start = blk * BLOCK_KV
        offs_kv = kv_start + tl.arange(0, BLOCK_KV)
        mask_kv = offs_kv < kv_seq_len

        # ---- Load K metadata and quantized data ----
        k_s_base = K_scale_ptr + pid_b * stride_kb + pid_h * stride_ks
        k_z_base = K_zp_ptr + pid_b * stride_kb + pid_h * stride_ks
        scales_k = tl.load(k_s_base + offs_kv, mask=mask_kv, other=1.0)
        zps_k = tl.load(k_z_base + offs_kv, mask=mask_kv, other=0.0)

        # K quant: [BLOCK_KV, head_dim]
        k_ptrs = (K_quant_ptr
                  + pid_b * stride_kb + pid_h * stride_ks
                  + offs_kv[:, None] * 1
                  + offs_d[None, :] * 1)
        k_mask = mask_kv[:, None] & (offs_d[None, :] < head_dim)
        k_q = tl.load(k_ptrs, mask=k_mask, other=0.0)

        # Dequantize: (k_q - zp) * scale
        k_deq = (k_q - zps_k[:, None]) * scales_k[:, None]

        # Q · K^T -> [BLOCK_KV]
        attn_block = tl.sum(q[None, :] * k_deq, axis=1)  # [BLOCK_KV]

        # ---- Online softmax: update running max and sum ----
        block_max = tl.max(attn_block, axis=0)
        new_max = tl.maximum(running_max, block_max)

        # Rescale previous accumulator
        if running_max != float('-inf'):
            rescale = tl.exp(running_max - new_max)
            running_sum *= rescale
            acc_out = acc_out * rescale

        # Exp of current block scores
        exp_scores = tl.exp(attn_block - new_max)  # [BLOCK_KV]
        exp_scores = tl.where(mask_kv, exp_scores, 0.0)
        block_sum = tl.sum(exp_scores, axis=0)

        # ---- Load V metadata and quantized data ----
        v_s_base = V_scale_ptr + pid_b * stride_vb + pid_h * stride_vs
        v_z_base = V_zp_ptr + pid_b * stride_vb + pid_h * stride_vs
        scales_v = tl.load(v_s_base + offs_kv, mask=mask_kv, other=1.0)
        zps_v = tl.load(v_z_base + offs_kv, mask=mask_kv, other=0.0)

        # V quant: [BLOCK_KV, head_dim]
        v_ptrs = (V_quant_ptr
                  + pid_b * stride_vb + pid_h * stride_vs
                  + offs_kv[:, None] * 1
                  + offs_d[None, :] * 1)
        v_mask = mask_kv[:, None] & (offs_d[None, :] < head_dim)
        v_q = tl.load(v_ptrs, mask=v_mask, other=0.0)

        # Dequantize V
        v_deq = (v_q - zps_v[:, None]) * scales_v[:, None]

        # Accumulate: softmax_weighted V
        acc_out += tl.sum(exp_scores[:, None] * v_deq, axis=0)

        running_sum += block_sum
        running_max = new_max

    # ---- Final normalization ----
    acc_out = acc_out / tl.maximum(running_sum, 1e-8)

    # ---- Write output ----
    out_base = Out_ptr + pid_b * stride_ob + pid_h * stride_oh
    tl.store(out_base + offs_d * stride_od, acc_out)


# =====================================================================
# Kernel 2: Prefill — multiple query tokens against quantized KV cache
# =====================================================================

@triton.jit
def _fused_dequant_attn_forward_kernel(
    Q_ptr, K_quant_ptr, K_scale_ptr, K_zp_ptr,
    V_quant_ptr, V_scale_ptr, V_zp_ptr,
    Out_ptr,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    kv_seq_len,
    q_seq_len,
    head_dim: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
):
    """
    Prefill path: each program handles a (batch, head, q_block) tile.
    Iterates over kv_seq_len in blocks, computing attention for BLOCK_Q query
    positions at a time with online softmax.
    """
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_q = tl.program_id(2)

    q_start = pid_q * BLOCK_Q
    offs_q = q_start + tl.arange(0, BLOCK_Q)
    mask_q = offs_q < q_seq_len

    offs_d = tl.arange(0, head_dim)

    # Load Q tile: [BLOCK_Q, head_dim]
    q_ptrs = (Q_ptr
              + pid_b * stride_qb + pid_h * stride_qh
              + offs_q[:, None] * stride_qs
              + offs_d[None, :] * stride_qd)
    q_mask = mask_q[:, None]
    q_tile = tl.load(q_ptrs, mask=q_mask, other=0.0)

    # Online softmax accumulators per query position
    running_max = tl.full([BLOCK_Q], float('-inf'), dtype=tl.float32)
    running_sum = tl.zeros([BLOCK_Q], dtype=tl.float32)
    acc_out = tl.zeros([BLOCK_Q, head_dim], dtype=tl.float32)

    num_kv_blocks = tl.cdiv(kv_seq_len, BLOCK_KV)

    for blk in range(num_kv_blocks):
        kv_start = blk * BLOCK_KV
        offs_kv = kv_start + tl.arange(0, BLOCK_KV)
        mask_kv = offs_kv < kv_seq_len

        # Load K quantized: [BLOCK_KV, head_dim]
        k_ptrs = (K_quant_ptr
                  + pid_b * stride_kb + pid_h * stride_kh
                  + offs_kv[:, None] * stride_ks
                  + offs_d[None, :] * stride_kd)
        k_load_mask = mask_kv[:, None]
        k_q = tl.load(k_ptrs, mask=k_load_mask, other=0.0)

        # Load K scales and zero-points
        k_s = tl.load(K_scale_ptr + pid_b * stride_kb + pid_h * stride_kh
                       + offs_kv * stride_ks,
                       mask=mask_kv, other=1.0)
        k_z = tl.load(K_zp_ptr + pid_b * stride_kb + pid_h * stride_kh
                       + offs_kv * stride_ks,
                       mask=mask_kv, other=0.0)

        k_deq = (k_q - k_z[:, None]) * k_s[:, None]

        # Attention: [BLOCK_Q, BLOCK_KV]
        attn = tl.dot(q_tile.to(k_deq.dtype), tl.trans(k_deq))

        # Online softmax step
        block_max = tl.max(attn, axis=1)  # [BLOCK_Q]
        new_max = tl.maximum(running_max, block_max)

        # Rescale
        old_exp = tl.exp(running_max - new_max)
        running_sum *= old_exp
        acc_out = acc_out * old_exp[:, None]

        exp_scores = tl.exp(attn - new_max[:, None])
        exp_scores = tl.where(mask_q[:, None] & mask_kv[None, :], exp_scores, 0.0)
        block_sum = tl.sum(exp_scores, axis=1)

        # Load V quantized: [BLOCK_KV, head_dim]
        v_ptrs = (V_quant_ptr
                  + pid_b * stride_vb + pid_h * stride_vh
                  + offs_kv[:, None] * stride_vs
                  + offs_d[None, :] * stride_vd)
        v_q = tl.load(v_ptrs, mask=mask_kv[:, None], other=0.0)

        v_s = tl.load(V_scale_ptr + pid_b * stride_vb + pid_h * stride_vh
                       + offs_kv * stride_vs,
                       mask=mask_kv, other=1.0)
        v_z = tl.load(V_zp_ptr + pid_b * stride_vb + pid_h * stride_vh
                       + offs_kv * stride_vs,
                       mask=mask_kv, other=0.0)
        v_deq = (v_q - v_z[:, None]) * v_s[:, None]

        # Accumulate weighted V
        acc_out += tl.dot(exp_scores.to(v_deq.dtype), v_deq)

        running_sum += block_sum
        running_max = new_max

    # Normalize
    acc_out = acc_out / tl.maximum(running_sum[:, None], 1e-8)

    # Store output: [BLOCK_Q, head_dim]
    out_ptrs = (Out_ptr
                + pid_b * stride_ob + pid_h * stride_oh
                + offs_q[:, None] * stride_os
                + offs_d[None, :] * stride_od)
    tl.store(out_ptrs, acc_out, mask=mask_q[:, None])


# =====================================================================
# Kernel 0 (legacy): single-head dot product
# =====================================================================

@triton.jit
def _fused_dequant_attention_kernel(
        Q_ptr, K_quant_ptr, K_scale_ptr, K_zp_ptr, Out_ptr,
        stride_qh, stride_qd,
        stride_ks, stride_kd,
        head_dim,
        BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_d = tl.arange(0, 128)
    q = tl.load(Q_ptr + offs_d * stride_qd)
    offs_s = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    scales = tl.load(K_scale_ptr + offs_s)
    zps = tl.load(K_zp_ptr + offs_s)
    k_ptrs = K_quant_ptr + (offs_s[:, None] * stride_ks + offs_d[None, :] * stride_kd)
    k_q = tl.load(k_ptrs)
    k_bf16 = (k_q - zps[:, None]) * scales[:, None]
    attn_scores = tl.sum(q[None, :] * k_bf16, axis=1)
    tl.store(Out_ptr + offs_s, attn_scores)


# =====================================================================
# Public API wrappers
# =====================================================================

def fused_dequant_attention(q, k_q, k_s, k_z):
    """Legacy single-head fused dequant attention."""
    if q.dim() == 1:
        q = q.unsqueeze(0)
    seq_len = k_q.shape[0]
    head_dim = q.shape[-1]
    out = torch.empty((seq_len,), device=q.device, dtype=q.dtype)
    grid = lambda META: (triton.cdiv(seq_len, META['BLOCK_SIZE']),)
    _fused_dequant_attention_kernel[grid](
        q, k_q, k_s, k_z, out,
        q.stride(0), q.stride(1),
        k_q.stride(0), k_q.stride(1),
        head_dim,
        BLOCK_SIZE=16,
    )
    return out


def fused_dequant_attn_decode(
    q: torch.Tensor,
    k_data: torch.Tensor,
    k_scales: torch.Tensor,
    k_zps: torch.Tensor,
    v_data: torch.Tensor,
    v_scales: torch.Tensor,
    v_zps: torch.Tensor,
    sm_scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Decode-path fused attention: single query token against quantized KV cache.

    Args:
        q: [batch, num_heads, 1, head_dim] query tensor (BF16/FP16)
        k_data, k_scales, k_zps: quantized Key [batch, num_heads, kv_seq_len, head_dim]
        v_data, v_scales, v_zps: quantized Value [batch, num_heads, kv_seq_len, head_dim]
        sm_scale: optional softmax scale (default 1/sqrt(head_dim))

    Returns:
        attn_output: [batch, num_heads, 1, head_dim]
    """
    assert q.dim() == 4 and q.shape[2] == 1, f"Decode expects [B,H,1,D], got {q.shape}"
    batch, num_heads, _, head_dim = q.shape

    # Ensure KV tensors are [batch, num_heads, kv_seq_len, head_dim]
    if k_data.dim() == 2:
        k_data = k_data.unsqueeze(0).unsqueeze(0)
        k_scales = k_scales.unsqueeze(0).unsqueeze(0)
        k_zps = k_zps.unsqueeze(0).unsqueeze(0)
        v_data = v_data.unsqueeze(0).unsqueeze(0)
        v_scales = v_scales.unsqueeze(0).unsqueeze(0)
        v_zps = v_zps.unsqueeze(0).unsqueeze(0)
    elif k_data.dim() == 3:
        k_data = k_data.unsqueeze(0)
        k_scales = k_scales.unsqueeze(0)
        k_zps = k_zps.unsqueeze(0)
        v_data = v_data.unsqueeze(0)
        v_scales = v_scales.unsqueeze(0)
        v_zps = v_zps.unsqueeze(0)

    kv_seq_len = k_data.shape[-2]

    # Squeeze the query dim=2 so q is [batch, num_heads, head_dim]
    q_3d = q.squeeze(2)

    # Output: [batch, num_heads, head_dim]
    out = torch.empty((batch, num_heads, head_dim), device=q.device, dtype=torch.float32)

    BLOCK_KV = 64

    grid = (batch, num_heads)
    _fused_dequant_attn_decode_kernel[grid](
        q_3d,
        k_data, k_scales, k_zps,
        v_data, v_scales, v_zps,
        out,
        q_3d.stride(0), q_3d.stride(1), q_3d.stride(2),
        k_data.stride(0), k_data.stride(-2), k_data.stride(-1),
        v_data.stride(0), v_data.stride(-2), v_data.stride(-1),
        out.stride(0), out.stride(1), out.stride(2),
        kv_seq_len,
        head_dim=head_dim,
        BLOCK_KV=BLOCK_KV,
    )

    # Unsqueeze back to [batch, num_heads, 1, head_dim]
    return out.unsqueeze(2).to(q.dtype)


def fused_dequant_attn_forward(
    q: torch.Tensor,
    k_data: torch.Tensor,
    k_scales: torch.Tensor,
    k_zps: torch.Tensor,
    v_data: torch.Tensor,
    v_scales: torch.Tensor,
    v_zps: torch.Tensor,
    sm_scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Prefill-path fused attention: multiple query tokens against quantized KV cache.

    Args:
        q: [batch, num_heads, q_seq_len, head_dim] query tensor
        k_data, k_scales, k_zps: quantized Key [batch, num_heads, kv_seq_len, head_dim]
        v_data, v_scales, v_zps: quantized Value [batch, num_heads, kv_seq_len, head_dim]
        sm_scale: optional softmax scale

    Returns:
        attn_output: [batch, num_heads, q_seq_len, head_dim]
    """
    assert q.dim() == 4, f"Forward expects [B,H,S,D], got {q.shape}"
    batch, num_heads, q_seq_len, head_dim = q.shape

    if k_data.dim() == 2:
        k_data = k_data.unsqueeze(0).unsqueeze(0)
        k_scales = k_scales.unsqueeze(0).unsqueeze(0)
        k_zps = k_zps.unsqueeze(0).unsqueeze(0)
        v_data = v_data.unsqueeze(0).unsqueeze(0)
        v_scales = v_scales.unsqueeze(0).unsqueeze(0)
        v_zps = v_zps.unsqueeze(0).unsqueeze(0)
    elif k_data.dim() == 3:
        k_data = k_data.unsqueeze(0)
        k_scales = k_scales.unsqueeze(0)
        k_zps = k_zps.unsqueeze(0)
        v_data = v_data.unsqueeze(0)
        v_scales = v_scales.unsqueeze(0)
        v_zps = v_zps.unsqueeze(0)

    kv_seq_len = k_data.shape[-2]

    out = torch.empty((batch, num_heads, q_seq_len, head_dim),
                      device=q.device, dtype=torch.float32)

    BLOCK_Q = 16
    BLOCK_KV = 64

    grid = (batch, num_heads, triton.cdiv(q_seq_len, BLOCK_Q))
    _fused_dequant_attn_forward_kernel[grid](
        q,
        k_data, k_scales, k_zps,
        v_data, v_scales, v_zps,
        out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k_data.stride(0), k_data.stride(1), k_data.stride(-2), k_data.stride(-1),
        v_data.stride(0), v_data.stride(1), v_data.stride(-2), v_data.stride(-1),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        kv_seq_len, q_seq_len,
        head_dim=head_dim,
        BLOCK_Q=BLOCK_Q,
        BLOCK_KV=BLOCK_KV,
    )

    return out.to(q.dtype)
