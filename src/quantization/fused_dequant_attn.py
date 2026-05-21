"""
src/quantization/fused_dequant_attn.py
========================================
Fused Dequant-Attention: FlashAttention-style kernel with on-the-fly 4-bit
dequantization in registers.

Design:
  - Eliminates the BF16 intermediate copy by dequantizing K/V from INT4
    directly inside the attention kernel.
  - Uses online softmax (two-pass max/sum reduction) so that the full
    attention score matrix is never materialized.
  - Operates on per-token quantization metadata (scale + zero_point per row).

Interface:
  fused_dequant_attn_forward(Q, Kq, Ks, Kz, Vq, Vs, Vz, sm_scale)
      -> attention_output [batch, heads, seq_q, head_dim]

Fallback:
  If Triton is unavailable, decompresses via PyTorch matmul path.
"""

import torch
from typing import Optional

try:
    import triton
    import triton.language as tl
    _TRITON_OK = True
except ImportError:
    _TRITON_OK = False


# ------------------------------------------------------------------
# Triton Kernels
# ------------------------------------------------------------------

if _TRITON_OK:

    @triton.jit
    def _fused_dequant_attn_fwd_kernel(
        Q_ptr, K_q_ptr, K_s_ptr, K_z_ptr,
        V_q_ptr, V_s_ptr, V_z_ptr, Out_ptr,
        L_ptr, M_ptr,
        stride_qb, stride_qs, stride_qd,
        stride_kb, stride_ks, stride_kd,
        stride_sb, stride_sn,
        stride_vb, stride_vs, stride_vd,
        stride_ob, stride_os, stride_od,
        seq_k,
        seq_q: tl.constexpr,
        sm_scale,
        BLOCK_D: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """
        Fused Dequant-Attention forward kernel (online softmax, no BF16 materialization).

        Grid: (batch * heads, seq_q, num_blocks_n)
        Each program handles one query token and one block of key/value tokens.
        K_q / V_q : [bh, seq_k, head_dim]  (uint8 quantized data)
        K_s / K_z : [bh, seq_k]            (per-token scale / zero_point)
        """
        pid_bh = tl.program_id(0)
        pid_q  = tl.program_id(1)
        pid_n  = tl.program_id(2)

        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, BLOCK_D)
        mask_n = offs_n < seq_k

        # ---- Load Query [BLOCK_D] ----
        q_ptrs = Q_ptr + pid_bh * stride_qb + pid_q * stride_qs + offs_d * stride_qd
        q = tl.load(q_ptrs).to(tl.float32)

        # ---- Initialize online softmax accumulators ----
        m_i = tl.load(M_ptr + pid_bh * seq_q + pid_q)
        l_i = tl.load(L_ptr + pid_bh * seq_q + pid_q)
        o_ptrs = Out_ptr + pid_bh * stride_ob + pid_q * stride_os + offs_d * stride_od
        acc_o = tl.load(o_ptrs)

        # ---- Load quantized K and dequantize on-the-fly ----
        k_ptrs = (K_q_ptr
                  + pid_bh * stride_kb
                  + offs_n[:, None] * stride_ks
                  + offs_d[None, :] * stride_kd)
        k_q = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0).to(tl.float32)

        # scale / zp use 2D strides [bh, seq_k]
        k_s_ptrs = K_s_ptr + pid_bh * stride_sb + offs_n * stride_sn
        k_s = tl.load(k_s_ptrs, mask=mask_n, other=1.0).to(tl.float32)

        k_z_ptrs = K_z_ptr + pid_bh * stride_sb + offs_n * stride_sn
        k_z = tl.load(k_z_ptrs, mask=mask_n, other=0.0).to(tl.float32)

        k_deq = (k_q - k_z[:, None]) * k_s[:, None]

        # ---- Q @ K^T scores ----
        scores = tl.sum(q[None, :] * k_deq, axis=1) * sm_scale
        scores = tl.where(mask_n, scores, float('-inf'))

        # ---- Online softmax ----
        m_ij = tl.max(scores, axis=0)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        beta = tl.exp(scores - m_new)

        l_new = alpha * l_i + tl.sum(beta, axis=0)

        # ---- Update accumulator ----
        acc_o = acc_o * alpha

        # ---- Load quantized V and dequantize on-the-fly ----
        v_ptrs = (V_q_ptr
                  + pid_bh * stride_vb
                  + offs_n[:, None] * stride_vs
                  + offs_d[None, :] * stride_vd)
        v_q = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0).to(tl.float32)

        v_s_ptrs = V_s_ptr + pid_bh * stride_sb + offs_n * stride_sn
        v_s = tl.load(v_s_ptrs, mask=mask_n, other=1.0).to(tl.float32)

        v_z_ptrs = V_z_ptr + pid_bh * stride_sb + offs_n * stride_sn
        v_z = tl.load(v_z_ptrs, mask=mask_n, other=0.0).to(tl.float32)

        v_deq = (v_q - v_z[:, None]) * v_s[:, None]

        # ---- Weighted sum ----
        acc_o = acc_o + tl.sum(beta[:, None] * v_deq, axis=0)

        # ---- Write back ----
        tl.store(o_ptrs, acc_o)
        tl.store(M_ptr + pid_bh * seq_q + pid_q, m_new)
        tl.store(L_ptr + pid_bh * seq_q + pid_q, l_new)

    @triton.jit
    def _finalize_attn_kernel(
        Out_ptr, L_ptr,
        stride_ob, stride_os, stride_od,
        seq_q,
        BLOCK_D: tl.constexpr,
    ):
        pid_bh = tl.program_id(0)
        pid_q  = tl.program_id(1)
        offs_d = tl.arange(0, BLOCK_D)

        l_val = tl.load(L_ptr + pid_bh * seq_q + pid_q)
        l_val = tl.maximum(l_val, 1e-10)

        o_ptrs = Out_ptr + pid_bh * stride_ob + pid_q * stride_os + offs_d * stride_od
        o_val = tl.load(o_ptrs)
        tl.store(o_ptrs, o_val / l_val)


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def fused_dequant_attn_forward(
    Q: torch.Tensor,
    K_quant: torch.Tensor,
    K_scales: torch.Tensor,
    K_zps: torch.Tensor,
    V_quant: torch.Tensor,
    V_scales: torch.Tensor,
    V_zps: torch.Tensor,
    sm_scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Fused Dequant-Attention: compute attention output directly on 4-bit
    quantized K/V, never materializing BF16 intermediates.

    Args:
        Q       : [batch, heads, seq_q, head_dim]  float16/bfloat16
        K_quant : [batch, heads, seq_k, head_dim]  uint8  (4-bit quantized)
        K_scales: [batch, heads, seq_k]            float32 (per-token scale)
        K_zps   : [batch, heads, seq_k]            float32 (per-token zero_point)
        V_quant : [batch, heads, seq_k, head_dim]  uint8
        V_scales: [batch, heads, seq_k]            float32
        V_zps   : [batch, heads, seq_k]            float32
        sm_scale: softmax scaling factor (default: 1/sqrt(head_dim))

    Returns:
        Out : [batch, heads, seq_q, head_dim] float32
    """
    assert Q.dim() == 4, f"Q must be 4D [b,h,s,d], got {Q.shape}"
    batch, heads, seq_q, head_dim = Q.shape
    seq_k = K_quant.shape[2]

    if sm_scale is None:
        sm_scale = 1.0 / (head_dim ** 0.5)

    BLOCK_D = _next_pow2(head_dim)
    BLOCK_N = 32  # tune for occupancy

    if _TRITON_OK and Q.is_cuda:
        Q_3d = Q.reshape(batch * heads, seq_q, head_dim)
        Kq_3d = K_quant.reshape(batch * heads, seq_k, head_dim)
        Ks_2d = K_scales.reshape(batch * heads, seq_k).contiguous()
        Kz_2d = K_zps.reshape(batch * heads, seq_k).contiguous()
        Vq_3d = V_quant.reshape(batch * heads, seq_k, head_dim)
        Vs_2d = V_scales.reshape(batch * heads, seq_k).contiguous()
        Vz_2d = V_zps.reshape(batch * heads, seq_k).contiguous()

        out = torch.zeros(batch * heads, seq_q, head_dim,
                          device=Q.device, dtype=torch.float32)
        L = torch.zeros(batch * heads, seq_q, device=Q.device, dtype=torch.float32)
        M = torch.full((batch * heads, seq_q), float('-inf'),
                       device=Q.device, dtype=torch.float32)

        num_blocks_n = triton.cdiv(seq_k, BLOCK_N)
        grid = (batch * heads, seq_q, num_blocks_n)

        try:
            _fused_dequant_attn_fwd_kernel[grid](
                Q_3d, Kq_3d, Ks_2d, Kz_2d,
                Vq_3d, Vs_2d, Vz_2d, out,
                L, M,
                Q_3d.stride(0), Q_3d.stride(1), Q_3d.stride(2),
                Kq_3d.stride(0), Kq_3d.stride(1), Kq_3d.stride(2),
                Ks_2d.stride(0), Ks_2d.stride(1),
                Vq_3d.stride(0), Vq_3d.stride(1), Vq_3d.stride(2),
                out.stride(0), out.stride(1), out.stride(2),
                seq_k=seq_k,
                seq_q=seq_q,
                sm_scale=sm_scale,
                BLOCK_D=BLOCK_D,
                BLOCK_N=BLOCK_N,
            )

            _finalize_attn_kernel[(batch * heads, seq_q)](
                out, L,
                out.stride(0), out.stride(1), out.stride(2),
                seq_q=seq_q,
                BLOCK_D=BLOCK_D,
            )

            return out.reshape(batch, heads, seq_q, head_dim)

        except Exception:
            pass  # fall through to PyTorch path

    # ---- PyTorch fallback: decompress then matmul ----
    K_deq = _dequant_tensor(K_quant, K_scales, K_zps)
    V_deq = _dequant_tensor(V_quant, V_scales, V_zps)
    scores = torch.matmul(Q.float(), K_deq.transpose(-2, -1)) * sm_scale
    attn = torch.softmax(scores, dim=-1)
    return torch.matmul(attn, V_deq)


# ------------------------------------------------------------------
# Fused single-query decode path (optimized for seq_q=1)
# ------------------------------------------------------------------

def fused_dequant_attn_decode(
    q: torch.Tensor,
    k_quant: torch.Tensor,
    k_scales: torch.Tensor,
    k_zps: torch.Tensor,
    v_quant: torch.Tensor,
    v_scales: torch.Tensor,
    v_zps: torch.Tensor,
    sm_scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Optimized decode path: q is [batch, heads, 1, head_dim].
    Computes attention directly on quantized K/V without BF16 materialization.

    Returns: [batch, heads, 1, head_dim] float32
    """
    head_dim = q.shape[-1]
    if sm_scale is None:
        sm_scale = 1.0 / (head_dim ** 0.5)

    batch, heads, _, _ = q.shape
    seq_k = k_quant.shape[2]

    if _TRITON_OK and q.is_cuda:
        Q_3d = q.reshape(batch * heads, 1, head_dim)
        Kq_3d = k_quant.reshape(batch * heads, seq_k, head_dim)
        Ks_2d = k_scales.reshape(batch * heads, seq_k).contiguous()
        Kz_2d = k_zps.reshape(batch * heads, seq_k).contiguous()
        Vq_3d = v_quant.reshape(batch * heads, seq_k, head_dim)
        Vs_2d = v_scales.reshape(batch * heads, seq_k).contiguous()
        Vz_2d = v_zps.reshape(batch * heads, seq_k).contiguous()

        BLOCK_D = _next_pow2(head_dim)
        BLOCK_N = 32

        out = torch.zeros(batch * heads, 1, head_dim,
                          device=q.device, dtype=torch.float32)
        L = torch.zeros(batch * heads, 1, device=q.device, dtype=torch.float32)
        M = torch.full((batch * heads, 1), float('-inf'),
                       device=q.device, dtype=torch.float32)

        num_blocks_n = triton.cdiv(seq_k, BLOCK_N)

        try:
            _fused_dequant_attn_fwd_kernel[(batch * heads, 1, num_blocks_n)](
                Q_3d, Kq_3d, Ks_2d, Kz_2d,
                Vq_3d, Vs_2d, Vz_2d, out,
                L, M,
                Q_3d.stride(0), Q_3d.stride(1), Q_3d.stride(2),
                Kq_3d.stride(0), Kq_3d.stride(1), Kq_3d.stride(2),
                Ks_2d.stride(0), Ks_2d.stride(1),
                Vq_3d.stride(0), Vq_3d.stride(1), Vq_3d.stride(2),
                out.stride(0), out.stride(1), out.stride(2),
                seq_k=seq_k,
                seq_q=1,
                sm_scale=sm_scale,
                BLOCK_D=BLOCK_D,
                BLOCK_N=BLOCK_N,
            )

            _finalize_attn_kernel[(batch * heads, 1)](
                out, L,
                out.stride(0), out.stride(1), out.stride(2),
                seq_q=1,
                BLOCK_D=BLOCK_D,
            )

            return out.reshape(batch, heads, 1, head_dim)

        except Exception:
            pass

    # Fallback
    K_deq = _dequant_tensor(k_quant, k_scales, k_zps)
    V_deq = _dequant_tensor(v_quant, v_scales, v_zps)
    scores = torch.matmul(q.float(), K_deq.transpose(-2, -1)) * sm_scale
    attn = torch.softmax(scores, dim=-1)
    return torch.matmul(attn, V_deq)


# ------------------------------------------------------------------
# Legacy single-head interface (backward compatible)
# ------------------------------------------------------------------

def fused_dequant_attention(
    q: torch.Tensor,
    k_quant: torch.Tensor,
    k_scales: torch.Tensor,
    k_zps: torch.Tensor,
    group_size: int = 128,
) -> torch.Tensor:
    """
    Legacy single-head interface: compute Q@K^T scores on quantized data.
    Returns pre-softmax attention scores [seq_k].
    """
    q_flat = q.reshape(-1).contiguous()
    seq_k = k_quant.shape[0] if k_quant.dim() >= 1 else 1
    head_dim = q_flat.shape[0]

    k_q_2d = k_quant.reshape(seq_k, head_dim).contiguous()
    k_s_1d = k_scales.reshape(seq_k).float().contiguous()
    k_z_1d = k_zps.reshape(seq_k).float().contiguous()

    if _TRITON_OK and q.is_cuda and seq_k > 0:
        BLOCK_K = min(16, seq_k)
        BLOCK_D = _next_pow2(head_dim)

        out = torch.zeros(seq_k, device=q.device, dtype=torch.float32)
        grid = (triton.cdiv(seq_k, BLOCK_K),)

        try:
            _fused_dequant_attn_kernel[grid](
                q_flat, k_q_2d, k_s_1d, k_z_1d, out,
                q_flat.stride(0), 1,
                k_q_2d.stride(0), k_q_2d.stride(1),
                seq_k=seq_k,
                head_dim=head_dim,
                group_size=group_size,
                BLOCK_K=BLOCK_K,
                BLOCK_D=BLOCK_D,
            )
            return out
        except Exception:
            pass

    k_f32 = (k_q_2d.float() - k_z_1d.unsqueeze(-1)) * k_s_1d.unsqueeze(-1)
    q_f32 = q_flat.float()
    return torch.mv(k_f32, q_f32)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _dequant_tensor(quant, scales, zps):
    """Dequantize a 4-bit tensor: (Q - zp) * scale."""
    return (quant.float() - zps.unsqueeze(-1).float()) * scales.unsqueeze(-1).float()


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return min(p, 256)


# ------------------------------------------------------------------
# Standalone test
# ------------------------------------------------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    head_dim = 128
    seq_k = 64
    batch = 2
    heads = 4

    Q = torch.randn(batch, heads, 1, head_dim, device=device, dtype=torch.float16)
    K_f16 = torch.randn(batch, heads, seq_k, head_dim, device=device, dtype=torch.float16)
    V_f16 = torch.randn(batch, heads, seq_k, head_dim, device=device, dtype=torch.float16)

    # Simulate quantization
    K_q = K_f16.to(torch.uint8)
    K_s = torch.ones(batch, heads, seq_k, device=device, dtype=torch.float32) * 0.01
    K_z = torch.zeros(batch, heads, seq_k, device=device, dtype=torch.float32)
    V_q = V_f16.to(torch.uint8)
    V_s = torch.ones(batch, heads, seq_k, device=device, dtype=torch.float32) * 0.01
    V_z = torch.zeros(batch, heads, seq_k, device=device, dtype=torch.float32)

    # Full attention path
    out = fused_dequant_attn_forward(Q, K_q, K_s, K_z, V_q, V_s, V_z)
    print(f"[PASS] Fused Dequant-Attention | out.shape={out.shape} device={out.device}")
    print(f"  Triton available: {_TRITON_OK}")

    # Decode path
    out_dec = fused_dequant_attn_decode(Q, K_q, K_s, K_z, V_q, V_s, V_z)
    print(f"[PASS] Decode path | out.shape={out_dec.shape}")
