import torch
import triton
import triton.language as tl


@triton.jit
def _fused_dequant_attention_kernel(
        Q_ptr, K_quant_ptr, K_scale_ptr, K_zp_ptr, Out_ptr,
        stride_qh, stride_qd,
        stride_ks, stride_kd,
        head_dim,
        BLOCK_SIZE: tl.constexpr,
):
    """
    Triton Kernel: perform 4-bit -> BF16 conversion in registers and direct dot product
    """
    pid = tl.program_id(0)

    # 1. Load Query (1, head_dim)
    offs_d = tl.arange(0, 128)
    q = tl.load(Q_ptr + offs_d * stride_qd)

    # 2. Compute K block start position
    offs_s = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # 3. Load quantization metadata
    scales = tl.load(K_scale_ptr + offs_s)
    zps = tl.load(K_zp_ptr + offs_s)

    # 4. Load 4-bit quantized Key and dequantize on-the-fly
    k_ptrs = K_quant_ptr + (offs_s[:, None] * stride_ks + offs_d[None, :] * stride_kd)
    k_q = tl.load(k_ptrs)

    # Register-level dequantization
    k_bf16 = (k_q - zps[:, None]) * scales[:, None]

    # 5. Fused computation: Dot Product (Q @ K.T)
    attn_scores = tl.sum(q[None, :] * k_bf16, axis=1)

    # 6. Write back final scores
    tl.store(Out_ptr + offs_s, attn_scores)


def fused_dequant_attention(q, k_q, k_s, k_z):
    """
    Wrapper function: interfaces with PyTorch tensors
    """
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
        BLOCK_SIZE=16
    )
    return out
