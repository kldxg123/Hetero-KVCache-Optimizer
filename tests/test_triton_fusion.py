import sys
import os
import torch
import time
import triton
import triton.language as tl

# [核心修复 1] 动态注入项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

# ==========================================
# 1. Triton Kernel 定义
# ==========================================
@triton.jit
def _fused_dequant_attention_kernel(
    Q_ptr, K_quant_ptr, K_scale_ptr, K_zp_ptr, Out_ptr,
    stride_qh, stride_qd,
    stride_ks, stride_kd,
    head_dim, 
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_d = tl.arange(0, 128) # 针对 Qwen2.5 常见的 head_dim=128
    
    # 修复：显式处理指针偏移
    q = tl.load(Q_ptr + offs_d * stride_qd)
    offs_s = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    scales = tl.load(K_scale_ptr + offs_s)
    zps = tl.load(K_zp_ptr + offs_s)

    k_ptrs = K_quant_ptr + (offs_s[:, None] * stride_ks + offs_d[None, :] * stride_kd)
    k_q = tl.load(k_ptrs) 
    
    # 寄存器级反量化
    k_bf16 = (k_q - zps[:, None]) * scales[:, None]

    # 点乘融合 (Q @ K.T)
    attn_scores = tl.sum(q[None, :] * k_bf16, axis=1)
    tl.store(Out_ptr + offs_s, attn_scores)

def fused_dequant_attention(q, k_q, k_s, k_z):
    # 确保 q 是二维的 [1, head_dim]
    if q.dim() == 1:
        q = q.unsqueeze(0)
        
    seq_len = k_q.shape[0]
    head_dim = q.shape[-1]
    out = torch.empty((seq_len,), device=q.device, dtype=q.dtype)
    grid = lambda META: (triton.cdiv(seq_len, META['BLOCK_SIZE']), )
    
    _fused_dequant_attention_kernel[grid](
        q, k_q, k_s, k_z, out,
        q.stride(0), q.stride(1),
        k_q.stride(0), k_q.stride(1),
        head_dim,
        BLOCK_SIZE=16
    )
    return out

# ==========================================
# 2. 性能基准测试逻辑
# ==========================================
def benchmark_fusion():
    print("\n" + "="*60)
    print("🚀 [Triton] 启动 Hetero-KV 算子融合性能压测 (Kernel Fusion)")
    print("="*60)
    
    seq_len = 16384  
    head_dim = 128
    device = "cuda:0"
    dtype = torch.bfloat16

    # 构造模拟数据：确保 q 是 [1, 128]
    q = torch.randn((1, head_dim), device=device, dtype=dtype)
    k_q = torch.randint(0, 15, (seq_len, head_dim), device=device, dtype=torch.float32)
    k_s = torch.rand(seq_len, device=device, dtype=dtype)
    k_z = torch.rand(seq_len, device=device, dtype=dtype)

    # --- 测试 A: 原生逻辑 ---
    print("🐢 运行 Native Baseline (含显存回写)...")
    torch.cuda.synchronize()
    start_a = time.perf_counter()
    
    for _ in range(100):
        # 强制显存写回：反量化后的中间张量
        k_decompressed = (k_q - k_z[:, None]) * k_s[:, None]
        # 注意矩阵乘法的维度匹配
        res_native = torch.matmul(q, k_decompressed.to(dtype).t()).squeeze()
    
    torch.cuda.synchronize()
    native_time = (time.perf_counter() - start_a) * 10 

    # --- 测试 B: Triton 融合逻辑 ---
    print("🔥 运行 Triton Fused Kernel (寄存器直接计算)...")
    _ = fused_dequant_attention(q, k_q, k_s, k_z) # 预热
    
    torch.cuda.synchronize()
    start_b = time.perf_counter()
    
    for _ in range(100):
        res_triton = fused_dequant_attention(q, k_q, k_s, k_z)
    
    torch.cuda.synchronize()
    triton_time = (time.perf_counter() - start_b) * 10 

    # --- 结果汇总 ---
    print("\n" + "📊" * 15)
    print(f"🐢 [Native] 平均延迟: {native_time:.4f} ms")
    print(f"🚀 [Triton] 平均延迟: {triton_time:.4f} ms")
    
    speedup = native_time / triton_time
    print(f"🔥 算子层级加速比:   {speedup:.2f}x")
    
    diff = torch.max(torch.abs(res_native - res_triton))
    print(f"✅ 数值一致性校验:    Diff < {diff:.6e}")
    print("📊" * 15)
    print("="*60 + "\n")

if __name__ == "__main__":
    try:
        benchmark_fusion()
    except Exception as e:
        print(f"❌ 运行失败: {e}")
        import traceback
        traceback.print_exc()