"""
Triton算子的真实意义 vs 当前实现的矛盾
========================================

问题：如果取回后还是要占HBM，那Triton融合算子还有什么意义？

答案：当前实现**根本没用Triton算子做self-healing**。

## 当前代码路径分析

### engine_wrapper.py 第137-141行 (self-healing)
```python
dram_k, dram_v, count = manager.decompress_dram_chunks(layer_idx)
# decompress_dram_chunks做了什么？
# → 解压所有DRAM chunks到BF16
# → torch.cat()拼接成大tensor
# → 这些tensors都在GPU内存中！
out_k = torch.cat([dram_k, out_k], dim=-2)
out_v = torch.cat([dram_v, out_v], dim=-2)
```

**HBM分配过程**：
1. `decompress()` 调用 compressor → 返回 BF16 tensor → GPU分配
2. `torch.cat()` 创建新tensor → GPU分配
3. 结果：**每个decode step有O(N)的HBM峰值**

### engine_wrapper.py 第243-290行 (fused_attn_on_swapped)
```python
def fused_attn_on_swapped(self, q, chunk_key, sm_scale=None):
    quant_data = self.swap_in_quantized(chunk_key)
    # swap_in_quantized返回什么？
    # → 4-bit量化数据（还在GPU内存中，但很小）
    if q.shape[2] == 1:
        return fused_dequant_attn_decode(q, quant_data["k_data"], ...)
```

**关键问题**：`fused_attn_on_swapped()` **从未被标准generate流程调用**！

## Triton算子的真正价值

### 理论上的Zero-Copy路径
```
DRAM (4-bit, CPU)
  ↓ PCIe传输（小）
GPU (4-bit, HBM)
  ↓ 直接传给Triton kernel
GPU寄存器
  ↓ 反量化 + attention计算
GPU (output, HBM)
```

**HBM分配**：只有输入的4-bit数据（很小），没有中间BF16 spike

### 当前实现的路径
```
DRAM (4-bit, CPU)
  ↓ PCIe传输
GPU (4-bit, HBM)
  ↓ decompress() → BF16 tensor (GPU分配！)
GPU (BF16, HBM) ← 762MB at 128K
  ↓ torch.cat() → 新tensor (GPU分配！)
GPU (BF16, HBM)
  ↓ attention计算
GPU (output, HBM)
```

**HBM分配**：4-bit + BF16 spike（762MB） + concat tensor（更多）

## 代码证据

### src/quantization/kernels/fused_dequant_attn.py 第23-153行
```python
@triton.jit
def _fused_dequant_attn_decode_kernel(...):
    """
    在寄存器中：
      1. 加载4-bit K → dequantize → Q·K^T
      2. 在线softmax
      3. 加载4-bit V → dequantize → 加权求和
    """
    # 从K_quant_ptr直接加载4-bit数据
    k_q = tl.load(k_ptrs, mask=k_mask, other=0.0)
    k_deq = (k_q - zps_k[:, None]) * scales_k[:, None]
    # 在寄存器中计算attention
    attn_block = tl.sum(q[None, :] * k_deq, axis=1)
```

这个kernel设计明确指出：**直接从4-bit数据反量化**，不需要BF16中间tensor。

但当前代码在调用这个kernel**之前**就已经解压成BF16了！

## 结论

### Triton算子的存在价值
**仅当直接传4-bit数据时才有价值**：
- 避免512MB BF16中间tensor（论文声称）
- Zero-copy路径：4-bit → 寄存器 → 计算
- 这就是为什么论文声称"eliminates 512MB transient allocation"

### 当前实现的问题
**Triton算子形同虚设**：
1. `decompress_dram_chunks()` 先解压成BF16 → 762MB HBM spike
2. `fused_attn_on_swapped()` 从未被调用
3. 标准路径使用的是FlashAttention，不是fused kernel

### 论文承诺 vs 实际实现
| 论文声称 | 实际实现 |
|---------|---------|
| "Eliminates 512MB transient" | 解压时仍分配762MB BF16 |
| "Zero-copy in registers" | 先解压到HBM再计算 |
| "Fused dequant-attention" | 标本从未被使用 |

## 修复方案

### 方案A：真正的Fused路径
```python
def decompress_dram_chunks_fused(self, layer_idx: int):
    """不解压，直接返回4-bit数据给fused kernel"""
    dram_keys = [...]
    quant_data = []
    for key in dram_keys:
        entry = self._dram.retrieve(key)
        # 不调用decompress()，直接返回4-bit
        quant_data.append({
            "k_data": entry["k_data"].to(self.device),
            "k_scales": entry["k_scales"].to(self.device),
            "k_zps": entry["k_zps"].to(self.device),
            ...
        })
    return quant_data  # 返回4-bit，不是BF16

# 在engine_wrapper.py中
quant_data = manager.decompress_dram_chunks_fused(layer_idx)
# 直接传给fused kernel，不经过BF16
output = fused_dequant_attn_decode(q, quant_data, ...)
```

**问题**：需要重写attention计算流程，因为FlashAttention不接受4-bit输入

### 方案B：承认实现限制
- 保留当前实现（BF16解压）
- 论文修正：删除"eliminates transient allocation"的声称
- 诚实说明："当前版本在恢复时有O(N) HBM spike，未来可用fused kernel优化"

## 最终答案

**Triton算子在当前实现中没有实际作用**。

它是一个"未兑现的承诺"：
- 代码存在，但未被使用
- 论文声称的性能优化（0MB瞬态）在实际中不存在
- 要让它真正work，需要大幅重构attention计算路径

这就是为什么您看到HBM峰值上升——因为当前实现**根本没有走zero-copy路径**。
"""

# 实际验证：检查当前是否使用了fused kernel
def verify_fused_kernel_usage():
    """
    检查当前代码是否真正使用了Triton fused kernel
    """
    import inspect
    from src.core.engine_wrapper import FusedHeteroCache

    # 检查update方法的调用链
    source = inspect.getsource(FusedHeteroCache.update)
    has_fused_call = "fused_dequant" in source or "fused_attn_on_swapped" in source

    # 检查self-healing路径
    has_decompress_call = "decompress_dram_chunks" in source

    print(f"Update method calls fused kernel: {has_fused_call}")
    print(f"Update method calls decompress_dram_chunks: {has_decompress_call}")

    # 真相：decompress_dram_chunks → BF16 → torch.cat → FlashAttention
    # fused_attn_on_swapped存在但未被调用

if __name__ == "__main__":
    verify_fused_kernel_usage()
