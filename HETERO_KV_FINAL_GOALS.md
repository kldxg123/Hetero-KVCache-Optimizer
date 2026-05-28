# Hetero-KVCache-Optimizer 最终目标与执行守则

## 项目定位

Hetero-KVCache-Optimizer 定位为 **Approximate Long-Context Cache**。

本项目不是 128K full KV cache 的无损复现，也不以 token-level logits 与原生 full attention 完全等价为目标。

项目要证明的是：在固定 GPU HBM 显存预算下，通过近似 KV 管理机制，让 7B 模型在 128K 长上下文下可生成、显存稳定、语义损失受控、延迟可解释。

## 核心目标

核心目标是在固定 GPU HBM 预算下，通过以下机制：

- Sink 全局锚点 token
- Tail 局部窗口 token
- Heavy-Hitter 累计高注意力 token
- 4-bit DRAM KV 存储
- Query-aware token-level dot-product retrieval

让 **Qwen2.5-7B-Instruct** 在 **128K 长上下文** 下完成真实 generate 测试，并展示 Hetero-KVCache-Optimizer 的项目价值。

## 展示口径

最终展示重点不是证明“近似缓存等价于原生 full KV cache”，而是证明：

- 128K 上下文下不 OOM。
- GPU HBM 中活跃 KV 长度保持近似 O(1) 稳定。
- 被驱逐 KV 进入 CPU/DRAM 侧压缩存储。
- Query-aware dot-product retrieval 能从 DRAM 找回相关信息。
- NIAH 多深度任务准确率达标。
- WikiText-2 真实 PPL 劣化受控。
- `generate()` 接口可正常运行。
- latency 增长有拆解、有解释。

## 4090-24G 生存证明口径

远程服务器是 A100，但最终希望展示的是：**128K 上下文 + 7B 模型在 4090 24G 显存预算下可以生存**。

因此验收组必须在 A100 上设置近似 4090 的显存限制：

- 默认目标上限：22 GiB，更保守地模拟 4090 24G。
- 可补充测试：24 GiB cap。
- HeteroKV 验收必须在显存 cap 下完成。
- baseline 可以放开显存，但只能在目标 GPU 空闲且不影响其他用户时运行。

结论表述必须谨慎：

- 可以说：“在 A100 上受限到 4090-24G 显存预算时，HeteroKV 支撑了 128K 上下文生存。”
- 不能只凭 A100 latency 声称真实 4090 latency 已被证明。
- 若要严格证明 4090 latency，需要真实 4090 复测。

## 必须验证的主线

1. 阶段 0：安全与静态检查
   - `CUDA_VISIBLE_DEVICES=1`
   - 进程内使用 `cuda:0`
   - `enable_triton=False`
   - 未修改 transformers 源码
   - 未修改模型权重
   - `_prefill_update()` 不再返回 full K/V
   - `mean_k` 不在主检索路径

2. 阶段 1：小张量机制测试
   - prefill 返回短 KV
   - Sink/Tail/Heavy-Hitter 长度符合预算
   - DRAM entry 包含量化 K/V
   - active HBM KV 长度稳定
   - dot-product retrieval 能命中人工目标 chunk

3. 阶段 2：真实模型短上下文冒烟测试
   - Qwen2.5-7B-Instruct
   - 2K、4K、8K
   - `generate()` 正常
   - batch size 1/2 正常
   - `position_ids/cache_position/attention_mask` 不报错

4. 阶段 3：16K/32K 消融对照
   - SinkTail only
   - Sink + Tail + Heavy-Hitter，无 retrieval
   - 旧 mean_k retrieval 对照
   - 新 Query-Key dot-product retrieval
   - NIAH depth：0%、25%、50%、75%、90%、99%

5. 阶段 4：64K/128K 生存测试
   - 22 GiB cap
   - 128K context
   - Qwen2.5-7B-Instruct
   - HBM active KV 不随上下文线性增长
   - DRAM compressed KV 可随上下文增长
   - generate 正常输出

6. 阶段 5：语义损失测试
   - 真实 NIAH，多深度准确率基础目标 >=95%，优秀目标 100%
   - WikiText-2 真实 PPL，不使用 MSE 代理
   - HeteroKV PPL 相对可运行 full KV baseline 劣化 <=5%

7. 阶段 6：Latency breakdown
   - prefill time
   - decode ms/token
   - retrieval scoring time
   - dequant / transfer time
   - end-to-end generate time

## 禁止偏离项

- 禁止把短 KV pad 回 128K。
- 禁止为了修复 shape mismatch 把 full KV 留在 HBM。
- 禁止修改模型权重。
- 禁止修改 transformers 源码。
- 禁止继续使用 mean_k、pooled embedding、cosine similarity 作为主检索路径。
- 禁止在 correctness、NIAH、PPL、显存稳定达标前引入或启用自定义 Triton/CUDA kernel。
- 禁止用日志伪造成功，必须有物理 KV 长度、DRAM entry、显存曲线和真实 generate 结果。
- 禁止在服务器目标 GPU 有其他用户重型任务时启动 128K/baseline 重型实验。

## 最终交付判断

只有当以下证据齐全时，才可判断项目核心目标完成：

- 128K under 22 GiB cap 不 OOM。
- active HBM KV 长度稳定。
- `_prefill_update()` 返回短 KV。
- 4-bit DRAM KV 存储真实发生。
- Query-Key dot-product retrieval 真实生效。
- NIAH 多深度准确率达标。
- WikiText-2 PPL 劣化受控。
- `generate()` 兼容性通过。
- latency 有清晰拆解。
- 对照实验能说明 Sink、Tail、Heavy-Hitter、Retrieval 各自贡献。

