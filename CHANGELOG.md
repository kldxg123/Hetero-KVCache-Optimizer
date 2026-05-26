# Changelog

## v1.0.2 (2026-05-26) — Method D: Query-Aware Retrieval

### New Features
- **Method D (Query-Aware Retrieval)**: 新增基于查询感知的DRAM zone检索策略
  - `src/memory/query_aware_retriever.py` — 核心实现：使用 query embedding 与 chunk embeddings 的余弦相似度，动态选择与当前查询最相关的 KV chunk
  - `src/core/engine_wrapper.py` — 新增 `enable_method_d` / `method_d_alpha` 参数，支持独立启用 Method D
  - `src/core/fused_attention_patch.py` — 在注意力计算中集成 Method D 的检索结果

- **三区架构 (Three-Zone Architecture)**: Sink / DRAM / Tail 三级存储
  - `src/memory/manager_three_zone.py` — 三区管理器实现
  - `src/memory/manager_three_zone_fixed.py` — 修复版三区管理器

- **注意力竞争队列**: `src/memory/attention_competition_queue.py`

- **混合自愈策略**: `src/memory/method_e_hybrid_healing.py`

### Changed (from v1.0.1)
- `src/memory/manager.py` — 扩展管理器接口，支持 Method D 的 query-aware 检索

### Test & Benchmark
- `test_method_d_v2.py` — Method D vs Method C 对比测试 (WikiText-2 + Needle)
- `test_method_d_comparison.py` — 初版 Method D 对比测试
- `benchmark_128k.py` / `benchmark_128k_simple.py` — 128K 长上下文基准测试
- `benchmark_long_context.py` — 长上下文基准测试
- `benchmark_real_dataset.py` — 真实数据集基准测试
- `test_real_datasets.py` / `test_real_datasets_v2.py` — 真实数据集测试

### v1.0.1 → v1.0.2 差异总结
| 维度 | v1.0.1 | v1.0.2 |
|------|--------|--------|
| DRAM 检索策略 | Method C: 历史注意力分数预测 | Method D: Query-aware 余弦相似度 |
| 新增文件 | — | +6 核心模块, +10 测试/基准 |
| 代码变更 | — | ~9900 行新增 |

---

## v1.0.1 (2026-05-25) — Critical Fixes

### Fixed
- 修复 tensor 索引越界、mask 尺寸不匹配、prefill 截断等关键bug
- 确保 O(1) KV 内存增长 + 100% 准确率

## v1.0.0 (2026-05-24) — Initial Release

### Features
- FusedHeteroCache: 异构 KV Cache 管理
- 4-bit KV 量化 (group-wise uniform quantization)
- Sink + DRAM + Tail 三区架构
- 自适应动态窗口自愈机制
- Triton 内核加速
