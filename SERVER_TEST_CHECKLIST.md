# 服务器开机后测试清单

## 第一步：发送文件到服务器

```bash
# 在本地执行，将修改后的文件发送到服务器
scp /home/app-ahr/Hetero-KVCache-Optimizer/src/memory/manager.py root@208.18.0.10:/root/Hetero-KVCache-Optimizer/src/memory/manager.py

scp /home/app-ahr/Hetero-KVCache-Optimizer/src/memory/attention_competition_queue.py root@208.18.0.10:/root/Hetero-KVCache-Optimizer/src/memory/attention_competition_queue.py

scp /home/app-ahr/Hetero-KVCache-Optimizer/benchmark_128k_fixed.py root@208.18.0.10:/root/Hetero-KVCache-Optimizer/benchmark_128k_fixed.py

scp /home/app-ahr/Hetero-KVCache-Optimizer/REPAIRS_COMPLETED_SUMMARY.md root@208.18.0.10:/root/Hetero-KVCache-Optimizer/REPAIRS_COMPLETED_SUMMARY.md
```

## 第二步：SSH连接到服务器

```bash
ssh root@208.18.0.10
cd /root/Hetero-KVCache-Optimizer
```

## 第三步：运行128K极限压测

```bash
# 激活环境（如果需要）
source activate heterokv  # 或者你的环境名称

# 运行修复后的基准测试
python benchmark_128k_fixed.py
```

## 第四步：观察关键日志

### 1. Triton Kernel调用确认
查找以下日志输出：
```
[Triton-Optimized Adaptive Self-Healing] layer=X | total_chunks=Y window=Z
```

### 2. 三区域架构确认
查找输出头部：
```
╔══════════════════════════════════════════════════════════════════╗
║  HeteroKV FIXED - Three-Zone Architecture (O(1) Memory)        ║
╠══════════════════════════════════════════════════════════════════╣
║  Design:                                                          ║
║    • Sink: 64 tokens (fixed)                                      ║
║    • Tail: 2048 tokens (fixed sliding window)                      ║
║    • HeavyHitter: 4096 tokens (dynamic, high-attention)          ║
║    • Total HBM: ~6208 tokens = O(1)                                 ║
```

### 3. 显存行为观察
查找显存统计：
```
[2000 pairs] Testing... OK | Tokens: XXXXX | Peak: XXXXMB (XX%) | Time: XXs | Acc: ✓
```

## 第五步：验证O(1)行为

### 预期结果：
- **显存峰值**: 应该保持在 ~14-15GB
- **显存增长**: 不应随上下文长度显著增长（O(1)）
- **准确率**: 应该保持100%（零质量下降）

### 成功标志：
```
✓ SUCCESS: Memory is bounded (~14500MB) = O(1) behavior confirmed!
```

### 对比基准：
- **修复前**: 14.4GB → 18GB（+3.6GB增长，非O(1)）
- **修复后（预期）**: 14.5GB → 14.5GB（恒定，O(1)）

## 第六步：查看详细结果

测试完成后，结果保存在：
```
benchmark_128k_fixed_results.json
```

查看结果：
```bash
cat benchmark_128k_fixed_results.json | python -m json.tool
```

## 故障排查

### 如果出现错误：

1. **导入错误**:
   ```bash
   # 检查文件是否正确发送
   ls -la src/memory/attention_competition_queue.py
   ls -la src/memory/manager.py
   ```

2. **CUDA错误**:
   ```bash
   # 检查GPU状态
   nvidia-smi
   ```

3. **内存不足（OOM）**:
   - 这正是我们要测试的场景！
   - 如果在128K tokens前OOM，说明还有问题
   - 如果能通过128K tokens，说明修复成功

### 如果Triton kernel未被调用：
- 检查 `patch_model_for_fused_attention` 是否被正确应用
- 查看 `benchmark_128k_fixed.py` 第119行是否使用 `with patch_model_for_fused_attention(...)`

### 如果显存仍然增长：
- 检查是否正确使用了 `FusedHeteroCache`
- 查看日志中是否有竞争队列的处理信息

## 预期测试时长

- **模型加载**: ~1-2分钟
- **基准测试**: ~10-20分钟（取决于上下文长度和GPU性能）
- **总计**: 约15-25分钟

## 测试完成后

1. 保存结果日志
2. 记录峰值显存
3. 验证准确率
4. 对比修复前后的差异

---

## 快速命令（复制粘贴版）

```bash
# 发送文件
scp src/memory/manager.py root@208.18.0.10:/root/Hetero-KVCache-Optimizer/src/memory/manager.py
scp src/memory/attention_competition_queue.py root@208.18.0.10:/root/Hetero-KVCache-Optimizer/src/memory/attention_competition_queue.py
scp benchmark_128k_fixed.py root@208.18.0.10:/root/Hetero-KVCache-Optimizer/benchmark_128k_fixed.py

# SSH连接并测试
ssh root@208.18.0.10 "cd /root/Hetero-KVCache-Optimizer && nohup python benchmark_128k_fixed.py > test_log.txt 2>&1 &"

# 查看日志
ssh root@208.18.0.10 "tail -f /root/Hetero-KVCache-Optimizer/test_log.txt"
```
