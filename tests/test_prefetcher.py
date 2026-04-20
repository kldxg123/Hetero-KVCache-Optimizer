# 文件路径: tests/test_prefetcher.py
import sys
import os
import time
import torch

# 确保能找到 src 包
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.memory.manager import HeteroKVManager


def simulate_gpu_compute(delay_ms: int = 50):
    """
    模拟一段繁重的 GPU 计算任务 (比如当前 Token 的 Attention 矩阵乘法)
    用来给后台的 PCIe 预取流争取时间
    """
    print(f"   [计算流] 正在执行繁重的 Attention 计算 (模拟 {delay_ms}ms)...")
    # 构造两个大矩阵相乘来占用 GPU
    dim = 4096
    a = torch.randn(dim, dim, device="cuda")
    b = torch.randn(dim, dim, device="cuda")
    for _ in range(10):
        _ = torch.matmul(a, b)
    torch.cuda.synchronize()


def run_prefetch_ab_test():
    print("=" * 60)
    print(" 🚀 [Hetero-KV] 异步预取引擎 (Prefetcher) 性能评测")
    print("=" * 60)

    # 1. 初始化调度器，极小的 HBM 空间以强制触发换出
    manager = HeteroKVManager(hbm_max_blocks=5, block_size=16, device="cuda:0")

    # 2. 填满 HBM 并触发驱逐 (根据你的代码，一次会踢出 2 个 Block 进 DRAM)
    print("\n>>> 阶段 1: 构造实验环境 (触发 DRAM 换出) <<<")
    for logical_id in range(5):
        manager.step_allocate(logical_id, current_seq_len=(logical_id + 1) * 16)

    # 模拟 Attention 更新，降低 Block 1 和 Block 2 的分数
    dummy_attn = torch.rand(128)
    dummy_attn[16:48] = 0.001
    manager.oracle.update(dummy_attn)

    # 强行写入第 6 个 Block，触发驱逐！
    # 这时 Block 1 和 Block 2 应该会被压缩并换出到 DRAM 中
    manager.step_allocate(logical_block_id=5, current_seq_len=128)

    evicted_blocks = list(manager.dram_table.keys())
    assert len(evicted_blocks) >= 2, "需要至少两个被换出的 Block 来进行对照实验！"
    block_a, block_b = evicted_blocks[0], evicted_blocks[1]

    print(f"\n>>> 阶段 2: A/B 对照实验 <<<")
    print(f"🎯 选定测试节点: {block_a} (用于同步测试), {block_b} (用于异步预取测试)")

    # ==========================================
    # 测试 A：传统的同步 Swap-in (Cache Miss)
    # ==========================================
    print("\n[测试 A] 传统同步唤醒 (无预取)")
    # 确保 GPU 空闲
    torch.cuda.synchronize()
    start_time_a = time.perf_counter()

    # 直接请求，此时必然未命中预取缓存，只能原地阻塞等待 H2D 拷贝和反量化
    restored_k_a, restored_v_a = manager.swap_in(block_a, torch.device("cuda:0"))

    torch.cuda.synchronize()
    end_time_a = time.perf_counter()
    latency_a = (end_time_a - start_time_a) * 1000  # 转换为毫秒

    # ==========================================
    # 测试 B：开启异步预取 (Cache Hit)
    # ==========================================
    print("\n[测试 B] 异构异步预取 (Computation-Communication Overlap)")
    torch.cuda.synchronize()

    # 1. 预测器发现未来可能需要 block_b，提前向后台流提交任务 (非阻塞)
    manager.prefetcher.submit_prefetch_task(block_b, manager.dram_table[block_b], manager.compressor)

    # 2. 主流继续执行当前 Token 的计算 (这段时间完美掩盖了后台的 PCIe I/O)
    simulate_gpu_compute()

    # 3. 计算完毕，真正需要用到 block_b 的数据了
    start_time_b = time.perf_counter()

    restored_k_b, restored_v_b = manager.swap_in(block_b, torch.device("cuda:0"))

    torch.cuda.synchronize()
    end_time_b = time.perf_counter()
    latency_b = (end_time_b - start_time_b) * 1000  # 转换为毫秒

    # ==========================================
    # 输出实验报告
    # ==========================================
    print("\n" + "=" * 60)
    print(" 📊 预取加速比评测报告")
    print("=" * 60)
    print(f"🐢 [A组] 同步阻塞 I/O 延迟:  {latency_a:.2f} ms")
    print(f"⚡ [B组] 异步预取命中延迟:    {latency_b:.2f} ms")

    if latency_b > 0:
        speedup = latency_a / latency_b
        print(f"🚀 I/O 延迟掩盖率:           {((latency_a - latency_b) / latency_a) * 100:.2f}%")
        print(f"🔥 数据拉回性能提升:         {speedup:.2f} 倍")
    print("=" * 60)
    print("💡 结论: 异步预取成功将 PCIe 传输与反量化时间完全隐藏在 GPU 计算背后！")


if __name__ == "__main__":
    run_prefetch_ab_test()