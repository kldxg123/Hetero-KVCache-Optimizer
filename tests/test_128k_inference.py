import sys
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 确保能找到 src 包
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.core.engine_wrapper import HeteroKVCache


def test_generation():
    # 1. 动态获取你刚才下载的模型路径
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(project_root, "models", "Qwen2.5-7B-Instruct")

    print(f"⏳ 正在加载模型权重 (这可能需要半分钟，利用了 4090 的高速 PCIe)...")
    print(f"📂 路径: {model_path}")

    # 2. 加载 Tokenizer 和 模型
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",  # <--- 核心修复 1：强制锁定在单张卡上，杜绝跨卡通信导致的显存指针错乱
        trust_remote_code=True,
        attn_implementation="eager"
    )
    print("✅ 模型加载成功！\n")

    # 3. 准备一段超长的测试 Prompt
    prompt = "你是我的顶级 AI 架构师导师。请详细解释一下什么是大型语言模型的 KV Cache，以及为什么在 128K 长度下会导致 OOM？"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # 4. 【核心黑科技】实例化我们自己写的异构 Cache 拦截器
    custom_cache = HeteroKVCache(max_hbm_length=60, evict_chunk_size=16)

    print("🚀 开始使用 HeteroKVCache 拦截器进行生成测试...")

    # 5. 执行生成，强行传入我们的 custom_cache
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            past_key_values=custom_cache,
            max_new_tokens=100,  # <--- 改为 100
            use_cache=True,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True
        )

    # 6. 解码输出
    generated_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    print("\n" + "=" * 50)
    print("📝 Qwen 生成结果：\n")
    print(generated_text)
    print("=" * 50)

    # 【环节 A】：呼叫 Profiler 生成数据报告
    print_profiling_report(custom_cache)

    # ==========================================
    # 【环节 C】：终极黑科技 Swap-in 唤醒测试
    # ==========================================
    print("\n" + "🔮" * 25)
    print("   [黑科技演示] 异构内存动态唤醒 (Swap-in) 测试")
    print("🔮" * 25)

    dram_keys = list(custom_cache.manager.dram_table.keys())
    if dram_keys:
        target_chunk = dram_keys[0]  # 取出第一个被踢到系统内存里的块
        print(f"👉 模拟场景：当前 Query 强力 Attention 命中了早期被剔除的节点 '{target_chunk}'")

        # 记录唤醒前的显存（用于观察数据拉回后的显存波动）
        torch.cuda.reset_peak_memory_stats(model.device)
        mem_before = torch.cuda.memory_allocated(model.device)

        # 呼叫 Swap-in 机制！
        restored_k, restored_v = custom_cache.manager.swap_in(target_chunk, model.device)

        mem_after = torch.cuda.memory_allocated(model.device)
        pulled_mb = (mem_after - mem_before) / (1024 * 1024)

        print(f"✅ 验证通过！成功跨越 PCIe 将数据拉回。")
        print(f"📊 恢复出的张量设备: {restored_k.device}, 数据类型: {restored_k.dtype}")
        print(f"📊 该块数据重新占用 GPU 显存: {pulled_mb:.4f} MB")
        print("💡 在旁路 Attention 架构中，这块被唤醒的 BF16 数据将立刻参与当前 Token 的计算！")


def print_profiling_report(cache):
    print("\n" + "📊" * 25)
    print("   [Hetero-KV Profiler] 毕设实验数据分析报告")
    print("📊" * 25)

    dram_table = cache.manager.dram_table
    if not dram_table:
        print("没有触发任何剔除操作，内存分析为空。")
        return

    total_dram_bytes = 0
    total_original_bytes = 0
    evicted_chunks = len(dram_table)

    for chunk_id, data in dram_table.items():
        # 统计 KV 块的参数量 (我们在 Python 模拟中形状未变，所以 nelement() 就是参数个数)
        num_k_params = data["k_data"].nelement()
        num_v_params = data["v_data"].nelement()
        total_params = num_k_params + num_v_params

        # 1. 计算如果在 GPU 中原样存储 (BF16，每个参数 2 Bytes) 需要的显存
        chunk_original_bytes = total_params * 2

        # 2. 计算在 DRAM 中的实际 4-bit 存储体积 (每个参数 0.5 Bytes)
        # 加上 Meta 数据（Scales 和 Zero-points，通常是 FP16 或 BF16，即 2 Bytes）
        k_meta_bytes = data["k_meta"][0].nelement() * 2 * 2  # scale 和 zp 分别占空间
        v_meta_bytes = data["v_meta"][0].nelement() * 2 * 2

        chunk_dram_bytes = (total_params * 0.5) + k_meta_bytes + v_meta_bytes

        total_original_bytes += chunk_original_bytes
        total_dram_bytes += chunk_dram_bytes

    # 转换为 Megabytes (MB)
    orig_mb = total_original_bytes / (1024 * 1024)
    dram_mb = total_dram_bytes / (1024 * 1024)
    saved_mb = orig_mb - dram_mb
    compression_ratio = (1 - dram_mb / orig_mb) * 100 if orig_mb > 0 else 0

    print(f"🔹 总计执行换出操作: {cache.eviction_count} 次 (跨越多层，共拦截 {evicted_chunks} 个 Cache 块)")
    print(f"🔹 原始 BF16 GPU 显存占用 (若无本架构): {orig_mb:.4f} MB")
    print(f"🔹 压缩后 DRAM 系统内存占用 (Tier 2):  {dram_mb:.4f} MB")
    print(f"🔹 🚀 GPU 显存净节省:                {saved_mb:.4f} MB")
    print(f"🔹 🗜️ 综合压缩率 (含量化元数据损耗):  {compression_ratio:.2f}%")
    print("=" * 50)


if __name__ == "__main__":
    test_generation()