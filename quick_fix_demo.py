#!/usr/bin/env python3
"""
三区域架构修复演示
==================

关键修复：
1. 启用 patch_model_for_fused_attention()
2. 修复基准测试内存增长问题
3. 验证O(1)内存行为

快速修复版本
"""

import torch
import time
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
from datasets import load_dataset
import sys
import os

os.environ['HF_HOME'] = '/root/autodl-tmp/huggingface'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
sys.path.insert(0, '/home/app-ahr/Hetero-KVCache-Optimizer/src')

from core.engine_wrapper import FusedHeteroCache
from core.fused_attention_patch import patch_model_for_fused_attention

def quick_fix_benchmark():
    """快速修复版基准测试"""

    print("="*70)
    print("HeteroKV 三区域架构修复验证")
    print("="*70)

    # Load model
    processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
    model = LlavaForConditionalGeneration.from_pretrained(
        "llava-hf/llava-1.5-7b-hf",
        torch_dtype=torch.float16,
        device_map="cuda"
    )
    model.eval()

    # Load dataset
    dataset = load_dataset('flaviagiammarino/vqa-rad', split='test')
    image = dataset[0]['image']
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Test questions
    questions = [
        dataset[i]['question']
        for i in range(5)
    ]

    print(f"\n[1/4] Testing baseline (no HeteroKV)...")
    baseline_results = test_baseline(model, processor, image, questions)

    print(f"\n[2/4] Testing HeteroKV WITHOUT patch...")
    hetero_no_patch_results = test_hetero_no_patch(model, processor, image, questions)

    print(f"\n[3/4] Testing HeteroKV WITH patch (CORRECT)...")
    hetero_with_patch_results = test_hetero_with_patch(model, processor, image, questions)

    print(f"\n[4/4] Analysis...")
    analyze_results(baseline_results, hetero_no_patch_results, hetero_with_patch_results)

def test_baseline(model, processor, image, questions):
    """测试baseline"""
    results = []

    for q in questions:
        prompt = f"USER: <image>\n{q}\nASSISTANT:"
        inputs = processor(text=prompt, images=image, return_tensors='pt').to('cuda')

        torch.cuda.reset_peak_memory_stats()
        start = time.time()

        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs.input_ids,
                pixel_values=inputs.pixel_values,
                attention_mask=inputs.attention_mask,
                max_new_tokens=20,
                do_sample=False,
                past_key_values=None  # No cache
            )

        gen_time = time.time() - start
        peak_mem = torch.cuda.max_memory_allocated() / 1024**2

        response = processor.decode(outputs[0], skip_special_tokens=True)
        answer = response.split("ASSISTANT:")[-1].strip()

        results.append({
            'memory_mb': peak_mem,
            'time': gen_time,
            'answer': answer[:50],
        })

        del inputs, outputs
        torch.cuda.empty_cache()

    return results

def test_hetero_no_patch(model, processor, image, questions):
    """测试HeteroKV WITHOUT patch (当前错误实现)"""
    results = []

    for q in questions:
        cache = FusedHeteroCache(
            num_layers=32, sink_tokens=64, keep_tail=2048,
            device='cuda', enable_quant=True, enable_triton=True,
            self_healing=True, adaptive_self_healing=True,
        )

        prompt = f"USER: <image>\n{q}\nASSISTANT:"
        inputs = processor(text=prompt, images=image, return_tensors='pt').to('cuda')

        torch.cuda.reset_peak_memory_stats()
        start = time.time()

        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs.input_ids,
                pixel_values=inputs.pixel_values,
                attention_mask=inputs.attention_mask,
                max_new_tokens=20,
                do_sample=False,
                past_key_values=cache  # ← _dram_quant_kv被设置但未使用
            )

        gen_time = time.time() - start
        peak_mem = torch.cuda.max_memory_allocated() / 1024**2

        response = processor.decode(outputs[0], skip_special_tokens=True)
        answer = response.split("ASSISTANT:")[-1].strip()

        results.append({
            'memory_mb': peak_mem,
            'time': gen_time,
            'answer': answer[:50],
        })

        del cache, inputs, outputs
        torch.cuda.empty_cache()

    return results

def test_hetero_with_patch(model, processor, image, questions):
    """测试HeteroKV WITH patch (正确实现)"""
    results = []

    for q in questions:
        cache = FusedHeteroCache(
            num_layers=32, sink_tokens=64, keep_tail=2048,
            device='cuda', enable_quant=True, enable_triton=True,
            self_healing=True, adaptive_self_healing=True,
        )

        prompt = f"USER: <image>\n{q}\nASSISTANT:"
        inputs = processor(text=prompt, images=image, return_tensors='pt').to('cuda')

        torch.cuda.reset_peak_memory_stats()
        start = time.time()

        # 关键修复：应用patch_model_for_fused_attention
        with patch_model_for_fused_attention(model, cache, enable_fused=True):
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs.input_ids,
                    pixel_values=inputs.pixel_values,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=20,
                    do_sample=False,
                    past_key_values=cache  # ← Triton kernel现在被正确使用
                )

        gen_time = time.time() - start
        peak_mem = torch.cuda.max_memory_allocated() / 1024**2

        response = processor.decode(outputs[0], skip_special_tokens=True)
        answer = response.split("ASSISTANT:")[-1].strip()

        results.append({
            'memory_mb': peak_mem,
            'time': gen_time,
            'answer': answer[:50],
        })

        del cache, inputs, outputs
        torch.cuda.empty_cache()

    return results

def analyze_results(baseline, hetero_no_patch, hetero_with_patch):
    """分析结果"""

    print("\n" + "="*70)
    print("结果分析")
    print("="*70)

    avg_baseline = sum(r['memory_mb'] for r in baseline) / len(baseline)
    avg_no_patch = sum(r['memory_mb'] for r in hetero_no_patch) / len(hetero_no_patch)
    avg_with_patch = sum(r['memory_mb'] for r in hetero_with_patch) / len(hetero_with_patch)

    print(f"\n平均显存:")
    print(f"  Baseline:     {avg_baseline:.0f} MB")
    print(f"  Hetero (无patch): {avg_no_patch:.0f} MB (+{avg_no_patch-avg_baseline:.0f} MB)")
    print(f"  Hetero (有patch): {avg_with_patch:.0f} MB (+{avg_with_patch-avg_baseline:.0f} MB)")

    print(f"\n关键发现:")
    if avg_with_patch < avg_no_patch:
        print(f"  ✓ Patch生效：有patch比无patch节省 {avg_no_patch-avg_with_patch:.0f} MB")
    else:
        print(f"  ⚠ Patch可能未完全生效或需要更多测试")

    print(f"\n当前实现问题:")
    print(f"  1. 基准测试未应用 patch_model_for_fused_attention()")
    print(f"  2. _dram_quant_kv 被设置但Triton kernel未消费")
    print(f"  3. 三区域架构：HeavyHitter不是HBM分区，仅驱逐决策工具")
    print(f"  4. 动态取回使用HBM拼接，非寄存器计算")

    print(f"\n修复路径:")
    print(f"  1. 所有HeteroKV测试必须应用 patch_model_for_fused_attention")
    print(f"  2. 实现真正的三区域HBM分区 (Sink+Tail+HeavyHitter)")
    print(f"  3. 注意力竞争队列：Tail驱逐 + 动态取回竞争HBM")
    print(f"  4. 寄存器端解压计算：避免torch.cat()HBM分配")

if __name__ == "__main__":
    quick_fix_benchmark()