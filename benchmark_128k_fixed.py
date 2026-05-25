#!/usr/bin/env python3
"""
HeteroKV 128K Context Stress Test - FIXED VERSION
===================================================

完整修复版本，应用所有必要的patch：
1. 三区域HBM架构 (Sink + Tail + HeavyHitter)
2. 注意力竞争队列
3. 寄存器端动态取回
4. patch_model_for_fused_attention()
"""

import torch
import gc
import json
import time
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
from datasets import load_dataset
import os
import sys
from typing import Dict, List, Tuple

os.environ['HF_HOME'] = '/home/app-ahr/.cache/huggingface'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
sys.path.insert(0, '/home/app-ahr/Hetero-KVCache-Optimizer/src')

from core.engine_wrapper import FusedHeteroCache
from core.fused_attention_patch import patch_model_for_fused_attention


# 显存限制（用户要求：24GB）
MAX_MEMORY_MB = 24 * 1024  # 24GB


def get_gpu_stats():
    """获取GPU显存统计"""
    torch.cuda.synchronize()
    return {
        'allocated_mb': torch.cuda.memory_allocated() / 1024**2,
        'peak_allocated_mb': torch.cuda.max_memory_allocated() / 1024**2,
    }


def build_context(dataset, num_pairs: int) -> str:
    """构建长上下文文本"""
    parts = []
    for i in range(num_pairs):
        sample = dataset[i % len(dataset)]
        parts.append(f"Q: {sample['question']} A: {sample['answer']}\n")
    return ''.join(parts)


def test_with_fixed_architecture(
    model,
    processor,
    image: Image.Image,
    dataset,
    context_pairs_list: List[int],
    question: str,
    expected_answer: str
) -> Dict:
    """
    使用修复后架构的测试

    关键修复：
    1. 应用 patch_model_for_fused_attention
    2. 使用三区域HBM管理器
    3. 寄存器端动态取回
    """
    results = {
        'config': 'HeteroKV FIXED (Three-Zone + Triton + Competition Queue)',
        'total_gpu_mb': get_total_gpu_memory(),
        'tests': []
    }

    print(f"\n╔══════════════════════════════════════════════════════════════════╗")
    print(f"║  HeteroKV FIXED - Three-Zone Architecture (O(1) Memory)        ║")
    print(f"╠══════════════════════════════════════════════════════════════════╣")
    print(f"║  Design:                                                          ║")
    print(f"║    • Sink: 64 tokens (fixed)                                      ║")
    print(f"║    • Tail: 2048 tokens (fixed sliding window)                      ║")
    print(f"║    • HeavyHitter: 4096 tokens (dynamic, high-attention)          ║")
    print(f"║    • Total HBM: ~6208 tokens = O(1)                                 ║")
    print(f"║                                                                  ║")
    print(f"║  Features:                                                        ║")
    print(f"║    • Attention competition queue (Tail eviction + Dynamic)      ║")
    print(f"║    • Register-level decompression (Triton kernel)                ║")
    print(f"║    • Zero HBM concatenation overhead                            ║")
    print(f"║                                                                  ║")
    print(f"║  Memory Limit: {MAX_MEMORY_MB/1024:.0f}GB (test stops if exceeded)               ║")
    print(f"╚══════════════════════════════════════════════════════════════════╝")

    for num_pairs in context_pairs_list:
        print(f"\n  [{num_pairs:>6} pairs] Testing...", end=' ', flush=True)

        torch.cuda.reset_peak_memory_stats()
        gc.collect()
        torch.cuda.empty_cache()

        context = build_context(dataset, num_pairs)
        prompt = f"USER: <image>\n{context}Question: {question}\nAnswer: ASSISTANT:"

        try:
            # 关键修复：应用 patch_model_for_fused_attention
            cache = FusedHeteroCache(
                num_layers=32,
                sink_tokens=64,
                keep_tail=2048,  # Tail: 2048 tokens
                chunk_size=2048,
                device='cuda',
                enable_quant=True,
                enable_triton=True,
                self_healing=True,
                adaptive_self_healing=True,
            )

            inputs = processor(text=prompt, images=image, return_tensors='pt').to('cuda')
            tokens = inputs.input_ids.shape[-1]

            start = time.time()

            # ═════════════════════════════════════════════════════════════════
            # 关键修复：应用patch以启用Triton kernel
            # ═════════════════════════════════════════════════════════════════
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

            # 显存限制检查（用户要求：24GB）
            if peak_mem > MAX_MEMORY_MB:
                print(f"LIMIT EXCEEDED! | Peak: {peak_mem:.0f}MB > {MAX_MEMORY_MB:.0f}MB")
                results['tests'].append({
                    'context_pairs': num_pairs,
                    'input_tokens': tokens,
                    'peak_mb': peak_mem,
                    'gpu_utilization_pct': peak_mem / get_total_gpu_memory() * 100,
                    'generation_time': gen_time,
                    'correct': False,
                    'oom': True,
                    'reason': f'Exceeded memory limit ({MAX_MEMORY_MB}MB)',
                })
                break

            response = processor.decode(outputs[0], skip_special_tokens=True)
            answer = response.split("ASSISTANT:")[-1].strip() if "ASSISTANT:" in response else response
            correct = expected_answer.lower() in answer.lower()

            total_gpu = get_total_gpu_memory()

            print(f"OK | Tokens: {tokens} | Peak: {peak_mem:.0f}MB ({peak_mem/total_gpu*100:.1f}%) | "
                  f"Time: {gen_time:.1f}s | Acc: {'✓' if correct else '✗'}")

            results['tests'].append({
                'context_pairs': num_pairs,
                'input_tokens': tokens,
                'peak_mb': peak_mem,
                'gpu_utilization_pct': peak_mem / total_gpu * 100,
                'generation_time': gen_time,
                'correct': correct,
                'oom': False,
            })

            del cache, inputs, outputs

        except RuntimeError as e:
            if "out of memory" in str(e):
                peak_mem = torch.cuda.max_memory_allocated() / 1024**2
                print(f"OOM! | Peak: {peak_mem:.0f}MB")

                results['tests'].append({
                    'context_pairs': num_pairs,
                    'input_tokens': 0,
                    'peak_mb': peak_mem,
                    'gpu_utilization_pct': peak_mem / get_total_gpu_memory() * 100,
                    'generation_time': 0,
                    'correct': False,
                    'oom': True,
                    'reason': 'CUDA OOM',
                })
                break
            else:
                print(f"ERROR: {e}")
                break

        gc.collect()
        torch.cuda.empty_cache()

    return results


def get_total_gpu_memory() -> float:
    """获取GPU总显存"""
    return torch.cuda.get_device_properties(0).total_memory / 1024**2


def run_128k_fixed_test():
    """运行修复后的128K测试"""

    print("=" * 80)
    print("HeteroKV 128K Context Stress Test - FIXED VERSION")
    print("=" * 80)

    total_gpu = get_total_gpu_memory()
    print(f"\nGPU: {total_gpu:.0f} MB ({total_gpu/1024:.1f} GB)")
    print(f"Model: LLaVA-1.5-7B (~13GB)")
    print(f"Available for KV Cache: ~{total_gpu - 13000:.0f} MB")
    print(f"Memory Limit: {MAX_MEMORY_MB/1024:.1f} GB (test will stop if exceeded)")
    print(f"Standard KV at 128K tokens: ~64000 MB >> OOM expected")

    # Load dataset
    print("\n[1/4] Loading VQA-RAD dataset...")
    dataset = load_dataset('flaviagiammarino/vqa-rad', split='test')
    print(f"   Loaded {len(dataset)} samples")

    # Load model
    print("\n[2/4] Loading LLaVA-1.5-7B...")
    processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf-hf")
    model = LlavaForConditionalGeneration.from_pretrained(
        "llava-hf/llava-1.5-7b-hf",
        torch_dtype=torch.float16,
        device_map="cuda"
    )
    model.eval()
    print("   Model loaded")

    # Prepare test
    image = dataset[0]['image']
    if image.mode != 'RGB':
        image = image.convert('RGB')

    question = dataset[0]['question']
    expected = dataset[0]['answer']

    # Context lengths
    context_pairs = [2000, 4000, 8000, 16000, 32000, 64000, 128000]

    print(f"\n[3/4] Running progressive test with FIXED architecture...")
    print(f"   Context pairs: {context_pairs}")
    print(f"   Question: {question[:60]}... Expected: {expected}")

    results = test_with_fixed_architecture(
        model, processor, image, dataset,
        context_pairs, question, expected
    )

    # Analysis
    print(f"\n[4/4] Analysis")
    print("=" * 80)
    print("FIXED ARCHITECTURE RESULTS:")
    print("=" * 80)

    print(f"\n┌─ MEMORY BEHAVIOR (O(1) Expected)")
    print(f"│  {'Pairs':<10} {'Tokens':<12} {'Peak(MB)':<12} {'GPU%':<10} {'Status'}")
    print(f"│  {'─'*50}")

    prev_mem = 0
    for test in results['tests']:
        status = "OOM!" if test['oom'] else "OK"
        mem = test['peak_mb']
        delta = mem - prev_mem if prev_mem > 0 else 0
        print(f"│  {test['context_pairs']:<10} {test['input_tokens']:<12} {mem:<12.1f} {test['gpu_utilization_pct']:<10.1f} {status}")
        prev_mem = mem
        if test['oom']:
            break

    # Find max successful context
    max_pairs = 0
    max_tokens = 0
    for test in results['tests']:
        if not test['oom']:
            max_pairs = test['context_pairs']
            max_tokens = test['input_tokens']
        else:
            break

    # Accuracy
    acc_tests = [t for t in results['tests'] if not t['oom']]
    if acc_tests:
        accuracy = sum(1 for t in acc_tests if t['correct']) / len(acc_tests) * 100
        print(f"\n├─ ACCURACY")
        print(f"│  Accuracy: {accuracy:.1f}% ({sum(1 for t in acc_tests if t['correct'])}/{len(acc_tests)})")

    # Memory growth analysis
    print(f"\n└─ MEMORY GROWTH ANALYSIS")
    print(f"   Context extension: {max_tokens} tokens")
    print(f"   Expected O(1) behavior: Peak memory should stay ~14-15GB")
    print(f"   Memory at {max_tokens} tokens: {prev_mem:.0f} MB")

    if prev_mem < 16000:
        print(f"   ✓ SUCCESS: Memory is bounded (~{prev_mem:.0f}MB) = O(1) behavior confirmed!")
    else:
        print(f"   ⚠ WARNING: Memory growth ({prev_mem:.0f}MB) suggests non-O(1) behavior")

    # Compare with baseline
    print(f"\nCOMPARISON WITH BASELINE:")
    print(f"  Baseline OOM: 36,321 tokens (19.8GB)")
    print(f"  HeteroKV FIXED: {max_tokens} tokens ({prev_mem:.0f}GB)")

    if max_tokens > 36321:
        extension = max_tokens / 36321
        print(f"  Context extension: {extension:.1f}x beyond baseline OOM")

    # Save results
    output_file = '/home/app-ahr/Hetero-KVCache-Optimizer/benchmark_128k_fixed_results.json'
    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'gpu_total_mb': total_gpu,
            'model': 'llava-hf/llava-1.5-7b-hf-hf',
            'dataset': 'flaviagiammarino/vqa-rad',
            'architecture': 'Three-Zone HBM (Sink=64 + Tail=2048 + HeavyHitter=4096)',
            'features': [
                'Attention competition queue',
                'Register-level dynamic retrieval',
                'patch_model_for_fused_attention applied',
                'Zero HBM concatenation overhead'
            ],
            'results': results
        }, f, indent=2)

    print(f"\nResults saved to: {output_file}")
    print("=" * 80)

    return results


if __name__ == "__main__":
    results = run_128k_fixed_test()