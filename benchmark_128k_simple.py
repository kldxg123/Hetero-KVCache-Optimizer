#!/usr/bin/env python3
"""
HeteroKV 128K Context Stress Test - Simplified
Progressive context scaling: 1K → 2K → 4K → 8K → 16K → 32K → 64K
Focus on memory monitoring and OOM behavior
"""

import torch
import gc
import json
import time
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
from datasets import load_dataset
from typing import Dict, List
import os
import sys

# Set paths
HF_HOME = '/root/autodl-tmp/huggingface'
os.environ['HF_HOME'] = HF_HOME
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# Add src to path
sys.path.insert(0, '/home/app-ahr/Hetero-KVCache-Optimizer/src')

from core.engine_wrapper import FusedHeteroCache


def get_gpu_stats() -> Dict:
    """Get detailed GPU memory stats"""
    torch.cuda.synchronize()
    return {
        'allocated_mb': torch.cuda.memory_allocated() / 1024**2,
        'reserved_mb': torch.cuda.memory_reserved() / 1024**2,
        'peak_allocated_mb': torch.cuda.max_memory_allocated() / 1024**2,
        'peak_reserved_mb': torch.cuda.max_memory_reserved() / 1024**2,
    }


def get_total_gpu_memory() -> float:
    """Get total GPU memory in MB"""
    return torch.cuda.get_device_properties(0).total_memory / 1024**2


def build_long_context(dataset, num_pairs: int) -> str:
    """Build long context text"""
    parts = []
    for i in range(num_pairs):
        sample = dataset[i % len(dataset)]
        parts.append(f"Q: {sample['question']} A: {sample['answer']}\n")
    return "".join(parts)


def test_single_context(
    model,
    processor,
    image: Image.Image,
    context_text: str,
    question: str,
    expected_answer: str,
    use_hetero: bool
) -> Dict:
    """Test a single context length"""
    result = {
        'use_hetero': use_hetero,
        'context_length': len(context_text),
        'oom': False,
        'error': None,
        'memory': {},
        'answer': '',
        'correct': False,
        'time': 0,
        'input_tokens': 0
    }

    torch.cuda.reset_peak_memory_stats()
    gc.collect()
    torch.cuda.empty_cache()

    baseline = get_gpu_stats()

    try:
        prompt = f"USER: <image>\n{context_text}\nQuestion: {question}\nAnswer: ASSISTANT:"
        inputs = processor(text=prompt, images=image, return_tensors='pt').to('cuda')
        result['input_tokens'] = inputs.input_ids.shape[-1]

        cache = None
        if use_hetero:
            cache = FusedHeteroCache(
                num_layers=32,
                sink_tokens=64,
                keep_tail=2048,
                chunk_size=2048,
                device='cuda',
                enable_quant=True,
                enable_triton=True,
                self_healing=True,
                adaptive_self_healing=True,
            )

        start_time = time.time()
        with torch.no_grad():
            # Use direct parameters instead of **inputs to avoid cache_position issues
            outputs = model.generate(
                input_ids=inputs.input_ids,
                pixel_values=inputs.pixel_values,
                attention_mask=inputs.attention_mask,
                max_new_tokens=20,
                do_sample=False,
                past_key_values=cache
            )
        gen_time = time.time() - start_time

        result['time'] = gen_time

        # Decode
        response = processor.decode(outputs[0], skip_special_tokens=True)
        answer = response.split("ASSISTANT:")[-1].strip() if "ASSISTANT:" in response else response
        result['answer'] = answer

        # Check correctness (simple keyword match)
        result['correct'] = expected_answer.lower() in answer.lower()

        # Memory stats
        post = get_gpu_stats()
        result['memory'] = {
            'peak_allocated_mb': post['peak_allocated_mb'],
            'gpu_utilization_pct': post['peak_allocated_mb'] / get_total_gpu_memory() * 100,
        }

        if cache:
            del cache
        del inputs, outputs

    except RuntimeError as e:
        if "out of memory" in str(e):
            result['oom'] = True
            post = get_gpu_stats()
            result['memory']['peak_allocated_mb'] = post['peak_allocated_mb']
            result['memory']['gpu_utilization_pct'] = post['peak_allocated_mb'] / get_total_gpu_memory() * 100
        else:
            result['error'] = str(e)

    except Exception as e:
        result['error'] = str(e)

    gc.collect()
    torch.cuda.empty_cache()
    return result


def run_128k_simplified_test():
    """Simplified 128K test"""

    print("=" * 70)
    print("HeteroKV 128K Context Stress Test - Simplified")
    print("=" * 70)

    total_gpu = get_total_gpu_memory()
    print(f"\nGPU: {total_gpu:.0f} MB ({total_gpu/1024:.1f} GB)")
    print(f"Available for KV: ~{total_gpu - 13000:.0f} MB")
    print(f"Standard KV at 128K: ~64000 MB >> OOM expected")

    # Load dataset
    print("\n[1/4] Loading VQA-RAD...")
    dataset = load_dataset('flaviagiammarino/vqa-rad', split='test')
    print(f"   Loaded {len(dataset)} samples")

    # Load model
    print("\n[2/4] Loading LLaVA-1.5-7B...")
    processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
    model = LlavaForConditionalGeneration.from_pretrained(
        "llava-hf/llava-1.5-7b-hf",
        torch_dtype=torch.float16,
        device_map="cuda"
    )
    model.eval()
    print("   Model loaded")

    # Prepare image and test
    image = dataset[0]['image']
    if image.mode != 'RGB':
        image = image.convert('RGB')

    question = dataset[0]['question']
    expected = dataset[0]['answer']
    print(f"\n[3/4] Test: {question[:60]}... Expected: {expected}")

    # Progressive context lengths (Q&A pairs)
    context_pairs_list = [50, 100, 200, 400, 800, 1600]  # Will result in ~128K tokens
    print(f"\n[4/4] Running progressive tests...")
    print(f"   Context pairs: {context_pairs_list}")

    results = {
        'metadata': {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'gpu_total_mb': total_gpu,
            'test_question': question,
            'expected_answer': expected,
        },
        'tests': []
    }

    print(f"\n{'='*70}")
    print(f"{'Pairs':<8} {'Tokens':<10} {'Status':<8} {'Peak(MB)':<12} {'GPU%':<8} {'Acc':<6}")
    print(f"{'='*70}")

    hetero_max_tokens = 0
    baseline_oom_at = None

    # Test HeteroKV ON first (it should handle all lengths)
    for num_pairs in context_pairs_list:
        context = build_long_context(dataset, num_pairs)

        print(f"\n  [{num_pairs} pairs] HeteroKV ON...", end=' ')
        result_on = test_single_context(model, processor, image, context, question, expected, use_hetero=True)

        if not result_on['oom'] and not result_on.get('error'):
            hetero_max_tokens = result_on['input_tokens']
            print(f"OK | {result_on['input_tokens']} tokens | "
                  f"{result_on['memory']['peak_allocated_mb']:.0f}MB "
                  f"({result_on['memory']['gpu_utilization_pct']:.1f}%) | "
                  f"{'✓' if result_on['correct'] else '✗'}")
        else:
            status = "OOM" if result_on['oom'] else "ERR"
            print(f"{status} | {result_on.get('error', 'OOM')[:40]}")

        results['tests'].append({
            'context_pairs': num_pairs,
            'hetero_on': result_on,
        })

    # Test baseline until OOM
    print(f"\n{'='*70}")
    print("Testing baseline (OFF) - will stop at first OOM...")

    for num_pairs in context_pairs_list:
        if num_pairs > context_pairs_list[2]:  # Only test shorter contexts for baseline
            break

        context = build_long_context(dataset, num_pairs)

        print(f"\n  [{num_pairs} pairs] Baseline OFF...", end=' ')
        result_off = test_single_context(model, processor, image, context, question, expected, use_hetero=False)

        if result_off['oom']:
            baseline_oom_at = result_off['input_tokens']
            print(f"OOM at {result_off['input_tokens']} tokens!")
            break
        elif result_off.get('error'):
            print(f"ERR: {result_off['error'][:40]}")
            break
        else:
            print(f"OK | {result_off['input_tokens']} tokens | "
                  f"{result_off['memory']['peak_allocated_mb']:.0f}MB "
                  f"({result_off['memory']['gpu_utilization_pct']:.1f}%) | "
                  f"{'✓' if result_off['correct'] else '✗'}")

        # Update the test results
        for test in results['tests']:
            if test['context_pairs'] == num_pairs:
                test['hetero_off'] = result_off

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"\n  Baseline OOM: {baseline_oom_at or 'N/A'} tokens")
    print(f"  HeteroKV max: {hetero_max_tokens} tokens")

    if baseline_oom_at and hetero_max_tokens > baseline_oom_at:
        print(f"  Context extension: {hetero_max_tokens / baseline_oom_at:.1f}x")

    # Memory growth analysis
    print(f"\n  HeteroKV Memory Growth:")
    prev_mem = 0
    for test in results['tests']:
        if 'hetero_on' in test and not test['hetero_on']['oom']:
            r = test['hetero_on']
            mem = r['memory']['peak_allocated_mb']
            delta = mem - prev_mem if prev_mem > 0 else 0
            print(f"    {r['input_tokens']:>7} tokens → {mem:.0f}MB (Δ={delta:+.0f}MB)")
            prev_mem = mem

    # Accuracy
    acc_on = sum(1 for t in results['tests'] if 'hetero_on' in t and not t['hetero_on']['oom'] and t['hetero_on']['correct'])
    total_on = sum(1 for t in results['tests'] if 'hetero_on' in t and not t['hetero_on']['oom'])
    if total_on > 0:
        print(f"\n  HeteroKV Accuracy: {acc_on}/{total_on} ({acc_on/total_on*100:.0f}%)")

    # Save
    output_file = '/home/app-ahr/Hetero-KVCache-Optimizer/benchmark_128k_simple.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")
    print(f"{'='*70}")

    return results


if __name__ == "__main__":
    results = run_128k_simplified_test()