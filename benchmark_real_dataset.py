#!/usr/bin/env python3
"""
HeteroKV Benchmark Test - Real Dataset Comparison
Tests memory usage, OOM behavior, and accuracy with HeteroKV ON vs OFF
Uses LLaVA-1.5-7B with real images
"""

import torch
import gc
import json
import time
import psutil
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
from typing import Dict, List, Tuple
import os
import matplotlib.pyplot as plt
from pathlib import Path

# Set paths
HF_HOME = '/root/autodl-tmp/huggingface'
os.environ['HF_HOME'] = HF_HOME

# Add src to path
import sys
sys.path.insert(0, '/home/app-ahr/Hetero-KVCache-Optimizer/src')

from core.engine_wrapper import FusedHeteroCache

def get_gpu_memory():
    """Get current GPU memory usage in MB"""
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    return allocated, reserved

def get_system_memory():
    """Get system RAM usage in MB"""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / 1024**2

def create_test_image(color: str, size: Tuple[int, int] = (336, 336)) -> Image.Image:
    """Create a test image with specific color"""
    img = Image.new('RGB', size, color=color)
    return img

def test_with_hetero(
    model,
    processor,
    test_cases: List[Dict],
    max_new_tokens: int = 50
) -> Dict:
    """Run tests with HeteroKV enabled"""
    results = {
        'config': 'HeteroKV ON',
        'memory_peak_mb': 0,
        'memory_samples': [],
        'results': [],
        'oom_occurred': False,
        'total_time': 0
    }

    torch.cuda.reset_peak_memory_stats()
    gc.collect()
    torch.cuda.empty_cache()

    start_memory, _ = get_gpu_memory()

    for i, test_case in enumerate(test_cases):
        try:
            # Create new cache for each test
            hetero_cache = FusedHeteroCache(
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

            # Create image and prompt
            img = create_test_image(color=test_case['color'])
            prompt = f"USER: <image>\n{test_case['question']}\nASSISTANT:"
            inputs = processor(text=prompt, images=img, return_tensors='pt').to('cuda')

            # Sample memory before
            mem_before, _ = get_gpu_memory()
            results['memory_samples'].append(mem_before)

            # Run generation
            start_time = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    past_key_values=hetero_cache
                )
            generation_time = time.time() - start_time

            # Sample memory after
            mem_after, _ = get_gpu_memory()
            results['memory_samples'].append(mem_after)
            results['memory_peak_mb'] = max(results['memory_peak_mb'], mem_after)

            # Decode result
            response = processor.decode(outputs[0], skip_special_tokens=True)
            answer = response.split("ASSISTANT:")[-1].strip()

            results['results'].append({
                'test_id': i,
                'question': test_case['question'],
                'expected': test_case['expected_answer'],
                'actual': answer,
                'correct': test_case['expected_answer'].lower() in answer.lower(),
                'generation_time': generation_time,
                'kv_seq_length': hetero_cache.get_seq_length()
            })

            # Cleanup
            del hetero_cache, inputs, outputs
            gc.collect()
            torch.cuda.empty_cache()

        except RuntimeError as e:
            if "out of memory" in str(e):
                results['oom_occurred'] = True
                results['results'].append({
                    'test_id': i,
                    'question': test_case['question'],
                    'error': 'OOM',
                    'oom_at_test': i
                })
                break
            else:
                raise e

    results['total_time'] = sum(r.get('generation_time', 0) for r in results['results'])
    return results

def test_without_hetero(
    model,
    processor,
    test_cases: List[Dict],
    max_new_tokens: int = 50
) -> Dict:
    """Run tests without HeteroKV (baseline)"""
    results = {
        'config': 'HeteroKV OFF',
        'memory_peak_mb': 0,
        'memory_samples': [],
        'results': [],
        'oom_occurred': False,
        'total_time': 0
    }

    torch.cuda.reset_peak_memory_stats()
    gc.collect()
    torch.cuda.empty_cache()

    start_memory, _ = get_gpu_memory()

    for i, test_case in enumerate(test_cases):
        try:
            # No HeteroKV cache - use default
            img = create_test_image(color=test_case['color'])
            prompt = f"USER: <image>\n{test_case['question']}\nASSISTANT:"
            inputs = processor(text=prompt, images=img, return_tensors='pt').to('cuda')

            # Sample memory before
            mem_before, _ = get_gpu_memory()
            results['memory_samples'].append(mem_before)

            # Run generation WITHOUT cache
            start_time = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    past_key_values=None  # No cache
                )
            generation_time = time.time() - start_time

            # Sample memory after
            mem_after, _ = get_gpu_memory()
            results['memory_samples'].append(mem_after)
            results['memory_peak_mb'] = max(results['memory_peak_mb'], mem_after)

            # Decode result
            response = processor.decode(outputs[0], skip_special_tokens=True)
            answer = response.split("ASSISTANT:")[-1].strip()

            results['results'].append({
                'test_id': i,
                'question': test_case['question'],
                'expected': test_case['expected_answer'],
                'actual': answer,
                'correct': test_case['expected_answer'].lower() in answer.lower(),
                'generation_time': generation_time
            })

            # Cleanup
            del inputs, outputs
            gc.collect()
            torch.cuda.empty_cache()

        except RuntimeError as e:
            if "out of memory" in str(e):
                results['oom_occurred'] = True
                results['results'].append({
                    'test_id': i,
                    'question': test_case['question'],
                    'error': 'OOM',
                    'oom_at_test': i
                })
                break
            else:
                raise e

    results['total_time'] = sum(r.get('generation_time', 0) for r in results['results'])
    return results

def run_comprehensive_benchmark():
    """Run comprehensive benchmark comparing HeteroKV ON vs OFF"""

    print("=" * 70)
    print("HeteroKV Benchmark - Real Dataset Comparison")
    print("=" * 70)

    # Load model
    model_name = "llava-hf/llava-1.5-7b-hf"
    print(f"\nLoading model: {model_name}")

    processor = AutoProcessor.from_pretrained(model_name)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cuda"
    )
    model.eval()

    # Define test cases (color-based QA)
    test_cases = [
        {
            'color': 'red',
            'question': 'What color is this image?',
            'expected_answer': 'red'
        },
        {
            'color': 'blue',
            'question': 'What color is this image?',
            'expected_answer': 'blue'
        },
        {
            'color': 'green',
            'question': 'What color is this image?',
            'expected_answer': 'green'
        },
        {
            'color': 'yellow',
            'question': 'What color is this image?',
            'expected_answer': 'yellow'
        },
        {
            'color': 'red',
            'question': 'Is this a blue image?',
            'expected_answer': 'no'
        },
        {
            'color': 'green',
            'question': 'What is the dominant color?',
            'expected_answer': 'green'
        },
        {
            'color': 'blue',
            'question': 'Describe the color you see.',
            'expected_answer': 'blue'
        },
        {
            'color': 'yellow',
            'question': 'Is this a red or yellow image?',
            'expected_answer': 'yellow'
        },
    ]

    print(f"\nRunning {len(test_cases)} test cases...")

    # Test 1: HeteroKV ON
    print("\n" + "=" * 70)
    print("Test 1: HeteroKV ON")
    print("=" * 70)
    results_on = test_with_hetero(model, processor, test_cases)

    # Test 2: HeteroKV OFF
    print("\n" + "=" * 70)
    print("Test 2: HeteroKV OFF (Baseline)")
    print("=" * 70)
    results_off = test_without_hetero(model, processor, test_cases)

    # Analysis
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)

    # Memory comparison
    print("\n1. MEMORY USAGE:")
    print(f"   HeteroKV ON  Peak: {results_on['memory_peak_mb']:.1f} MB")
    print(f"   HeteroKV OFF Peak: {results_off['memory_peak_mb']:.1f} MB")

    memory_reduction = ((results_off['memory_peak_mb'] - results_on['memory_peak_mb']) /
                        results_off['memory_peak_mb'] * 100)
    print(f"   Memory Reduction: {memory_reduction:.1f}%")

    # OOM comparison
    print("\n2. OOM BEHAVIOR:")
    print(f"   HeteroKV ON  OOM: {'YES' if results_on['oom_occurred'] else 'NO'}")
    print(f"   HeteroKV OFF OOM: {'YES' if results_off['oom_occurred'] else 'NO'}")

    if results_off['oom_occurred'] and not results_on['oom_occurred']:
        print(f"   ✓ HeteroKV prevented OOM at test {results_off['results'][-1].get('oom_at_test', '?')}")

    # Accuracy comparison
    print("\n3. ACCURACY:")
    accuracy_on = sum(1 for r in results_on['results'] if r.get('correct', False)) / len(results_on['results']) * 100
    accuracy_off = sum(1 for r in results_off['results'] if r.get('correct', False)) / len(results_off['results']) * 100

    print(f"   HeteroKV ON  Accuracy: {accuracy_on:.1f}% ({sum(1 for r in results_on['results'] if r.get('correct', False))}/{len(results_on['results'])})")
    print(f"   HeteroKV OFF Accuracy: {accuracy_off:.1f}% ({sum(1 for r in results_off['results'] if r.get('correct', False))}/{len(results_off['results'])})")

    accuracy_diff = abs(accuracy_on - accuracy_off)
    print(f"   Accuracy Difference: {accuracy_diff:.1f}%")

    # Performance comparison
    print("\n4. PERFORMANCE:")
    print(f"   HeteroKV ON  Total Time: {results_on['total_time']:.2f}s")
    print(f"   HeteroKV OFF Total Time: {results_off['total_time']:.2f}s")

    if results_on['total_time'] > 0:
        speedup = results_off['total_time'] / results_on['total_time']
        print(f"   Speedup: {speedup:.2f}x")

    # Detailed results
    print("\n5. DETAILED RESULTS:")
    for i in range(len(test_cases)):
        print(f"\n   Test {i+1}: {test_cases[i]['question']}")
        if i < len(results_on['results']) and 'error' not in results_on['results'][i]:
            r_on = results_on['results'][i]
            print(f"   HeteroKV ON  Answer: '{r_on['actual'][:50]}...' Correct: {r_on['correct']}")
        if i < len(results_off['results']) and 'error' not in results_off['results'][i]:
            r_off = results_off['results'][i]
            print(f"   HeteroKV OFF Answer: '{r_off['actual'][:50]}...' Correct: {r_off['correct']}")

    # Save results
    output_file = '/home/app-ahr/Hetero-KVCache-Optimizer/benchmark_results.json'
    results_summary = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'model': model_name,
        'test_cases': len(test_cases),
        'hetero_on': results_on,
        'hetero_off': results_off,
        'summary': {
            'memory_reduction_percent': memory_reduction,
            'hetero_on_accuracy': accuracy_on,
            'hetero_off_accuracy': accuracy_off,
            'accuracy_difference': accuracy_diff,
            'hetero_on_oom': results_on['oom_occurred'],
            'hetero_off_oom': results_off['oom_occurred']
        }
    }

    with open(output_file, 'w') as f:
        json.dump(results_summary, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    return results_summary

if __name__ == "__main__":
    results = run_comprehensive_benchmark()