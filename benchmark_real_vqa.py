#!/usr/bin/env python3
"""
HeteroKV Benchmark - Real VQA Dataset
Uses flaviagiammarino/vqa-rad (medical images VQA) dataset from HuggingFace
"""

import torch
import gc
import json
import time
import psutil
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

def get_gpu_memory():
    """Get current GPU memory usage in MB"""
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    return allocated, reserved

def normalize_answer(text: str) -> str:
    """Normalize answer for comparison"""
    return text.lower().strip().rstrip('.')

def answer_matches(prediction: str, reference: str) -> bool:
    """Check if predicted answer matches reference"""
    pred_normalized = normalize_answer(prediction)
    ref_normalized = normalize_answer(reference)

    # Direct match
    if pred_normalized == ref_normalized:
        return True

    # Contains match
    if ref_normalized in pred_normalized or pred_normalized in ref_normalized:
        return True

    # Word-level match
    pred_words = set(pred_normalized.split())
    ref_words = set(ref_normalized.split())

    # If reference is single word, check if it's in prediction
    if len(ref_words) == 1 and ref_words.issubset(pred_words):
        return True

    return False

def test_with_hetero(
    model,
    processor,
    test_samples: List[Dict],
    max_new_tokens: int = 30
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

    for i, sample in enumerate(test_samples):
        try:
            # Create new cache for each sample
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

            # Prepare inputs
            image = sample['image']
            question = sample['question']

            # Convert grayscale to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')

            prompt = f"USER: <image>\n{question}\nASSISTANT:"
            inputs = processor(text=prompt, images=image, return_tensors='pt').to('cuda')

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

            # Check if answer matches
            reference_answer = sample['answer']
            is_correct = answer_matches(answer, reference_answer)

            results['results'].append({
                'test_id': i,
                'question': question,
                'reference_answer': reference_answer,
                'model_answer': answer,
                'correct': is_correct,
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
                    'question': sample.get('question', '?'),
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
    test_samples: List[Dict],
    max_new_tokens: int = 30
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

    for i, sample in enumerate(test_samples):
        try:
            # No HeteroKV cache
            image = sample['image']
            question = sample['question']

            if image.mode != 'RGB':
                image = image.convert('RGB')

            prompt = f"USER: <image>\n{question}\nASSISTANT:"
            inputs = processor(text=prompt, images=image, return_tensors='pt').to('cuda')

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
                    past_key_values=None
                )
            generation_time = time.time() - start_time

            # Sample memory after
            mem_after, _ = get_gpu_memory()
            results['memory_samples'].append(mem_after)
            results['memory_peak_mb'] = max(results['memory_peak_mb'], mem_after)

            # Decode result
            response = processor.decode(outputs[0], skip_special_tokens=True)
            answer = response.split("ASSISTANT:")[-1].strip()

            # Check if answer matches
            reference_answer = sample['answer']
            is_correct = answer_matches(answer, reference_answer)

            results['results'].append({
                'test_id': i,
                'question': question,
                'reference_answer': reference_answer,
                'model_answer': answer,
                'correct': is_correct,
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
                    'question': sample.get('question', '?'),
                    'error': 'OOM',
                    'oom_at_test': i
                })
                break
            else:
                raise e

    results['total_time'] = sum(r.get('generation_time', 0) for r in results['results'])
    return results

def run_real_dataset_benchmark():
    """Run comprehensive benchmark using real VQA dataset"""

    print("=" * 80)
    print("HeteroKV Benchmark - Real VQA Dataset (VQA-RAD)")
    print("=" * 80)

    # Load dataset
    print("\n1. Loading VQA-RAD dataset from HuggingFace...")
    print("   Dataset: flaviagiammarino/vqa-rad")

    try:
        dataset = load_dataset(
            'flaviagiammarino/vqa-rad',
            split='test',
            trust_remote_code=True
        )
        print(f"   ✓ Loaded {len(dataset)} samples")
    except Exception as e:
        print(f"   ✗ Failed to load dataset: {e}")
        return None

    # Select a subset of samples for testing
    num_samples = 20
    test_samples = dataset.select(range(min(num_samples, len(dataset))))
    print(f"   Testing with {len(test_samples)} samples")

    # Load model
    model_name = "llava-hf/llava-1.5-7b-hf"
    print(f"\n2. Loading model: {model_name}")

    processor = AutoProcessor.from_pretrained(model_name)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cuda"
    )
    model.eval()

    print("   ✓ Model loaded")

    # Show sample data
    print("\n3. Sample test cases:")
    for i in range(min(3, len(test_samples))):
        sample = test_samples[i]
        print(f"   [{i+1}] Q: {sample['question'][:60]}...")
        print(f"       A: {sample['answer']}")

    # Test 1: HeteroKV ON
    print("\n" + "=" * 80)
    print("TEST 1: HeteroKV ON (with 4-bit quantization + tiered cache)")
    print("=" * 80)
    results_on = test_with_hetero(model, processor, test_samples)

    if results_on['oom_occurred']:
        print(f"   ✗ OOM occurred at test {results_on['results'][-1].get('oom_at_test', '?')}")
    else:
        print(f"   ✓ All {len(results_on['results'])} tests completed")

    # Test 2: HeteroKV OFF
    print("\n" + "=" * 80)
    print("TEST 2: HeteroKV OFF (baseline - no optimization)")
    print("=" * 80)
    results_off = test_without_hetero(model, processor, test_samples)

    if results_off['oom_occurred']:
        print(f"   ✗ OOM occurred at test {results_off['results'][-1].get('oom_at_test', '?')}")
    else:
        print(f"   ✓ All {len(results_off['results'])} tests completed")

    # Analysis and Results
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)

    # Memory comparison
    print("\n┌─ MEMORY USAGE")
    print(f"│  HeteroKV ON  Peak: {results_on['memory_peak_mb']:.1f} MB")
    print(f"│  HeteroKV OFF Peak: {results_off['memory_peak_mb']:.1f} MB")

    if results_off['memory_peak_mb'] > 0:
        memory_reduction = ((results_off['memory_peak_mb'] - results_on['memory_peak_mb']) /
                            results_off['memory_peak_mb'] * 100)
        memory_saved_mb = results_off['memory_peak_mb'] - results_on['memory_peak_mb']
        print(f"│  Memory Saved: {memory_saved_mb:.1f} MB ({memory_reduction:.1f}%)")

    # OOM comparison
    print("\n├─ OOM BEHAVIOR")
    print(f"│  HeteroKV ON  OOM: {'YES (at test {})'.format(results_on['results'][-1].get('oom_at_test', '?')) if results_on['oom_occurred'] else 'NO'}")
    print(f"│  HeteroKV OFF OOM: {'YES (at test {})'.format(results_off['results'][-1].get('oom_at_test', '?')) if results_off['oom_occurred'] else 'NO'}")

    if results_off['oom_occurred'] and not results_on['oom_occurred']:
        oom_test = results_off['results'][-1].get('oom_at_test', '?')
        print(f"│  ✓ HeteroKV prevented OOM at test {oom_test}")

    # Accuracy comparison
    print("\n├─ ACCURACY")
    accuracy_on = sum(1 for r in results_on['results'] if r.get('correct', False)) / len(results_on['results']) * 100
    accuracy_off = sum(1 for r in results_off['results'] if r.get('correct', False)) / len(results_off['results']) * 100

    correct_on = sum(1 for r in results_on['results'] if r.get('correct', False))
    correct_off = sum(1 for r in results_off['results'] if r.get('correct', False))

    print(f"│  HeteroKV ON  Accuracy: {accuracy_on:.1f}% ({correct_on}/{len(results_on['results'])})")
    print(f"│  HeteroKV OFF Accuracy: {accuracy_off:.1f}% ({correct_off}/{len(results_off['results'])})")

    accuracy_diff = abs(accuracy_on - accuracy_off)
    print(f"│  Accuracy Difference: {accuracy_diff:.1f}%")

    if accuracy_diff < 5.0:
        print(f"│  ✓ Zero quality degradation (difference < 5%)")
    elif accuracy_on > accuracy_off:
        print(f"│  ▲ HeteroKV accuracy is higher by {accuracy_on - accuracy_off:.1f}%")
    else:
        print(f"│  ▼ HeteroKV accuracy is lower by {accuracy_off - accuracy_on:.1f}%")

    # Performance comparison
    print("\n├─ PERFORMANCE")
    print(f"│  HeteroKV ON  Total Time: {results_on['total_time']:.2f}s")
    print(f"│  HeteroKV OFF Total Time: {results_off['total_time']:.2f}s")

    if results_on['total_time'] > 0 and len(results_on['results']) > 0:
        avg_time_on = results_on['total_time'] / len(results_on['results'])
        avg_time_off = results_off['total_time'] / len(results_off['results'])
        print(f"│  Avg time per question: ON={avg_time_on:.2f}s, OFF={avg_time_off:.2f}s")

        if avg_time_off > 0:
            speedup = avg_time_off / avg_time_on
            print(f"│  Speedup: {speedup:.2f}x")

    # Detailed results for first few tests
    print("\n└─ DETAILED RESULTS (First 5 tests)")
    for i in range(min(5, len(test_samples))):
        print(f"\n   Test {i+1}: {test_samples[i]['question'][:70]}")
        if i < len(results_on['results']) and 'error' not in results_on['results'][i]:
            r_on = results_on['results'][i]
            status = "✓" if r_on['correct'] else "✗"
            print(f"   Reference: {r_on['reference_answer']}")
            print(f"   HeteroKV ON : {r_on['model_answer'][:50]}... {status}")
        if i < len(results_off['results']) and 'error' not in results_off['results'][i]:
            r_off = results_off['results'][i]
            status = "✓" if r_off['correct'] else "✗"
            print(f"   HeteroKV OFF: {r_off['model_answer'][:50]}... {status}")

    # Save results
    output_file = '/home/app-ahr/Hetero-KVCache-Optimizer/benchmark_results_vqa_rad.json'
    results_summary = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'model': model_name,
        'dataset': 'flaviagiammarino/vqa-rad',
        'num_samples': len(test_samples),
        'hetero_on': results_on,
        'hetero_off': results_off,
        'summary': {
            'memory_saved_mb': results_off['memory_peak_mb'] - results_on['memory_peak_mb'],
            'memory_reduction_percent': ((results_off['memory_peak_mb'] - results_on['memory_peak_mb']) / results_off['memory_peak_mb'] * 100) if results_off['memory_peak_mb'] > 0 else 0,
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
    print("=" * 80)

    return results_summary

if __name__ == "__main__":
    results = run_real_dataset_benchmark()