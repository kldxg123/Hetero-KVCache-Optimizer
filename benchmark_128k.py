#!/usr/bin/env python3
"""
HeteroKV 128K Context Stress Test
Progressive context scaling: 2K → 4K → 8K → 16K → 32K → 64K → 128K tokens
Monitors: peak memory, total memory, accuracy, OOM behavior
"""

import torch
import gc
import json
import time
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
from datasets import load_dataset
from typing import Dict, List, Optional
import os
import sys
import traceback

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


def answer_matches(prediction: str, reference: str) -> bool:
    """Check if predicted answer matches reference"""
    pred = prediction.lower().strip()
    ref = reference.lower().strip()
    if ref in pred:
        return True
    if ref in ('yes', 'no'):
        return any(ref == w for w in pred.split()[:5])
    # Check key words overlap
    ref_words = set(ref.split())
    pred_words = set(pred.split())
    overlap = ref_words & pred_words
    return len(overlap) >= len(ref_words) * 0.5


def build_context_text(dataset, target_tokens: int) -> str:
    """Build context text by repeating Q&A pairs to reach target token count"""
    context_parts = []
    estimated_tokens = 0
    idx = 0

    while estimated_tokens < target_tokens:
        sample_idx = idx % len(dataset)
        sample = dataset[sample_idx]
        qa_text = f"Q: {sample['question']} A: {sample['answer']}\n"
        context_parts.append(qa_text)
        estimated_tokens += len(qa_text) // 3  # rough token estimate
        idx += 1

        # Safety: don't loop forever
        if idx > target_tokens * 2:
            break

    return "".join(context_parts)


def run_progressive_test_hetero(
    model,
    processor,
    image: Image.Image,
    dataset,
    context_targets: List[int],
    test_questions: List[Dict],
    max_new_tokens: int = 30
) -> Dict:
    """Test HeteroKV with progressively longer contexts"""
    results = {
        'config': 'HeteroKV ON',
        'total_gpu_mb': get_total_gpu_memory(),
        'tests': []
    }

    for target_tokens in context_targets:
        print(f"\n  [{target_tokens} tokens] Building context...")
        test_result = {
            'target_tokens': target_tokens,
            'oom': False,
            'error': None,
            'memory': {},
            'accuracy': [],
            'generation_time': 0,
        }

        # Reset memory tracking
        torch.cuda.reset_peak_memory_stats()
        gc.collect()
        torch.cuda.empty_cache()

        # Measure baseline (model only)
        baseline = get_gpu_stats()

        try:
            # Build long context
            context_text = build_context_text(dataset, target_tokens - 100)  # leave room for question

            # Create HeteroKV cache
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

            # Test with multiple questions
            for qi, q in enumerate(test_questions):
                prompt = f"USER: <image>\n{context_text}\nQuestion: {q['question']}\nAnswer briefly. ASSISTANT:"
                inputs = processor(text=prompt, images=image, return_tensors='pt').to('cuda')

                test_result['actual_input_tokens'] = inputs.input_ids.shape[-1]

                # Generate
                start_time = time.time()
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        past_key_values=cache
                    )
                gen_time = time.time() - start_time
                test_result['generation_time'] += gen_time

                # Decode answer
                response = processor.decode(outputs[0], skip_special_tokens=True)
                answer = response.split("ASSISTANT:")[-1].strip()

                # Check accuracy
                is_correct = answer_matches(answer, q['answer'])
                test_result['accuracy'].append({
                    'question': q['question'],
                    'expected': q['answer'],
                    'actual': answer[:80],
                    'correct': is_correct
                })

                # Cleanup outputs
                del inputs, outputs
                gc.collect()

            # Record memory stats
            post_test = get_gpu_stats()
            test_result['memory'] = {
                'baseline_allocated_mb': baseline['allocated_mb'],
                'peak_allocated_mb': post_test['peak_allocated_mb'],
                'peak_reserved_mb': post_test['peak_reserved_mb'],
                'post_allocated_mb': post_test['allocated_mb'],
                'gpu_utilization_pct': post_test['peak_allocated_mb'] / results['total_gpu_mb'] * 100,
            }

            test_result['kv_seq_length'] = cache.get_seq_length()

            del cache
            gc.collect()
            torch.cuda.empty_cache()

        except RuntimeError as e:
            if "out of memory" in str(e):
                test_result['oom'] = True
                test_result['memory']['peak_allocated_mb'] = get_gpu_stats()['peak_allocated_mb']
                test_result['memory']['gpu_utilization_pct'] = (
                    test_result['memory']['peak_allocated_mb'] / results['total_gpu_mb'] * 100
                )
            else:
                test_result['error'] = str(e)
                traceback.print_exc()
            gc.collect()
            torch.cuda.empty_cache()

        except Exception as e:
            test_result['error'] = str(e)
            traceback.print_exc()
            gc.collect()
            torch.cuda.empty_cache()

        results['tests'].append(test_result)

        # Print progress
        status = "OOM!" if test_result['oom'] else ("OK" if not test_result['error'] else "ERR")
        if test_result.get('actual_input_tokens'):
            print(f"  [{target_tokens} tokens] {status} | "
                  f"Input: {test_result['actual_input_tokens']} | "
                  f"Peak: {test_result['memory'].get('peak_allocated_mb', 0):.0f}MB "
                  f"({test_result['memory'].get('gpu_utilization_pct', 0):.1f}%) | "
                  f"Acc: {sum(1 for a in test_result['accuracy'] if a['correct'])}/{len(test_result['accuracy'])}")
        else:
            print(f"  [{target_tokens} tokens] {status} | {test_result.get('error', 'OOM')[:60]}")

    return results


def run_progressive_test_baseline(
    model,
    processor,
    image: Image.Image,
    dataset,
    context_targets: List[int],
    test_questions: List[Dict],
    max_new_tokens: int = 30
) -> Dict:
    """Test baseline (no HeteroKV) with progressively longer contexts"""
    results = {
        'config': 'HeteroKV OFF',
        'total_gpu_mb': get_total_gpu_memory(),
        'tests': []
    }

    for target_tokens in context_targets:
        print(f"\n  [{target_tokens} tokens] Building context...")
        test_result = {
            'target_tokens': target_tokens,
            'oom': False,
            'error': None,
            'memory': {},
            'accuracy': [],
            'generation_time': 0,
        }

        torch.cuda.reset_peak_memory_stats()
        gc.collect()
        torch.cuda.empty_cache()

        baseline = get_gpu_stats()

        try:
            context_text = build_context_text(dataset, target_tokens - 100)

            for qi, q in enumerate(test_questions):
                prompt = f"USER: <image>\n{context_text}\nQuestion: {q['question']}\nAnswer briefly. ASSISTANT:"
                inputs = processor(text=prompt, images=image, return_tensors='pt').to('cuda')

                test_result['actual_input_tokens'] = inputs.input_ids.shape[-1]

                start_time = time.time()
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        past_key_values=None
                    )
                gen_time = time.time() - start_time
                test_result['generation_time'] += gen_time

                response = processor.decode(outputs[0], skip_special_tokens=True)
                answer = response.split("ASSISTANT:")[-1].strip()

                is_correct = answer_matches(answer, q['answer'])
                test_result['accuracy'].append({
                    'question': q['question'],
                    'expected': q['answer'],
                    'actual': answer[:80],
                    'correct': is_correct
                })

                del inputs, outputs
                gc.collect()

            post_test = get_gpu_stats()
            test_result['memory'] = {
                'baseline_allocated_mb': baseline['allocated_mb'],
                'peak_allocated_mb': post_test['peak_allocated_mb'],
                'peak_reserved_mb': post_test['peak_reserved_mb'],
                'post_allocated_mb': post_test['allocated_mb'],
                'gpu_utilization_pct': post_test['peak_allocated_mb'] / results['total_gpu_mb'] * 100,
            }

            gc.collect()
            torch.cuda.empty_cache()

        except RuntimeError as e:
            if "out of memory" in str(e):
                test_result['oom'] = True
                test_result['memory']['peak_allocated_mb'] = get_gpu_stats()['peak_allocated_mb']
                test_result['memory']['gpu_utilization_pct'] = (
                    test_result['memory']['peak_allocated_mb'] / results['total_gpu_mb'] * 100
                )
            else:
                test_result['error'] = str(e)
                traceback.print_exc()
            gc.collect()
            torch.cuda.empty_cache()

        except Exception as e:
            test_result['error'] = str(e)
            traceback.print_exc()
            gc.collect()
            torch.cuda.empty_cache()

        results['tests'].append(test_result)

        status = "OOM!" if test_result['oom'] else ("OK" if not test_result['error'] else "ERR")
        if test_result.get('actual_input_tokens'):
            print(f"  [{target_tokens} tokens] {status} | "
                  f"Input: {test_result['actual_input_tokens']} | "
                  f"Peak: {test_result['memory'].get('peak_allocated_mb', 0):.0f}MB "
                  f"({test_result['memory'].get('gpu_utilization_pct', 0):.1f}%) | "
                  f"Acc: {sum(1 for a in test_result['accuracy'] if a['correct'])}/{len(test_result['accuracy'])}")
        else:
            print(f"  [{target_tokens} tokens] {status} | {test_result.get('error', 'OOM')[:60]}")

    return results


def run_128k_benchmark():
    """Main benchmark: progressive scaling up to 128K tokens"""

    print("=" * 80)
    print("HeteroKV 128K Context Stress Test")
    print("Progressive scaling: 2K → 4K → 8K → 16K → 32K → 64K → 128K")
    print("=" * 80)

    total_gpu = get_total_gpu_memory()
    print(f"\nGPU Total Memory: {total_gpu:.0f} MB ({total_gpu/1024:.1f} GB)")
    print(f"Model: LLaVA-1.5-7B (~13GB)")
    print(f"Available for KV Cache: ~{total_gpu - 13000:.0f} MB")
    print(f"Standard KV Cache at 128K tokens: ~{128000 * 512 / 1024:.0f} MB >> GPU capacity")

    # Load dataset
    print("\n[1/5] Loading VQA-RAD dataset...")
    dataset = load_dataset('flaviagiammarino/vqa-rad', split='test')
    print(f"   Loaded {len(dataset)} samples")

    # Load model
    model_name = "llava-hf/llava-1.5-7b-hf"
    print(f"\n[2/5] Loading {model_name}...")
    processor = AutoProcessor.from_pretrained(model_name)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cuda"
    )
    model.eval()
    print("   Model loaded")

    # Prepare image
    image = dataset[0]['image']
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Test questions (real from dataset)
    test_questions = [
        {'question': dataset[i]['question'], 'answer': dataset[i]['answer']}
        for i in range(3)
    ]
    print(f"\n[3/5] Test questions:")
    for i, q in enumerate(test_questions):
        print(f"   Q{i+1}: {q['question'][:60]}... (Expected: {q['answer']})")

    # Context targets: progressive scaling
    context_targets = [2048, 4096, 8192, 16384, 32768, 65536, 131072]

    print(f"\n[4/5] Running progressive context scaling test")
    print(f"   Targets: {context_targets}")
    print(f"   Note: Baseline (OFF) will stop at first OOM")

    # ---- TEST 1: HeteroKV ON ----
    print("\n" + "=" * 80)
    print("TEST 1: HeteroKV ON (4-bit quant + tiered cache + self-healing)")
    print("=" * 80)

    results_on = run_progressive_test_hetero(
        model, processor, image, dataset,
        context_targets, test_questions
    )

    # ---- TEST 2: Baseline OFF ----
    # Determine max context that ON survived
    max_on_ctx = 0
    for t in results_on['tests']:
        if not t['oom'] and not t.get('error'):
            max_on_ctx = t['target_tokens']

    # For baseline, test same range but stop after OOM
    print("\n" + "=" * 80)
    print("TEST 2: HeteroKV OFF (baseline - standard cache)")
    print(f"   Will stop at first OOM or {max_on_ctx} tokens")
    print("=" * 80)

    baseline_targets = []
    for t in context_targets:
        baseline_targets.append(t)
        if t > max_on_ctx:
            break

    results_off = run_progressive_test_baseline(
        model, processor, image, dataset,
        baseline_targets, test_questions
    )

    # ---- ANALYSIS ----
    print("\n" + "=" * 80)
    print("128K STRESS TEST RESULTS")
    print("=" * 80)

    # Summary table
    print(f"\n{'─'*80}")
    print(f"{'Target':<10} {'Actual':<10} {'Status':<8} "
          f"{'Peak(MB)':<12} {'GPU%':<8} {'Acc':<6} {'Time(s)':<8}")
    print(f"{'─'*80}")

    print("\nHeteroKV ON:")
    for t in results_on['tests']:
        target = t['target_tokens']
        actual = t.get('actual_input_tokens', 0)
        status = "OOM" if t['oom'] else ("ERR" if t.get('error') else "OK")
        peak = t['memory'].get('peak_allocated_mb', 0)
        gpu_pct = t['memory'].get('gpu_utilization_pct', 0)
        acc = f"{sum(1 for a in t['accuracy'] if a['correct'])}/{len(t['accuracy'])}" if t['accuracy'] else "N/A"
        time_s = f"{t['generation_time']:.2f}"
        print(f"  {target:<8} {actual:<8} {status:<8} {peak:<12.1f} {gpu_pct:<8.1f} {acc:<6} {time_s:<8}")

    print("\nHeteroKV OFF:")
    for t in results_off['tests']:
        target = t['target_tokens']
        actual = t.get('actual_input_tokens', 0)
        status = "OOM" if t['oom'] else ("ERR" if t.get('error') else "OK")
        peak = t['memory'].get('peak_allocated_mb', 0)
        gpu_pct = t['memory'].get('gpu_utilization_pct', 0)
        acc = f"{sum(1 for a in t['accuracy'] if a['correct'])}/{len(t['accuracy'])}" if t['accuracy'] else "N/A"
        time_s = f"{t['generation_time']:.2f}"
        print(f"  {target:<8} {actual:<8} {status:<8} {peak:<12.1f} {gpu_pct:<8.1f} {acc:<6} {time_s:<8}")

    # Key findings
    print(f"\n{'='*80}")
    print("KEY FINDINGS")
    print(f"{'='*80}")

    # Find where baseline OOMs
    baseline_oom_at = None
    for t in results_off['tests']:
        if t['oom']:
            baseline_oom_at = t['target_tokens']
            break

    # Find max context HeteroKV survived
    hetero_max = 0
    for t in results_on['tests']:
        if not t['oom'] and not t.get('error'):
            hetero_max = t['target_tokens']

    print(f"\n  Baseline OOM at: {baseline_oom_at or 'N/A'} tokens")
    print(f"  HeteroKV max:    {hetero_max} tokens")
    if baseline_oom_at and hetero_max > baseline_oom_at:
        print(f"  Context extension: {hetero_max / baseline_oom_at:.1f}x")

    # Memory behavior analysis
    print("\n  Memory growth (HeteroKV ON):")
    prev_peak = 0
    for t in results_on['tests']:
        if not t['oom'] and not t.get('error'):
            peak = t['memory'].get('peak_allocated_mb', 0)
            growth = peak - prev_peak if prev_peak > 0 else 0
            actual = t.get('actual_input_tokens', 0)
            print(f"    {actual:>7} tokens → {peak:.0f} MB (Δ = {growth:+.0f} MB)")
            prev_peak = peak

    # Accuracy analysis
    print("\n  Accuracy comparison:")
    for t_on, t_off in zip(results_on['tests'], results_off['tests']):
        if t_on['accuracy'] and t_off['accuracy']:
            acc_on = sum(1 for a in t_on['accuracy'] if a['correct']) / len(t_on['accuracy']) * 100
            acc_off = sum(1 for a in t_off['accuracy'] if a['correct']) / len(t_off['accuracy']) * 100
            ctx = t_on['target_tokens']
            diff = acc_on - acc_off
            print(f"    {ctx:>7} tokens: ON={acc_on:.0f}% OFF={acc_off:.0f}% (Δ={diff:+.0f}%)")

    # Save full results
    output_file = '/home/app-ahr/Hetero-KVCache-Optimizer/benchmark_128k_results.json'
    full_results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'model': model_name,
        'dataset': 'flaviagiammarino/vqa-rad',
        'gpu_total_mb': total_gpu,
        'context_targets': context_targets,
        'hetero_on': results_on,
        'hetero_off': results_off,
    }

    with open(output_file, 'w') as f:
        json.dump(full_results, f, indent=2)

    print(f"\nFull results saved to: {output_file}")
    print("=" * 80)

    return full_results


if __name__ == "__main__":
    results = run_128k_benchmark()