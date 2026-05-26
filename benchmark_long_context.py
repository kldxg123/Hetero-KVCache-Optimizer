#!/usr/bin/env python3
"""
HeteroKV Benchmark - Single-Prompt Long Context Test
Uses real VQA-RAD dataset to build increasingly long prompts
Compares memory, accuracy, and OOM behavior
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


def get_gpu_memory():
    """Get current GPU memory usage in MB"""
    torch.cuda.synchronize()
    return torch.cuda.memory_allocated() / 1024**2


def answer_matches(prediction: str, reference: str) -> bool:
    """Check if predicted answer matches reference"""
    pred = prediction.lower().strip()
    ref = reference.lower().strip()
    if pred == ref or ref in pred:
        return True
    # For yes/no questions
    if ref in ('yes', 'no'):
        return ref in pred.split()[:3]  # Check first few words
    return False


def build_long_prompt(dataset, num_samples: int) -> tuple:
    """Build a long prompt by concatenating multiple Q&A pairs"""
    conversation = "USER: <image>\n"

    for i in range(num_samples):
        sample = dataset[i]
        question = sample['question']
        answer = sample['answer']
        conversation += f"Q{i+1}: {question}\nA{i+1}: {answer}\n"

    conversation += "ASSISTANT:"
    return conversation


def build_final_question(dataset, question_idx: int) -> str:
    """Build the final question to test accuracy on"""
    sample = dataset[question_idx]
    return sample['question'], sample['answer']


def run_single_test(
    model,
    processor,
    image: Image.Image,
    prompt: str,
    final_question: str,
    use_hetero: bool,
    max_new_tokens: int = 30
) -> Dict:
    """Run a single test with given context length"""
    result = {
        'use_hetero': use_hetero,
        'memory_peak_mb': 0,
        'oom_occurred': False,
        'answer': '',
        'generation_time': 0,
        'input_tokens': 0,
        'output_tokens': 0,
    }

    torch.cuda.reset_peak_memory_stats()
    gc.collect()
    torch.cuda.empty_cache()

    try:
        # Create cache if using HeteroKV
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

        full_prompt = prompt + f"Based on the above Q&A history, answer this: {final_question}\nAnswer:"

        inputs = processor(text=full_prompt, images=image, return_tensors='pt').to('cuda')
        result['input_tokens'] = inputs.input_ids.shape[-1]

        mem_before = get_gpu_memory()

        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                past_key_values=cache
            )
        generation_time = time.time() - start_time

        mem_after = get_gpu_memory()
        result['memory_peak_mb'] = mem_after
        result['generation_time'] = generation_time

        # Decode
        response = processor.decode(outputs[0], skip_special_tokens=True)
        answer = response.split("Answer:")[-1].strip() if "Answer:" in response else response.split("ASSISTANT:")[-1].strip()
        result['answer'] = answer
        result['output_tokens'] = outputs.shape[-1] - inputs.input_ids.shape[-1]

        if cache:
            result['kv_seq_length'] = cache.get_seq_length()
            del cache

    except RuntimeError as e:
        if "out of memory" in str(e):
            result['oom_occurred'] = True
        else:
            raise e

    gc.collect()
    torch.cuda.empty_cache()

    return result


def run_long_context_benchmark():
    """Run comprehensive long context benchmark"""

    print("=" * 80)
    print("HeteroKV Long Context Benchmark - Real VQA Dataset")
    print("Dataset: flaviagiammarino/vqa-rad (451 medical VQA samples)")
    print("=" * 80)

    # Load dataset
    print("\n[1/4] Loading VQA-RAD dataset...")
    dataset = load_dataset('flaviagiammarino/vqa-rad', split='test')
    print(f"   Loaded {len(dataset)} samples")

    # Load model
    model_name = "llava-hf/llava-1.5-7b-hf"
    print(f"\n[2/4] Loading model: {model_name}")

    processor = AutoProcessor.from_pretrained(model_name)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cuda"
    )
    model.eval()
    print("   Model loaded")

    # Get first image for testing
    sample = dataset[0]
    image = sample['image']
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Define context lengths to test
    context_lengths = [2, 5, 10, 15, 20, 30]
    final_q_idx = 0  # Use first question as final test question
    final_question, final_answer = build_final_question(dataset, final_q_idx)

    print(f"\n[3/4] Running benchmark with context lengths: {context_lengths}")
    print(f"   Final question: {final_question}")
    print(f"   Expected answer: {final_answer}")

    results = {
        'metadata': {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'model': model_name,
            'dataset': 'flaviagiammarino/vqa-rad',
            'final_question': final_question,
            'expected_answer': final_answer,
        },
        'tests': []
    }

    for num_ctx in context_lengths:
        print(f"\n{'─' * 70}")
        print(f"Context length: {num_ctx} Q&A pairs")

        # Build the long prompt
        prompt = build_long_prompt(dataset, num_ctx)
        # Replace image placeholder for actual processing
        prompt_no_img = prompt.replace("<image>", "")

        print(f"  Prompt tokens (approx): ~{len(prompt) // 4}")

        # Test with HeteroKV ON
        print(f"  Testing HeteroKV ON...")
        result_on = run_single_test(
            model, processor, image, prompt, final_question,
            use_hetero=True
        )

        # Test with HeteroKV OFF
        print(f"  Testing HeteroKV OFF...")
        result_off = run_single_test(
            model, processor, image, prompt, final_question,
            use_hetero=False
        )

        test_result = {
            'context_pairs': num_ctx,
            'hetero_on': result_on,
            'hetero_off': result_off,
        }
        results['tests'].append(test_result)

        # Print comparison for this context length
        print(f"  ┌─ Memory: ON={result_on['memory_peak_mb']:.0f}MB, OFF={result_off['memory_peak_mb']:.0f}MB")
        print(f"  ├─ Time:   ON={result_on['generation_time']:.2f}s, OFF={result_off['generation_time']:.2f}s")
        print(f"  ├─ OOM:    ON={'YES' if result_on['oom_occurred'] else 'NO'}, OFF={'YES' if result_off['oom_occurred'] else 'NO'}")
        if result_on['input_tokens'] > 0:
            print(f"  ├─ Tokens: {result_on['input_tokens']} input")

        # Show answers comparison
        if not result_on['oom_occurred'] and not result_off['oom_occurred']:
            print(f"  ├─ Answer ON : {result_on['answer'][:50]}")
            print(f"  └─ Answer OFF: {result_off['answer'][:50]}")

    # Final Summary
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)

    print("\n┌─ MEMORY COMPARISON")
    print(f"│  {'Context':<10} {'ON (MB)':<12} {'OFF (MB)':<12} {'Diff (MB)':<12} {'OOM?'}")
    print(f"│  {'─'*56}")
    for test in results['tests']:
        n = test['context_pairs']
        on_mem = test['hetero_on']['memory_peak_mb']
        off_mem = test['hetero_off']['memory_peak_mb']
        diff = off_mem - on_mem
        oom = ""
        if test['hetero_off']['oom_occurred']:
            oom = "OFF OOM!"
        elif test['hetero_on']['oom_occurred']:
            oom = "ON OOM!"
        print(f"│  {n:<10} {on_mem:<12.1f} {off_mem:<12.1f} {diff:<+12.1f} {oom}")

    print("\n├─ ACCURACY COMPARISON")
    for test in results['tests']:
        n = test['context_pairs']
        on_ans = test['hetero_on']['answer']
        off_ans = test['hetero_off']['answer']
        on_match = answer_matches(on_ans, final_answer) if not test['hetero_on']['oom_occurred'] else None
        off_match = answer_matches(off_ans, final_answer) if not test['hetero_off']['oom_occurred'] else None

        on_status = "✓" if on_match else "✗" if on_match is not None else "OOM"
        off_status = "✓" if off_match else "✗" if off_match is not None else "OOM"

        match_status = "SAME" if on_status == off_status else "DIFF"

        print(f"│  ctx={n:<4} ON:{on_status} OFF:{off_status} [{match_status}]")

    print("\n└─ PERFORMANCE")
    for test in results['tests']:
        n = test['context_pairs']
        on_time = test['hetero_on']['generation_time']
        off_time = test['hetero_off']['generation_time']
        speedup = off_time / on_time if on_time > 0 else 0
        print(f"   ctx={n:<4} ON:{on_time:.2f}s OFF:{off_time:.2f}s speedup={speedup:.2f}x")

    # Save results
    output_file = '/home/app-ahr/Hetero-KVCache-Optimizer/benchmark_long_context_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")
    print("=" * 80)

    return results


if __name__ == "__main__":
    results = run_long_context_benchmark()