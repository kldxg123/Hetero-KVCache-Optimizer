#!/usr/bin/env python3
"""
Real Dataset Long Text Test: WikiText-2 + Needle
==================================================
Uses WikiText-2 natural text with needle at multiple positions.
Tests Sink/DRAM/Tail zones for Method C, Method D, and baseline.
"""

import torch, time, sys, os, gc, json, re
from typing import Dict, List

sys.path.insert(0, '/home/app-ahr/Hetero-KVCache-Optimizer/src')
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from transformers import AutoProcessor, LlavaForConditionalGeneration
from core.engine_wrapper import FusedHeteroCache
from core.fused_attention_patch import patch_model_for_fused_attention
from datasets import load_dataset

CODE = "HETEROKV2026"

print("=" * 80)
print("  Real Dataset Test: WikiText-2 Long Text")
print("=" * 80)

# Load model
print("\n[1/3] Loading model...")
mp = "/home/app-ahr/.cache/huggingface/hub/models--llava-hf--llava-1.5-7b-hf/snapshots"
snaps = sorted([d for d in os.listdir(mp) if os.path.isdir(os.path.join(mp, d))])
mp = os.path.join(mp, snaps[-1])
proc = AutoProcessor.from_pretrained(mp)
model = LlavaForConditionalGeneration.from_pretrained(mp, torch_dtype=torch.float16, device_map="cuda")
model.eval()
print(f"   Model: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

# Load WikiText-2
print("\n[2/3] Loading WikiText-2...")
wiki = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", trust_remote_code=False)

# Filter for natural language paragraphs
passages = []
for ex in wiki:
    t = ex['text'].strip()
    if len(t) > 100 and len(t) < 1000:
        if re.search(r'[a-z]{4,}', t):  # Has lowercase words
            if not re.search(r'[\(\){}\[\]<>&]', t):  # No code brackets
                if not t.startswith('='):  # Not wiki markup
                    passages.append(t)
    if len(passages) >= 2000:
        break

corpus = " ".join(passages)
print(f"   Corpus: {len(corpus)} chars (~{len(corpus)//4} tokens)")
print(f"   Paragraphs: {len(passages)}")

# Pre-tokenize
all_ids = proc.tokenizer(corpus, return_tensors='pt').input_ids[0]
total_tokens = all_ids.shape[0]
print(f"   Total tokens available: {total_tokens}")


def build_needle_ids(ctx_target, needle_pos):
    """Build token IDs with needle at exact position."""
    needle_str = f"The secret code is {CODE}. Remember this code."
    needle_ids = proc.tokenizer(needle_str, add_special_tokens=False).input_ids
    needle_len = len(needle_ids)

    ctx_ids = all_ids[:ctx_target].clone()

    if needle_pos + needle_len < ctx_target:
        ctx_ids[needle_pos:needle_pos + needle_len] = torch.tensor(needle_ids)

    question = "\n\nQuestion: What is the secret code mentioned in the text? Answer:"
    question_ids = proc.tokenizer(question, add_special_tokens=False).input_ids

    full_ids = torch.cat([ctx_ids, torch.tensor(question_ids)])
    return full_ids


def run_test(method_name, enable_method_d, adaptive_triton, use_patch, ctx, needle_pos):
    """Run a single test."""
    input_ids = build_needle_ids(ctx, needle_pos)
    n_tokens = input_ids.shape[0]

    sink, tail = 1024, 2048
    if needle_pos < sink:
        zone = "Sink"
    elif needle_pos >= ctx - tail:
        zone = "Tail"
    else:
        zone = "DRAM"

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    gc.collect()

    input_ids_dev = input_ids.unsqueeze(0).to('cuda')
    attention_mask = torch.ones(1, n_tokens, dtype=torch.long).to('cuda')

    cache = FusedHeteroCache(
        num_layers=32, sink_tokens=sink, keep_tail=tail, chunk_size=2048,
        device='cuda', enable_quant=True, enable_prefetch=False,
        self_healing=enable_method_d or adaptive_triton,
        adaptive_self_healing=adaptive_triton,
        enable_triton=adaptive_triton,
        enable_method_d=enable_method_d,
    )

    t0 = time.time()
    try:
        if use_patch:
            with patch_model_for_fused_attention(model, cache, enable_fused=True):
                out = model.generate(
                    input_ids=input_ids_dev,
                    attention_mask=attention_mask,
                    max_new_tokens=30,
                    do_sample=False,
                    past_key_values=cache,
                )
        else:
            out = model.generate(
                input_ids=input_ids_dev,
                attention_mask=attention_mask,
                max_new_tokens=30,
                do_sample=False,
                past_key_values=cache,
            )

        elapsed = time.time() - t0
        peak = torch.cuda.max_memory_allocated() / 1024**2

        generated = out[0][n_tokens:]
        raw = proc.tokenizer.decode(generated, skip_special_tokens=True)
        ok = CODE in raw

        result = dict(
            method=method_name, ctx=ctx, needle_pos=needle_pos, zone=zone,
            tokens=n_tokens, peak=peak, time=elapsed, ok=ok,
            ans=raw.strip()[:80], oom=False,
        )

        return result

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            peak = torch.cuda.max_memory_allocated() / 1024**2
            return dict(
                method=method_name, ctx=ctx, needle_pos=needle_pos, zone=zone,
                tokens=n_tokens, peak=peak, time=0, ok=False, ans="OOM", oom=True,
            )
        raise
    finally:
        del input_ids_dev, attention_mask, cache
        gc.collect()
        torch.cuda.empty_cache()


# ── Run Tests ─────────────────────────────────────────────────────────────────
print("\n[3/3] Running tests...")
print("=" * 80)

results = []

# Test configs: (ctx, needle_pos, zone, description)
test_configs = [
    # Long context, needle in DRAM (critical case)
    (16384, 8192, "DRAM", "16K ctx, needle in middle DRAM"),
    (16384, 500, "Sink", "16K ctx, needle in Sink"),
    (16384, 15000, "Tail", "16K ctx, needle in Tail"),

    # Medium context, needle in DRAM
    (8192, 4096, "DRAM", "8K ctx, needle in middle DRAM"),
    (8192, 500, "Sink", "8K ctx, needle in Sink"),
    (8192, 7000, "Tail", "8K ctx, needle in Tail"),

    # Short context, needle in DRAM
    (6144, 3072, "DRAM", "6K ctx, needle in DRAM"),
]

for ctx, needle_pos, expected_zone, desc in test_configs:
    if total_tokens < ctx:
        print(f"\n  ⚠️  Skipping {desc}: not enough tokens")
        continue

    print(f"\n  {'─'*70}")
    print(f"  TEST: {desc}")
    print(f"    ctx={ctx}, needle_pos={needle_pos}, zone={expected_zone}")

    configs = [
        ("Baseline", False, False, False),
        ("Method C (Triton+Adaptive)", False, True, True),
        ("Method D (Query-aware)", True, False, False),
    ]

    for method_name, enable_d, adaptive, use_patch in configs:
        r = run_test(method_name, enable_d, adaptive, use_patch, ctx, needle_pos)
        results.append(r)

        icon = "✅" if r['ok'] else "❌" if not r['oom'] else "💥"
        print(f"    {icon} {method_name:25} | zone={r['zone']:5} | "
              f"tok={r['tokens']:5} | peak={r['peak']:7.0f}MB | time={r['time']:4.1f}s")

        if r['oom']:
            print(f"      💥 OOM at {r['peak']:.0f}MB")

# ── Analysis ───────────────────────────────────────────────────────────────────
print(f"\n{'='*80}")
print("SUMMARY - Real Dataset Test")
print(f"{'='*80}")

# Overall accuracy
print(f"\n  Overall Accuracy:")
for method in ["Baseline", "Method C (Triton+Adaptive)", "Method D (Query-aware)"]:
    method_results = [r for r in results if r['method'] == method and not r['oom']]
    if method_results:
        ok_count = sum(1 for r in method_results if r['ok'])
        total = len(method_results)
        print(f"    {method:25} | {ok_count}/{total} ({ok_count/total*100:.0f}%)")

# Per-zone accuracy
print(f"\n  Per-Zone Accuracy:")
print(f"  {'Zone':<8} {'Baseline':<12} {'Method C':<12} {'Method D':<12}")
print("  " + "-" * 50)

for zone in ['Sink', 'DRAM', 'Tail']:
    baseline = [r for r in results if r['method'] == 'Baseline' and r.get('zone') == zone and not r['oom']]
    method_c = [r for r in results if 'Method C' in r['method'] and r.get('zone') == zone and not r['oom']]
    method_d = [r for r in results if 'Method D' in r['method'] and r.get('zone') == zone and not r['oom']]

    b_acc = f"{sum(1 for r in baseline if r['ok'])}/{len(baseline)}" if baseline else "N/A"
    c_acc = f"{sum(1 for r in method_c if r['ok'])}/{len(method_c)}" if method_c else "N/A"
    d_acc = f"{sum(1 for r in method_d if r['ok'])}/{len(method_d)}" if method_d else "N/A"

    print(f"  {zone:<8} {b_acc:<12} {c_acc:<12} {d_acc:<12}")

# Memory and latency
print(f"\n  Memory & Latency:")
for method in ["Baseline", "Method C (Triton+Adaptive)", "Method D (Query-aware)"]:
    method_results = [r for r in results if r['method'] == method and not r['oom']]
    if method_results:
        avg_peak = sum(r['peak'] for r in method_results) / len(method_results)
        avg_time = sum(r['time'] for r in method_results) / len(method_results)
        print(f"    {method:25} | peak={avg_peak:7.0f}MB | time={avg_time:4.1f}s")

# Save results
output_file = "/home/app-ahr/Hetero-KVCache-Optimizer/test_real_dataset_results.json"
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n  Results saved to: {output_file}")

print(f"\n{'='*80}\n")
