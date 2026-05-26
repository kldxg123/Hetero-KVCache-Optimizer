#!/usr/bin/env python3
"""
128K Stress Test: Method D vs Method C vs Baseline
===================================================
Progressive context scaling: 4K → 8K → 16K → 32K → 64K → 128K
Tests OOM resistance, accuracy at extreme context lengths.
"""

import torch, time, sys, os, gc, json
from typing import Dict, List

sys.path.insert(0, '/home/app-ahr/Hetero-KVCache-Optimizer/src')
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from transformers import AutoProcessor, LlavaForConditionalGeneration
from core.engine_wrapper import FusedHeteroCache
from core.fused_attention_patch import patch_model_for_fused_attention

CODE = "SECRET2026"

print("=" * 80)
print("  128K Stress Test: Method D vs Method C vs Baseline")
print("=" * 80)

# Load model
print("\n[1/2] Loading model...")
mp = "/home/app-ahr/.cache/huggingface/hub/models--llava-hf--llava-1.5-7b-hf/snapshots"
snaps = sorted([d for d in os.listdir(mp) if os.path.isdir(os.path.join(mp, d))])
mp = os.path.join(mp, snaps[-1])
proc = AutoProcessor.from_pretrained(mp)
model = LlavaForConditionalGeneration.from_pretrained(mp, torch_dtype=torch.float16, device_map="cuda")
model.eval()
print(f"   Model: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

# Build long context
print("\n[2/2] Building long context...")
story_parts = [
    "In the year 2026, a team of AI researchers discovered a mysterious code: " + CODE + ". ",
    "This code was hidden in an ancient manuscript found in a digital library. ",
    "The team, led by Dr. Sarah Chen, embarked on a journey to decipher its meaning. ",
    "They traveled to various locations mentioned in the manuscript's cryptic notes. ",
    "Each location revealed another piece of the puzzle surrounding the secret code. ",
    "Professor Williams, an expert in cryptography, joined their expedition. ",
    "Together, they uncovered connections to historical events spanning centuries. ",
    "The code " + CODE + " appeared throughout history in unexpected places. ",
    "Ancient civilizations had used it as a symbol of knowledge and power. ",
    "Modern technology revealed new layers of meaning in this mysterious sequence. ",
]

story = " ".join(story_parts)
while len(story) < 150000:  # Aim for ~128K tokens
    story = story + " " + story

print(f"   Story length: {len(story)} chars")

# Tokenize
story_ids = proc.tokenizer(story, return_tensors='pt').input_ids[0]
print(f"   Story tokens: {story_ids.shape[0]}")


def build_prompt(ctx_tokens, needle_token_pos):
    """Build prompt with needle at specific position."""
    code_ids = proc.tokenizer(CODE, add_special_tokens=False).input_ids
    ctx_ids = story_ids[:ctx_tokens].clone()

    if needle_token_pos + len(code_ids) < ctx_tokens:
        ctx_ids[needle_token_pos:needle_token_pos + len(code_ids)] = torch.tensor(code_ids)

    question = "\n\nQuestion: What is the secret code mentioned throughout the text? Answer:"
    question_ids = proc.tokenizer(question, add_special_tokens=False).input_ids

    full_ids = torch.cat([ctx_ids, torch.tensor(question_ids)])
    return full_ids


def run_test(method_name, enable_method_d, adaptive_triton, use_patch, ctx, needle_pos):
    """Run a single test."""
    input_ids = build_prompt(ctx, needle_pos)
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


# ── Run Progressive Stress Test ───────────────────────────────────────────────
print("\n[3/3] Running progressive stress test...")
print("=" * 80)

results = []
context_lengths = [4096, 8192, 16384, 32768, 65536, 131072]  # 4K → 128K

for ctx in context_lengths:
    if story_ids.shape[0] < ctx:
        print(f"\n  ⚠️  Skipping ctx={ctx}: not enough tokens")
        continue

    print(f"\n  {'─'*70}")
    print(f"  CONTEXT LENGTH: {ctx//1024}K tokens")
    print(f"  Zones: Sink[0:1024] DRAM[1024:{ctx-2048}] Tail[{ctx-2048}:{ctx}]")

    # Test needle in DRAM zone (the critical case)
    needle_pos = min(ctx // 2, ctx - 3072)  # Middle of context, in DRAM zone
    if needle_pos < 1024:
        needle_pos = 2048  # Ensure it's in DRAM zone

    configs = [
        ("Baseline (No Healing)", False, False, False),
        ("Method C (Triton+Adaptive)", False, True, True),
        ("Method D (Query-aware)", True, False, False),
    ]

    for method_name, enable_d, adaptive, use_patch in configs:
        print(f"\n    Testing: {method_name}")
        r = run_test(method_name, enable_d, adaptive, use_patch, ctx, needle_pos)
        results.append(r)

        icon = "✅" if r['ok'] else "❌" if not r['oom'] else "💥"
        print(f"      {icon} | tokens={r['tokens']:6} | peak={r['peak']:8.0f}MB | "
              f"time={r['time']:5.1f}s | zone={r['zone']:5}")

        if r['oom']:
            print(f"      💥 OOM at {r['peak']:.0f}MB")
            # Continue with other methods at this context length

    if all(r['oom'] for r in results[-len(configets):]):
        print(f"\n  ⚠️  All methods OOM at ctx={ctx}, stopping test")
        break

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*80}")
print("SUMMARY - 128K Stress Test")
print(f"{'='*80}")

print(f"\n  {'Method':25} | {'4K':5} | {'8K':5} | {'16K':5} | {'32K':5} | {'64K':5} | {'128K':5}")
print("  " + "-" * 75)

for method in ["Baseline (No Healing)", "Method C (Triton+Adaptive)", "Method D (Query-aware)"]:
    row = f"  {method:25} |"
    for ctx in [4096, 8192, 16384, 32768, 65536, 131072]:
        method_results = [r for r in results if r['method'] == method and r['ctx'] == ctx]
        if not method_results:
            row += " -    |"
            continue

        r = method_results[0]
        if r['oom']:
            row += " 💥   |"
        elif r['ok']:
            row += " ✅   |"
        else:
            row += " ❌   |"

    print(row)

# Save results
output_file = "/home/app-ahr/Hetero-KVCache-Optimizer/benchmark_128k_method_d_results.json"
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n  Results saved to: {output_file}")

print(f"\n{'='*80}\n")
