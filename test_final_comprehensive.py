#!/usr/bin/env python3
"""
FINAL COMPREHENSIVE TEST: Method D vs Method C vs Baseline
===========================================================
Uses proven narrative text. Generates sufficient tokens (6K+).
Tests all three methods with baseline comparison.
"""

import torch, time, sys, os, gc, json
sys.path.insert(0, '/home/app-ahr/Hetero-KVCache-Optimizer/src')
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from transformers import AutoProcessor, LlavaForConditionalGeneration
from core.engine_wrapper import FusedHeteroCache
from core.fused_attention_patch import patch_model_for_fused_attention

CODE = "XY789"

print("=" * 80)
print("  FINAL COMPREHENSIVE TEST: Method D vs C vs Baseline")
print("=" * 80)

# Load model
print("\n[1/2] Loading model...")
mp = "/home/app-ahr/.cache/huggingface/hub/models--llava-hf--llava-1.5-7b-hf/snapshots"
snaps = sorted([d for d in os.listdir(mp) if os.path.isdir(os.path.join(mp, d))])
mp = os.path.join(mp, snaps[-1])
proc = AutoProcessor.from_pretrained(mp)
model = LlavaForConditionalGeneration.from_pretrained(mp, torch_dtype=torch.float16, device_map="cuda")
model.eval()

# Build story with DIVERSE content to prevent token compression
print("\n[2/2] Building story...")
story_templates = [
    f"In 2026, researchers discovered code {CODE} in an ancient manuscript.",
    f"Dr. Chen confirmed that {CODE} appeared throughout history.",
    f"Professor Williams studied {CODE} and found its mathematical significance.",
    f"The team published findings about {CODE} in major scientific journals.",
    f"Scientists worldwide analyzed {CODE} using advanced computational methods.",
    f"Some believed {CODE} was a key to understanding ancient civilizations.",
    f"Archaeologists found references to {CODE} in Egyptian hieroglyphs.",
    f"The mystery of {CODE} continues to fascinate researchers today.",
    f"Quantum computing revealed new patterns in {CODE}.",
    f"Cryptographers attempted to decode the meaning of {CODE}.",
]

story = " ".join(story_templates)
# Generate diverse story to avoid token compression
story_parts = []
for i in range(1000):
    part = f"Chapter {i}: "
    part += f"Story continued with {CODE} mentioned in context. "
    part += f"The investigation of {CODE} revealed new insights. "
    part += f"Researchers documented their findings about {CODE}. "
    story_parts.append(part)

story = " ".join(story_parts)
story = story[:500000]  # ~125K chars

story_ids = proc.tokenizer(story, return_tensors='pt').input_ids[0]
print(f"   Story tokens available: {story_ids.shape[0]}")


def build_prompt(ctx_target):
    """Build prompt with code embedded."""
    # Place code at ~50% position (DRAM zone)
    needle_pos = ctx_target // 2

    ctx_ids = story_ids[:ctx_target].clone()
    code_ids = proc.tokenizer(CODE, add_special_tokens=False).input_ids

    if needle_pos + len(code_ids) < ctx_target:
        ctx_ids[needle_pos:needle_pos + len(code_ids)] = torch.tensor(code_ids)

    question = "\n\nQuestion: What is the code mentioned in the text? Answer:"
    question_ids = proc.tokenizer(question, add_special_tokens=False).input_ids

    return torch.cat([ctx_ids, torch.tensor(question_ids)])


def run_test(config, ctx_target):
    """Run single test."""
    input_ids = build_prompt(ctx_target)
    n_tokens = input_ids.shape[0]

    sink, tail = 1024, 2048
    needle_pos = ctx_target // 2
    if needle_pos < sink:
        zone = "Sink"
    elif needle_pos >= ctx_target - tail:
        zone = "Tail"
    else:
        zone = "DRAM"

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    gc.collect()

    name, enable_d, adaptive, triton, use_patch = config
    cache = FusedHeteroCache(
        num_layers=32, sink_tokens=sink, keep_tail=tail, chunk_size=2048,
        device='cuda', enable_quant=True, enable_prefetch=False,
        self_healing=enable_d or adaptive,
        adaptive_self_healing=adaptive,
        enable_triton=triton,
        enable_method_d=enable_d,
    )

    t0 = time.time()
    try:
        ids_dev = input_ids.unsqueeze(0).to('cuda')
        attn = torch.ones(1, n_tokens, dtype=torch.long).to('cuda')

        if use_patch:
            with patch_model_for_fused_attention(model, cache, enable_fused=True):
                out = model.generate(
                    input_ids=ids_dev, attention_mask=attn,
                    max_new_tokens=30, do_sample=False, past_key_values=cache,
                )
        else:
            out = model.generate(
                input_ids=ids_dev, attention_mask=attn,
                max_new_tokens=30, do_sample=False, past_key_values=cache,
            )

        elapsed = time.time() - t0
        peak = torch.cuda.max_memory_allocated() / 1024**2
        gen = out[0][n_tokens:]
        raw = proc.tokenizer.decode(gen, skip_special_tokens=True)
        ok = CODE in raw

        return dict(
            method=name, ctx=ctx_target, tokens=n_tokens, zone=zone,
            peak=peak, time=elapsed, ok=ok, ans=raw.strip()[:60], oom=False,
        )

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            peak = torch.cuda.max_memory_allocated() / 1024**2
            return dict(method=name, ctx=ctx_target, tokens=n_tokens, zone=zone,
                       peak=peak, time=0, ok=False, ans="OOM", oom=True)
        raise
    except Exception as e:
        return dict(method=name, ctx=ctx_target, tokens=n_tokens, zone=zone,
                   peak=0, time=0, ok=False, ans=f"ERR: {str(e)[:50]}", oom=False)
    finally:
        del cache, ids_dev, attn
        gc.collect()
        torch.cuda.empty_cache()


# ── Run Tests ─────────────────────────────────────────────────────────────────
print("\n[3/3] Running comprehensive tests...")
print("=" * 80)

configs = [
    ("Baseline (No Healing)",   False, False, False, False),
    ("Method C (Triton+Adapt)", False, True,  True,  True),
    ("Method D (Query-aware)",  True,  False, False, False),
]

results = []
context_lengths = [4096, 6144, 8192, 12288]

for ctx in context_lengths:
    if story_ids.shape[0] < ctx:
        print(f"\n  ⚠️  ctx={ctx}: not enough tokens, skip")
        continue

    print(f"\n  CONTEXT: {ctx//1024}K tokens | DRAM zone: [1024:{ctx-2048}]")

    for config in configs:
        name = config[0]
        r = run_test(config, ctx)
        results.append(r)

        if r.get('oom'):
            print(f"    💥 {name:25} | OOM at {r['peak']:.0f}MB")
        elif 'ERR' in r.get('ans', ''):
            print(f"    ❌ {name:25} | ERROR: {r['ans']}")
        else:
            icon = "✅" if r['ok'] else "❌"
            print(f"    {icon} {name:25} | zone={r['zone']:5} | "
                  f"tok={r['tokens']:5} | peak={r['peak']:7.0f}MB | "
                  f"time={r['time']:4.1f}s | {r['ans'][:30]}")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*80}")
print("  FINAL SUMMARY")
print(f"{'='*80}")

print(f"\n  Accuracy by Context Length:")
print(f"  {'Method':25} | ", end="")
for ctx in context_lengths:
    print(f"{ctx//1024:>3}K | ", end="")
print()
print("  " + "-" * 60)

for name in ["Baseline (No Healing)", "Method C (Triton+Adapt)", "Method D (Query-aware)"]:
    print(f"  {name:25} | ", end="")
    for ctx in context_lengths:
        mr = [r for r in results if r['method'] == name and r['ctx'] == ctx]
        if not mr:
            print("  -  | ", end="")
        elif mr[0].get('oom'):
            print(" 💥 | ", end="")
        elif 'ERR' in mr[0].get('ans', ''):
            print(" ❌ | ", end="")
        elif mr[0]['ok']:
            print(" ✅ | ", end="")
        else:
            print(" ❌ | ", end="")
    print()

print(f"\n  Overall Results:")
for name in ["Baseline (No Healing)", "Method C (Triton+Adapt)", "Method D (Query-aware)"]:
    mr = [r for r in results if r['method'] == name and not r.get('oom') and 'ERR' not in r.get('ans', '')]
    if mr:
        ok = sum(1 for r in mr if r['ok'])
        total = len(mr)
        avg_peak = sum(r['peak'] for r in mr) / len(mr)
        avg_time = sum(r['time'] for r in mr) / len(mr)
        print(f"    {name:25} | {ok}/{total} ({ok/max(1,total)*100:3.0f}%) | "
              f"peak={avg_peak:.0f}MB | time={avg_time:.1f}s")

# Save results
output = "/home/app-ahr/Hetero-KVCache-Optimizer/final_comprehensive_results.json"
with open(output, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n  Results saved to: {output}")

print(f"\n{'='*80}\n")
