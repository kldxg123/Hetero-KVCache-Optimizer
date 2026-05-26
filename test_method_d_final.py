#!/usr/bin/env python3
"""
Method D vs C — Natural Language Only
======================================
Filters WikiText-2 to only natural language paragraphs (no code/markup).
"""

import torch, time, sys, os, gc, re
sys.path.insert(0, '/home/app-ahr/Hetero-KVCache-Optimizer/src')
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from transformers import AutoProcessor, LlavaForConditionalGeneration
from core.engine_wrapper import FusedHeteroCache
from core.fused_attention_patch import patch_model_for_fused_attention

NEEDLE = "HETEROKV2026"

print("=" * 80)
print("  Method D vs C — Natural Language Only")
print("=" * 80)

# Load model
print("\n[1/3] Loading model...")
mp = "/home/app-ahr/.cache/huggingface/hub/models--llava-hf--llava-1.5-7b-hf/snapshots"
snaps = sorted([d for d in os.listdir(mp) if os.path.isdir(os.path.join(mp, d))])
mp = os.path.join(mp, snaps[-1])
proc = AutoProcessor.from_pretrained(mp)
model = LlavaForConditionalGeneration.from_pretrained(mp, torch_dtype=torch.float16, device_map="cuda")
model.eval()

# Load and filter WikiText-2
print("\n[2/3] Loading and filtering WikiText-2...")
from datasets import load_dataset
wiki = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", trust_remote_code=False)

passages = []
for ex in wiki:
    t = ex['text'].strip()
    # Filter out code/markup: must have spaces, lowercase, normal punctuation
    if len(t) > 50 and len(t) < 1000:  # Reasonable paragraph length
        # Check if it's natural language (not code)
        if re.search(r'[a-z]{4,}', t):  # Has lowercase words
            if not re.search(r'[\(\){}\[\]<>&]', t):  # No code brackets
                if not t.startswith('='):  # Not wiki markup
                    passages.append(t)
    if len(passages) >= 1000:
        break

corpus = " ".join(passages)
print(f"   Filtered corpus: {len(corpus)} chars (~{len(corpus)//4} tokens)")
print(f"   Paragraphs: {len(passages)}")

# Pre-tokenize
all_ids = proc.tokenizer(corpus, return_tensors='pt').input_ids[0]
total_tokens_available = all_ids.shape[0]
print(f"   Total tokens available: {total_tokens_available}")


def build_needle_ids(ctx_target, needle_pos):
    """Build token IDs with needle at exact position."""
    needle_str = f"The secret code is {NEEDLE}. Please remember this code."
    needle_ids = proc.tokenizer(needle_str, add_special_tokens=False).input_ids
    needle_len = len(needle_ids)

    ctx_ids = all_ids[:ctx_target].clone()

    if needle_pos + needle_len < ctx_target:
        ctx_ids[needle_pos:needle_pos + needle_len] = torch.tensor(needle_ids)

    question = "\n\nQuestion: What is the secret code? Answer:"
    question_ids = proc.tokenizer(question, add_special_tokens=False).input_ids

    full_ids = torch.cat([ctx_ids, torch.tensor(question_ids)])
    return full_ids, needle_len


def run_test(method_name, enable_method_d, use_patch, ctx, needle_pos):
    """Run a single test."""
    input_ids, needle_len = build_needle_ids(ctx, needle_pos)
    n_tokens = input_ids.shape[0]

    sink, tail = 1024, 2048
    if needle_pos < sink:
        zone = "Sink"
    elif needle_pos >= ctx - tail:
        zone = "Tail"
    else:
        zone = "DRAM"

    print(f"\n  Testing: {method_name}")
    print(f"    Context: {n_tokens} tokens, Needle at {needle_pos} ({zone} zone)")

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    gc.collect()

    input_ids_dev = input_ids.unsqueeze(0).to('cuda')
    attention_mask = torch.ones(1, n_tokens, dtype=torch.long).to('cuda')

    cache = FusedHeteroCache(
        num_layers=32, sink_tokens=sink, keep_tail=tail, chunk_size=2048,
        device='cuda', enable_quant=True, enable_prefetch=True, enable_triton=True,
        self_healing=True, adaptive_self_healing=True,
        enable_method_d=enable_method_d,
        method_d_alpha=1.0,
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
        ok = NEEDLE in raw.upper()

        result = dict(
            method=method_name, ctx=ctx, needle_pos=needle_pos, zone=zone,
            tokens=n_tokens, peak=peak, time=elapsed, ok=ok,
            ans=raw.strip()[:80], oom=False,
        )

        icon = "✅" if ok else "❌"
        print(f"    {icon} {method_name:20} | {zone:5} | {peak:7.0f}MB | {elapsed:4.1f}s | {raw.strip()[:60]}")
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

# Test configs
test_configs = [
    (6144, 3072, "DRAM"),   # 6K, needle in DRAM
    (6144, 500, "Sink"),    # 6K, needle in Sink (baseline - should work)
]

for ctx, needle_pos, expected_zone in test_configs:
    print(f"\n{'─'*70}")
    print(f"  TEST: ctx={ctx}, needle_pos={needle_pos}, zone={expected_zone}")
    print(f"  Zones: Sink[0:1024] DRAM[1024:{ctx-2048}] Tail[{ctx-2048}:{ctx}]")

    if total_tokens_available < ctx + 100:
        print(f"  ⚠️  Skipping: not enough tokens")
        continue

    # Method C
    r_c = run_test("Method C (Triton+Patch)", False, True, ctx, needle_pos)
    results.append(r_c)

    # Method D
    r_d = run_test("Method D (Query-aware)", True, False, ctx, needle_pos)
    results.append(r_d)

    if r_c['oom'] and r_d['oom']:
        break

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}")

c_results = [r for r in results if 'Method C' in r['method']]
d_results = [r for r in results if 'Method D' in r['method']]

c_ok = sum(1 for r in c_results if r['ok'])
d_ok = sum(1 for r in d_results if r['ok'])

print(f"\n  Method C (Triton+Patch): {c_ok}/{len(c_results)}")
print(f"  Method D (Query-aware):  {d_ok}/{len(d_results)}")

for zone in ['Sink', 'DRAM']:
    cz = [r for r in c_results if r.get('zone') == zone]
    dz = [r for r in d_results if r.get('zone') == zone]
    if cz or dz:
        c_z_ok = sum(1 for r in cz if r['ok'])
        d_z_ok = sum(1 for r in dz if r['ok'])
        print(f"  {zone}: C={c_z_ok}/{len(cz)} | D={d_z_ok}/{len(dz)}")

print(f"\n{'='*80}\n")
