#!/usr/bin/env python3
"""
Method D vs Method C Comparison Test v2
========================================
Uses WikiText-2 natural text + embedded needle for fair comparison.
Tests DRAM zone retrieval accuracy (the critical case where Method D should help).
"""

import torch, time, sys, os, gc
import numpy as np

sys.path.insert(0, '/home/app-ahr/Hetero-KVCache-Optimizer/src')
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from transformers import AutoProcessor, LlavaForConditionalGeneration
from core.engine_wrapper import FusedHeteroCache

NEEDLE = "HETEROKV2026"

print("=" * 80)
print("  Method D vs Method C Comparison (Natural Text)")
print("=" * 80)

# Load model
print("\n[1/4] Loading model...")
mp = "/home/app-ahr/.cache/huggingface/hub/models--llava-hf--llava-1.5-7b-hf/snapshots"
snaps = sorted([d for d in os.listdir(mp) if os.path.isdir(os.path.join(mp, d))])
mp = os.path.join(mp, snaps[-1])
proc = AutoProcessor.from_pretrained(mp)
model = LlavaForConditionalGeneration.from_pretrained(mp, torch_dtype=torch.float16, device_map="cuda")
model.eval()
print(f"   Model: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

# Load WikiText-2 natural text
print("\n[2/4] Loading WikiText-2...")
from datasets import load_dataset
wiki = load_dataset("wikitext", "wikitext-2-raw-v1", split="test", trust_remote_code=False)

# Collect natural text passages
passages = []
for ex in wiki:
    t = ex['text'].strip()
    if len(t) > 100:
        passages.append(t)
    if len(passages) >= 100:
        break
print(f"   Loaded {len(passages)} passages")

def build_prompt_with_needle(ctx_tokens_target, needle_pos):
    """Build prompt using natural text with needle embedded at specified position."""
    needle_str = f"The secret passcode is {NEEDLE}. Remember this code. "

    # Approximate: 1 token ≈ 4 chars for natural text
    chars_per_token = 4
    total_chars = ctx_tokens_target * chars_per_token

    # Build before and after text from natural passages
    text = ""
    for p in passages:
        if len(text) >= total_chars:
            break
        text += p + " "

    # Find needle position in chars
    needle_char_pos = needle_pos * chars_per_token

    if needle_char_pos < len(text) and needle_char_pos > 0:
        before = text[:needle_char_pos]
        after = text[needle_char_pos:]
        full_text = before + needle_str + after
    else:
        full_text = text + needle_str

    prompt = f"{full_text[:total_chars]}\n\nQuestion: What is the secret passcode mentioned in the text?\nAnswer: The secret passcode is"

    return prompt

def run_single_test(method_name, enable_method_d, ctx, needle_pos):
    """Run a single test."""
    prompt = build_prompt_with_needle(ctx, needle_pos)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    inputs = proc(text=prompt, return_tensors='pt').to('cuda')
    n_tokens = inputs.input_ids.shape[-1]

    # Determine zone
    if needle_pos < 1024:
        zone = "Sink"
    elif needle_pos > ctx - 2048:
        zone = "Tail"
    else:
        zone = "DRAM"

    cache = FusedHeteroCache(
        num_layers=32, sink_tokens=1024, keep_tail=2048, chunk_size=2048,
        device='cuda', enable_quant=True, enable_prefetch=True, enable_triton=True,
        self_healing=True, adaptive_self_healing=True,
        enable_method_d=enable_method_d,
        method_d_alpha=1.0,
    )

    t0 = time.time()
    try:
        with torch.no_grad():
            out = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=20,
                do_sample=False,
                past_key_values=cache,
            )
        elapsed = time.time() - t0
        peak = torch.cuda.max_memory_allocated() / 1024**2
        raw = proc.decode(out[0], skip_special_tokens=True)
        ans = raw.split("Answer: The secret passcode is")[-1].strip() if "Answer: The secret passcode is" in raw else raw[-100:]
        ok = NEEDLE in ans.upper()

        result = dict(
            method=method_name, ctx=ctx, needle_pos=needle_pos, zone=zone,
            tokens=n_tokens, peak=peak, time=elapsed, ok=ok,
            ans=ans[:60], oom=False,
        )
        icon = "✅" if ok else "❌"
        print(f"    {icon} {method_name:15} | {zone:5} | ctx={ctx:5} | tok={n_tokens:5} | "
              f"{peak:8.0f}MB | {elapsed:5.1f}s | {ans[:40]}")
        return result

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            peak = torch.cuda.max_memory_allocated() / 1024**2
            result = dict(
                method=method_name, ctx=ctx, needle_pos=needle_pos, zone=zone,
                tokens=n_tokens, peak=peak, time=0, ok=False, ans="OOM", oom=True,
            )
            print(f"    ❌ {method_name:15} | {zone:5} | ctx={ctx:5} | OOM at {peak:.0f}MB")
            return result
        raise
    finally:
        del inputs, cache
        torch.cuda.empty_cache()

# ── Run Tests ─────────────────────────────────────────────────────────────────────

# Test DRAM zone specifically (where Method D should help)
print("\n[3/4] Running comparison tests...")
print("-" * 80)

results_c = []
results_d = []

test_configs = [
    # (ctx, needle_pos, zone_description)
    (4096, 2048, "DRAM"),      # 4K context, needle in middle (DRAM zone)
    (8192, 4096, "DRAM"),      # 8K context, needle in middle
    (16384, 8192, "DRAM"),     # 16K context, needle in middle
    (4096, 100, "Sink"),       # 4K context, needle in sink zone
    (4096, 3800, "Tail"),      # 4K context, needle in tail zone
]

for ctx, needle_pos, expected_zone in test_configs:
    print(f"\n  ctx={ctx}, needle_pos={needle_pos}, expected_zone={expected_zone}")

    # Method C
    r_c = run_single_test("Method C", False, ctx, needle_pos)
    results_c.append(r_c)

    # Method D
    r_d = run_single_test("Method D", True, ctx, needle_pos)
    results_d.append(r_d)

    if r_c['oom'] and r_d['oom']:
        break

# ── Analysis ───────────────────────────────────────────────────────────────────────

print("\n[4/4] Comparison Analysis")
print("=" * 80)

# Overall
c_ok = sum(1 for r in results_c if r['ok'])
d_ok = sum(1 for r in results_d if r['ok'])
total = len(results_c)

print(f"\n  Method C (Historical Attention):")
print(f"    Accuracy: {c_ok}/{total} ({c_ok/total*100:.0f}%)")
valid_c = [r for r in results_c if not r['oom']]
if valid_c:
    print(f"    Avg Peak: {np.mean([r['peak'] for r in valid_c]):.0f}MB")
    print(f"    Avg Time: {np.mean([r['time'] for r in valid_c]):.1f}s")

print(f"\n  Method D (Query-Aware):")
print(f"    Accuracy: {d_ok}/{total} ({d_ok/total*100:.0f}%)")
valid_d = [r for r in results_d if not r['oom']]
if valid_d:
    print(f"    Avg Peak: {np.mean([r['peak'] for r in valid_d]):.0f}MB")
    print(f"    Avg Time: {np.mean([r['time'] for r in valid_d]):.1f}s")

# Per-zone comparison
print(f"\n  Per-Zone Comparison:")
print(f"  {'Zone':<8} {'C OK':<8} {'D OK':<8} {'C Peak':<10} {'D Peak':<10}")
print("-" * 50)

for zone in ['Sink', 'DRAM', 'Tail']:
    cz = [r for r in results_c if r['zone'] == zone]
    dz = [r for r in results_d if r['zone'] == zone]
    if cz and dz:
        c_zone_ok = sum(1 for r in cz if r['ok'])
        d_zone_ok = sum(1 for r in dz if r['ok'])
        c_avg_peak = np.mean([r['peak'] for r in cz])
        d_avg_peak = np.mean([r['peak'] for r in dz])
        print(f"  {zone:<8} {c_zone_ok}/{len(cz)}     {d_zone_ok}/{len(dz)}     "
              f"{c_avg_peak:<10.0f} {d_avg_peak:<10.0f}")

# Final verdict
print(f"\n{'='*80}")
print("CONCLUSION")
print(f"{'='*80}")

acc_diff = (d_ok - c_ok) / total * 100 if total > 0 else 0

if acc_diff > 15:
    print(f"  ✅ Method D significantly improves accuracy: +{acc_diff:.0f}%")
elif acc_diff > 0:
    print(f"  ⚠️  Method D modestly improves accuracy: +{acc_diff:.0f}%")
elif acc_diff < -15:
    print(f"  ❌ Method D significantly harms accuracy: {acc_diff:.0f}%")
else:
    print(f"  ➡️  Method D has similar accuracy to Method C ({acc_diff:+.0f}%)")

if valid_c and valid_d:
    time_diff = np.mean([r['time'] for r in valid_d]) - np.mean([r['time'] for r in valid_c])
    mem_diff = np.mean([r['peak'] for r in valid_d]) - np.mean([r['peak'] for r in valid_c])

    if abs(time_diff) < 2:
        print(f"  ✅ Latency impact minimal: {time_diff:+.1f}s")
    else:
        print(f"  ⚠️  Latency: {time_diff:+.1f}s")

    if abs(mem_diff) < 500:
        print(f"  ✅ Memory impact minimal: {mem_diff:+.0f}MB")
    else:
        print(f"  ⚠️  Memory: {mem_diff:+.0f}MB")

print(f"\n{'='*80}\n")
