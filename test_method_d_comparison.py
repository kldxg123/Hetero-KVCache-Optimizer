#!/usr/bin/env python3
"""
Method D Comparison Test
=========================

Compare Method C (Historical Attention) vs Method D (Query-Aware Retrieval)
on the same workload to measure accuracy and performance differences.

Test Design:
1. Same workload (WikiText-2 + NIAH)
2. Toggle between Method C and Method D
3. Measure:
   - Accuracy (correct answers)
   - GPU memory usage
   - Latency per token
   - KV cache retrieval statistics
"""

import torch, time, sys, os, gc
import numpy as np
from PIL import Image

sys.path.insert(0, '/home/app-ahr/Hetero-KVCache-Optimizer/src')
from transformers import AutoProcessor, LlavaForConditionalGeneration
from core.engine_wrapper import FusedHeteroCache

MAX_MEM = 24 * 1024

print("=" * 80)
print("  Method D vs Method C Comparison Test")
print("=" * 80)

# ── Load Model ────────────────────────────────────────────────────────────────────
print("\n[1/4] Loading LLaVA-1.5-7B...")
mp = "/home/app-ahr/.cache/huggingface/hub/models--llava-hf--llava-1.5-7b-hf/snapshots"
snaps = sorted([d for d in os.listdir(mp) if os.path.isdir(os.path.join(mp, d))])
mp = os.path.join(mp, snaps[-1])
proc = AutoProcessor.from_pretrained(mp)
model = LlavaForConditionalGeneration.from_pretrained(mp, torch_dtype=torch.float16, device_map="cuda")
model.eval()
print(f"   Model: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

# ── Test Data ─────────────────────────────────────────────────────────────────────
# Load WikiText-2
from datasets import load_dataset
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
wikitext_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test", trust_remote_code=False)

long_text = ""
for example in wikitext_dataset:
    text = example['text']
    if len(text) > 500:
        long_text = text
        break

print(f"   Found WikiText-2 with {len(long_text)} characters")

# NIAH test configuration
NEEDLE = "HETEROKV2026"

def build_niah_prompt(ctx_len, needle_pos):
    """Build NIAH prompt with needle at specified position."""
    filler = "The quick brown fox jumps over the lazy dog. "
    base = 600
    before_chars = int(max(0, (needle_pos - base)) * 12 / 1)
    total_filler_chars = int(max(0, (ctx_len - base - 30)) * 12 / 1)
    after_chars = max(0, total_filler_chars - before_chars)

    before = filler * (before_chars // len(filler))
    after = filler * (after_chars // len(filler))

    return f"USER: <image>\n{before} The secret passcode is {NEEDLE}. Remember it. {after}\nWhat is the secret passcode?\nASSISTANT:"

def run_test(method_name, enable_method_d, ctx_lengths, needle_positions):
    """Run test with specified method configuration."""
    print(f"\n{'='*80}")
    print(f"  Testing: {method_name}")
    print(f"{'='*80}")

    results = []

    for ctx, needle_pos in zip(ctx_lengths, needle_positions):
        print(f"\n  Context: {ctx} tokens, Needle at position: {needle_pos}")

        # Build prompt
        prompt = build_niah_prompt(ctx, needle_pos)
        inputs = proc(text=prompt, images=Image.new('RGB', (224, 224), (255, 255, 255)),
                     return_tensors='pt').to('cuda')
        n_tokens = inputs.input_ids.shape[-1]

        # Determine needle zone
        if needle_pos < 1024:
            zone = "Sink"
        elif needle_pos > ctx - 2048:
            zone = "Tail"
        else:
            zone = "DRAM (evicted)"

        # Create cache with specified configuration
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

        cache = FusedHeteroCache(
            num_layers=32, sink_tokens=1024, keep_tail=2048, chunk_size=2048,
            device='cuda', enable_quant=True, enable_prefetch=True, enable_triton=True,
            self_healing=True, adaptive_self_healing=True,
            enable_method_d=enable_method_d,  # Toggle Method D
            method_d_alpha=1.0,  # Pure query-aware
        )

        # Run generation
        t0 = time.time()
        try:
            with torch.no_grad():
                out = model.generate(
                    input_ids=inputs.input_ids,
                    pixel_values=inputs.pixel_values,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=30,
                    do_sample=False,
                    past_key_values=cache,
                )
            elapsed = time.time() - t0
            peak = torch.cuda.max_memory_allocated() / 1024**2
            ans = proc.decode(out[0], skip_special_tokens=True).split("ASSISTANT:")[-1].strip()

            ok = NEEDLE in ans.upper()
            kv_info = cache.memory_summary()

            result = {
                'method': method_name,
                'ctx': ctx,
                'needle_pos': needle_pos,
                'zone': zone,
                'tokens': n_tokens,
                'peak': peak,
                'time': elapsed,
                'ok': ok,
                'ans': ans[:80],
                'kv_mb': kv_info.get('hbm_tokens', 0) * 32 * 2 * 32 * 128 * 2 / 1024**2,
                'hbm_tokens': kv_info.get('hbm_tokens', 0),
                'dram_entries': kv_info.get('dram_entries', 0),
            }

            status = "✅" if ok else "❌"
            print(f"    {status} Zone={zone:12} | Peak={peak:8.0f}MB | Time={elapsed:5.1f}s | "
                  f"KV={kv_info.get('hbm_tokens', 0):5} tok | Ans: {ans[:50]}...")

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                peak = torch.cuda.max_memory_allocated() / 1024**2
                result = {
                    'method': method_name,
                    'ctx': ctx,
                    'needle_pos': needle_pos,
                    'zone': zone,
                    'tokens': n_tokens,
                    'peak': peak,
                    'time': 0,
                    'ok': False,
                    'ans': "OOM",
                    'kv_mb': 0,
                    'hbm_tokens': 0,
                    'dram_entries': 0,
                }
                print(f"    ❌ OOM at {peak:.0f}MB")
            else:
                raise
        finally:
            del inputs, cache
            torch.cuda.empty_cache()

        results.append(result)

    return results

# ── Run Tests ─────────────────────────────────────────────────────────────────────

print("\n[2/4] Test Method C (Historical Attention)")
print("-" * 80)

# Test different context lengths with needle in different zones
ctx_lengths = [4096, 8192, 16384]
needle_positions = [500, 4000, 8000]  # Sink, DRAM, DRAM

results_c = run_test(
    "Method C (Historical)",
    enable_method_d=False,
    ctx_lengths=ctx_lengths,
    needle_positions=needle_positions,
)

print("\n[3/4] Test Method D (Query-Aware Retrieval)")
print("-" * 80)

results_d = run_test(
    "Method D (Query-Aware)",
    enable_method_d=True,
    ctx_lengths=ctx_lengths,
    needle_positions=needle_positions,
)

# ── Analysis ───────────────────────────────────────────────────────────────────────

print("\n[4/4] Comparison Analysis")
print("=" * 80)

# Group results by zone
zones = ['Sink', 'DRAM (evicted)', 'Tail']

print(f"\n{'Zone':<15} {'Method C OK':<12} {'Method D OK':<12} {'C Peak MB':<12} {'D Peak MB':<12} {'C Time':<10} {'D Time':<10}")
print("-" * 85)

for zone in zones:
    c_zone = [r for r in results_c if r['zone'] == zone]
    d_zone = [r for r in results_d if r['zone'] == zone]

    if not c_zone or not d_zone:
        continue

    c_ok = sum(1 for r in c_zone if r['ok']) / len(c_zone) * 100
    d_ok = sum(1 for r in d_zone if r['ok']) / len(d_zone) * 100
    c_peak = np.mean([r['peak'] for r in c_zone])
    d_peak = np.mean([r['peak'] for r in d_zone])
    c_time = np.mean([r['time'] for r in c_zone])
    d_time = np.mean([r['time'] for r in d_zone])

    c_status = "✅" if c_ok >= 80 else "⚠️" if c_ok >= 50 else "❌"
    d_status = "✅" if d_ok >= 80 else "⚠️" if d_ok >= 50 else "❌"

    print(f"{zone:<15} {c_status} {c_ok:.0f}%/{len(c_zone)}   {d_status} {d_ok:.0f}%/{len(d_zone)}   "
          f"{c_peak:<10.0f} {d_peak:<10.0f} {c_time:<8.1f}s {d_time:<8.1f}s")

# Overall statistics
print("\n" + "=" * 80)
print("OVERALL STATISTICS")
print("=" * 80)

c_total_ok = sum(1 for r in results_c if r['ok'])
d_total_ok = sum(1 for r in results_d if r['ok'])
total = len(results_c)

print(f"\nMethod C (Historical Attention):")
print(f"  Accuracy: {c_total_ok}/{total} ({c_total_ok/total*100:.0f}%)")
print(f"  Avg Peak: {np.mean([r['peak'] for r in results_c]):.0f}MB")
print(f"  Avg Time: {np.mean([r['time'] for r in results_c]):.1f}s")

print(f"\nMethod D (Query-Aware Retrieval):")
print(f"  Accuracy: {d_total_ok}/{total} ({d_total_ok/total*100:.0f}%)")
print(f"  Avg Peak: {np.mean([r['peak'] for r in results_d]):.0f}MB")
print(f"  Avg Time: {np.mean([r['time'] for r in results_d]):.1f}s")

# Improvement
acc_improvement = (d_total_ok - c_total_ok) / total * 100
time_diff = (np.mean([r['time'] for r in results_d]) - np.mean([r['time'] for r in results_c]))
mem_diff = (np.mean([r['peak'] for r in results_d]) - np.mean([r['peak'] for r in results_c]))

print(f"\n{'='*80}")
print("CONCLUSION")
print(f"{'='*80}")

if acc_improvement > 10:
    print(f"  ✅ Method D significantly improves accuracy: +{acc_improvement:.0f}%")
elif acc_improvement > 0:
    print(f"  ⚠️  Method D modestly improves accuracy: +{acc_improvement:.0f}%")
elif acc_improvement < -10:
    print(f"  ❌ Method D significantly harms accuracy: {acc_improvement:.0f}%")
else:
    print(f"  ➡️  Method D has similar accuracy to Method C")

if abs(time_diff) < 0.5:
    print(f"  ✅ Latency impact minimal: {time_diff:+.1f}s")
elif time_diff > 0:
    print(f"  ⚠️  Latency increased: +{time_diff:.1f}s")
else:
    print(f"  ✅ Latency improved: {time_diff:.1f}s")

if abs(mem_diff) < 500:
    print(f"  ✅ Memory impact minimal: {mem_diff:+.0f}MB")
elif mem_diff > 0:
    print(f"  ⚠️  Memory increased: +{mem_diff:.0f}MB")
else:
    print(f"  ✅ Memory reduced: {mem_diff:.0f}MB")

print(f"\n{'='*80}\n")
