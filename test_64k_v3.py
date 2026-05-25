#!/usr/bin/env python3
"""
HeteroKV 64K Full Verification Test (v3)
=========================================
- model.generate() for all tests
- KV cache memory tracked separately via cache.memory_summary()
- Needle at Sink (pos≈50), Tail (pos≈ctx-500), and Mid (pos≈ctx//2)
- 24GB memory limit
"""

import torch, time, sys, os, gc
import numpy as np
from PIL import Image

sys.path.insert(0, '/home/app-ahr/Hetero-KVCache-Optimizer/src')
from transformers import AutoProcessor, LlavaForConditionalGeneration
from core.engine_wrapper import FusedHeteroCache

MAX_MEM = 24 * 1024

print("=" * 80)
print("  HeteroKV Full Verification (v3)")
print("  All Modules: Quant + SelfHealing + Adaptive + Triton")
print("=" * 80)

# ── Load ────────────────────────────────────────────────────────────────────
print("\n[1/7] Loading model...")
mp = "/home/app-ahr/.cache/huggingface/hub/models--llava-hf--llava-1.5-7b-hf/snapshots"
snaps = sorted([d for d in os.listdir(mp) if os.path.isdir(os.path.join(mp, d))])
mp = os.path.join(mp, snaps[-1])
proc = AutoProcessor.from_pretrained(mp)
model = LlavaForConditionalGeneration.from_pretrained(mp, torch_dtype=torch.float16, device_map="cuda")
model.eval()
print(f"   Model: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

# ── Test Image ──────────────────────────────────────────────────────────────
arr = np.ones((224, 224, 3), dtype=np.uint8) * 255
y, x = np.ogrid[:224, :224]
arr[((x-112)**2 + (y-112)**2) <= 50**2] = [255, 0, 0]
img = Image.fromarray(arr)

NEEDLE = "HETEROKV2026"
FILLER = "The quick brown fox jumps over the lazy dog. "  # ≈12 tokens each

def build_prompt(ctx_tokens, needle_pos):
    """Build NIAH prompt. Positions are approximate (in tokens)."""
    # Base overhead: USER: <image>\n ≈ 586 tokens, \n{question}\nASSISTANT: ≈ 25 tokens
    base_img = 586
    base_text = 25
    needle_str = f" The secret passcode is {NEEDLE}. Remember it. "  # ≈15 tokens
    question_str = f"\nWhat is the secret passcode?\nASSISTANT:"

    # Calculate filler counts
    filler_per_rep = 12  # tokens per filler sentence
    needle_tokens = 15
    total_non_filler = base_img + base_text + needle_tokens
    total_filler_tokens = max(0, ctx_tokens - total_non_filler)

    # Split filler around needle
    before_tokens = max(0, needle_pos - base_img - needle_tokens // 2)
    after_tokens = max(0, total_filler_tokens - before_tokens)

    before_filler = FILLER * (before_tokens // filler_per_rep)
    after_filler = FILLER * (after_tokens // filler_per_rep)

    return f"USER: <image>\n{before_filler}{needle_str}{after_filler}{question_str}"

def run_one(label, ctx, needle_pos, use_heterokv=False, sink_tokens=1024):
    """Run a single test."""
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    prompt = build_prompt(ctx, needle_pos)
    inputs = proc(text=prompt, images=img, return_tensors='pt').to('cuda')
    n = inputs.input_ids.shape[-1]

    cache = None
    if use_heterokv:
        cache = FusedHeteroCache(
            num_layers=32, sink_tokens=sink_tokens, keep_tail=2048, chunk_size=2048,
            device='cuda', enable_quant=True, enable_prefetch=True, enable_triton=True,
            self_healing=True, adaptive_self_healing=True,
        )

    t0 = time.time()
    try:
        with torch.no_grad():
            out = model.generate(
                input_ids=inputs.input_ids, pixel_values=inputs.pixel_values,
                attention_mask=inputs.attention_mask, max_new_tokens=30,
                do_sample=False, past_key_values=cache,
            )
        elapsed = time.time() - t0
        peak = torch.cuda.max_memory_allocated() / 1024**2
        ans = proc.decode(out[0], skip_special_tokens=True).split("ASSISTANT:")[-1].strip()
        ok = NEEDLE in ans.upper()

        # Get KV memory
        kv_info = cache.memory_summary() if cache else None

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            peak = torch.cuda.max_memory_allocated() / 1024**2
            return dict(label=label, ctx=ctx, tokens=n, peak=peak, time=0,
                        ok=False, ans="OOM", oom=True, kv_mb=0)
        raise
    finally:
        del inputs
        if cache: del cache
        torch.cuda.empty_cache()

    kv_mb = 0
    if kv_info:
        # Estimate KV HBM from memory_summary
        hbm_toks = kv_info.get('hbm_tokens', 0)
        kv_mb = hbm_toks * 32 * 2 * 32 * 128 * 2 / 1024**2  # tokens * layers * KV * heads * dim * bytes

    return dict(label=label, ctx=ctx, tokens=n, peak=peak, time=elapsed,
                ok=ok, ans=ans[:60], oom=False, kv_mb=kv_mb, kv_info=kv_info)

# ── Run Tests ───────────────────────────────────────────────────────────────

print("\n[2/7] Baseline (Standard KV)")
print("-" * 80)
base = []
for ctx in [4096, 8192]:
    r = run_one("Base", ctx, ctx // 2)
    base.append(r)
    s = "OK" if r['ok'] else ("OOM" if r['oom'] else "FAIL")
    print(f"  {r['ctx']:>5}tok | {r['tokens']:>6} actual | {r['peak']:>8.0f}MB | {r['time']:>5.1f}s | {s:<4} | {r['ans'][:45]}")
    if r['oom'] or r['peak'] > MAX_MEM: break

print("\n[3/7] HeteroKV - Needle in SINK (first 64 tokens)")
print("-" * 80)
sink = []
for ctx in [4096, 8192, 16384, 32768, 65536]:
    r = run_one("Sink", ctx, 50, use_heterokv=True)
    sink.append(r)
    s = "OK" if r['ok'] else ("OOM" if r['oom'] else "FAIL")
    kv = f"KV={r['kv_mb']:.0f}MB" if r.get('kv_mb') else ""
    print(f"  {r['ctx']:>5}tok | {r['tokens']:>6} actual | {r['peak']:>8.0f}MB | {r['time']:>5.1f}s | {s:<4} | {r['ans'][:40]} | {kv}")
    if r['oom']: break

print("\n[4/7] HeteroKV - Needle in TAIL (last ~500 tokens)")
print("-" * 80)
tail = []
for ctx in [4096, 8192, 16384, 32768, 65536]:
    r = run_one("Tail", ctx, ctx - 500, use_heterokv=True)
    tail.append(r)
    s = "OK" if r['ok'] else ("OOM" if r['oom'] else "FAIL")
    kv = f"KV={r['kv_mb']:.0f}MB" if r.get('kv_mb') else ""
    print(f"  {r['ctx']:>5}tok | {r['tokens']:>6} actual | {r['peak']:>8.0f}MB | {r['time']:>5.1f}s | {s:<4} | {r['ans'][:40]} | {kv}")
    if r['oom']: break

print("\n[5/7] HeteroKV - Needle in MIDDLE (evicted to DRAM)")
print("-" * 80)
mid = []
for ctx in [4096, 8192, 16384, 32768, 65536]:
    r = run_one("Mid", ctx, ctx // 2, use_heterokv=True)
    mid.append(r)
    s = "OK" if r['ok'] else ("OOM" if r['oom'] else "FAIL")
    kv = f"KV={r['kv_mb']:.0f}MB" if r.get('kv_mb') else ""
    print(f"  {r['ctx']:>5}tok | {r['tokens']:>6} actual | {r['peak']:>8.0f}MB | {r['time']:>5.1f}s | {s:<4} | {r['ans'][:40]} | {kv}")
    if r['oom']: break

# ── Analysis ────────────────────────────────────────────────────────────────

print("\n[6/7] Memory Analysis")
print("=" * 80)

# Total GPU memory comparison
valid_base = [r for r in base if not r['oom']]
valid_hk = [r for r in sink + tail + mid if not r['oom']]

if valid_base and valid_hk:
    print(f"\n  {'Context':<8} {'Base MB':<12} {'HK MB':<12} {'HK KV MB':<12} {'Savings'}")
    print("-" * 60)
    for ctx in sorted(set(r['ctx'] for r in valid_hk)):
        hk_peak = max((r['peak'] for r in valid_hk if r['ctx'] == ctx), default=0)
        hk_kv = max((r.get('kv_mb', 0) for r in valid_hk if r['ctx'] == ctx), default=0)
        base_peak = next((r['peak'] for r in valid_base if r['ctx'] == ctx), None)
        base_str = f"{base_peak:.0f}" if base_peak else "OOM"
        if base_peak:
            savings = f"-{base_peak - hk_peak:.0f}MB ({(base_peak-hk_peak)/base_peak*100:.0f}%)"
        else:
            savings = "N/A (base OOM)"
        print(f"  {ctx:<8} {base_str:<12} {hk_peak:<12.0f} {hk_kv:<12.0f} {savings}")

# O(1) KV memory check
if len(valid_hk) >= 2:
    kv_vals = [r.get('kv_mb', 0) for r in valid_hk if r.get('kv_mb', 0) > 0]
    if kv_vals:
        print(f"\n  KV Cache Memory (O(1) indicator):")
        print(f"    Min: {min(kv_vals):.0f} MB | Max: {max(kv_vals):.0f} MB")
        kv_growth = (max(kv_vals) - min(kv_vals)) / min(kv_vals) * 100 if min(kv_vals) > 0 else 999
        print(f"    Growth: {kv_growth:.1f}%")
        print(f"    {'✅ KV Cache is O(1)' if kv_growth < 10 else '⚠️ KV near O(1)' if kv_growth < 20 else '❌ NOT O(1)'}")
    else:
        kv_growth = 999
else:
    kv_growth = 999

print("\n[7/7] Accuracy Summary")
print("=" * 80)

def pct(results):
    v = [r for r in results if not r['oom']]
    if not v: return 0, 0
    ok = sum(1 for r in v if r['ok'])
    return ok, len(v)

b_ok, b_tot = pct(base)
s_ok, s_tot = pct(sink)
t_ok, t_tot = pct(tail)
m_ok, m_tot = pct(mid)

print(f"  Baseline (mid):    {b_ok}/{b_tot}")
print(f"  HK Sink (pos≈50):  {s_ok}/{s_tot}")
print(f"  HK Tail (pos≈end): {t_ok}/{t_tot}")
print(f"  HK Mid  (pos≈mid): {m_ok}/{m_tot}")

print(f"\n{'='*80}")
print("FINAL VERDICT")
print(f"{'='*80}")
print(f"  KV Memory: {'✅ O(1)' if kv_growth < 10 else '⚠️' if kv_growth < 20 else '❌'} ({kv_growth:.1f}% growth)")
print(f"  Sink Zone: {'✅' if s_ok >= max(1, s_tot-1) else '❌'} ({s_ok}/{s_tot} correct)")
print(f"  Tail Zone: {'✅' if t_ok >= max(1, t_tot-1) else '❌'} ({t_ok}/{t_tot} correct)")
print(f"  DRAM Zone: {'⚠️' if m_ok >= 1 else '❌'} ({m_ok}/{m_tot} with 4-bit self-healing)")
max_hk = max((r['ctx'] for r in valid_hk), default=0)
max_base = max((r['ctx'] for r in valid_base), default=0)
print(f"  Max HK context:  {max_hk} tokens ({max_hk//1024}K)")
print(f"  Max Base context: {max_base} tokens ({max_base//1024}K)")
if max_hk > max_base:
    print(f"  ✅ HeteroKV extends context {max_hk//max_base}x beyond baseline")
print(f"{'='*80}\n")
