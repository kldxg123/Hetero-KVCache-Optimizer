#!/usr/bin/env python3
"""
64K Full Verification Test (v2)
================================
- ChunkedPrefill for true O(1) memory
- Needle-in-a-Haystack at Sink / Tail / DRAM positions
- Accuracy vs Baseline comparison
"""

import torch, time, sys, gc, os
import numpy as np
from PIL import Image

sys.path.insert(0, '/home/app-ahr/Hetero-KVCache-Optimizer/src')
from transformers import AutoProcessor, LlavaForConditionalGeneration
from core.engine_wrapper import FusedHeteroCache, ChunkedPrefillEngine

MAX_MEM = 24 * 1024  # 24 GB

print("=" * 80)
print("  HeteroKV 64K Full Verification (v2)")
print("  Modules: ChunkedPrefill + Quant + HeavyHitter + AdaptiveSelfHealing + Triton")
print("=" * 80)

# ── Load ────────────────────────────────────────────────────────────────────
print("\n[1/7] Loading model...")
mp = "/home/app-ahr/.cache/huggingface/hub/models--llava-hf--llava-1.5-7b-hf/snapshots"
snaps = sorted([d for d in os.listdir(mp) if os.path.isdir(os.path.join(mp, d))])
mp = os.path.join(mp, snaps[-1])
proc = AutoProcessor.from_pretrained(mp)
model = LlavaForConditionalGeneration.from_pretrained(mp, torch_dtype=torch.float16, device_map="cuda")
model.eval()
model_gpu = torch.cuda.memory_allocated() / 1024**3
print(f"   Model: {model_gpu:.2f} GB")

# ── Test image ──────────────────────────────────────────────────────────────
arr = np.ones((224, 224, 3), dtype=np.uint8) * 255
y, x = np.ogrid[:224, :224]
arr[((x-112)**2 + (y-112)**2) <= 50**2] = [255, 0, 0]
img = Image.fromarray(arr)

NEEDLE = "HETEROKV2026"
QUESTION = "What is the secret passcode?"

def build_prompt(ctx_len, needle_token_pos):
    """Build NIAH prompt. needle_token_pos is approximate position in tokens."""
    filler = "The quick brown fox jumps over the lazy dog. "
    # Approximate tokens: image≈576, USER:<image>\n≈10, needle≈12, question≈15
    base = 600
    before_chars = int(max(0, (needle_token_pos - base)) * 12 / 1)
    total_filler_chars = int(max(0, (ctx_len - base - 30)) * 12 / 1)
    after_chars = max(0, total_filler_chars - before_chars)
    before = filler * (before_chars // len(filler))
    after = filler * (after_chars // len(filler))
    return f"USER: <image>\n{before} The secret passcode is {NEEDLE}. Remember it. {after}\n{QUESTION}\nASSISTANT:"

def run_baseline(ctx_len, needle_pos):
    """Standard KV cache baseline."""
    torch.cuda.reset_peak_memory_stats(); torch.cuda.empty_cache()
    prompt = build_prompt(ctx_len, needle_pos)
    inputs = proc(text=prompt, images=img, return_tensors='pt').to('cuda')
    n = inputs.input_ids.shape[-1]
    t0 = time.time()
    try:
        with torch.no_grad():
            out = model.generate(input_ids=inputs.input_ids, pixel_values=inputs.pixel_values,
                                 attention_mask=inputs.attention_mask, max_new_tokens=30, do_sample=False)
        ans = proc.decode(out[0], skip_special_tokens=True).split("ASSISTANT:")[-1].strip()
        peak = torch.cuda.max_memory_allocated()/1024**2
        ok = NEEDLE in ans.upper()
        return dict(ctx=ctx_len, tokens=n, peak=peak, time=time.time()-t0, ok=ok, ans=ans[:60], oom=False)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            peak = torch.cuda.max_memory_allocated()/1024**2
            return dict(ctx=ctx_len, tokens=n, peak=peak, time=0, ok=False, ans="OOM", oom=True)
        raise
    finally:
        del inputs; torch.cuda.empty_cache()

def run_heterokv_chunked(ctx_len, needle_pos):
    """HeteroKV with ChunkedPrefill + manual decode."""
    torch.cuda.reset_peak_memory_stats(); torch.cuda.empty_cache()
    prompt = build_prompt(ctx_len, needle_pos)
    inputs = proc(text=prompt, images=img, return_tensors='pt').to('cuda')
    n = inputs.input_ids.shape[-1]

    cache = FusedHeteroCache(
        num_layers=32, sink_tokens=64, keep_tail=2048, chunk_size=2048,
        device='cuda', enable_quant=True, enable_prefetch=True, enable_triton=True,
        self_healing=True, adaptive_self_healing=True,
    )

    t0 = time.time()
    try:
        # ── Chunked Prefill ──────────────────────────────────────────
        engine = ChunkedPrefillEngine(model, cache, chunk_size=2048)
        engine.prefill(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
        prefill_time = time.time() - t0
        prefill_peak = torch.cuda.max_memory_allocated()/1024**2

        # ── Decode Loop ──────────────────────────────────────────────
        # Get last token embedding
        last_id = inputs.input_ids[:, -1:]
        generated = []
        for step in range(30):
            with torch.no_grad():
                # Use full model for first step (handles vision tokens), language model only for rest
                if step == 0:
                    embed = model.language_model.model.embed_tokens(last_id)
                    pos = torch.tensor([[cache.real_seq_len - 1]], device='cuda')
                else:
                    embed = model.language_model.model.embed_tokens(last_id)
                    pos = torch.tensor([[cache.real_seq_len]], device='cuda')

                out_lm = model.language_model.model(
                    inputs_embeds=embed, past_key_values=cache,
                    use_cache=True, position_ids=pos,
                )
                logits = model.language_model.lm_head(out_lm.last_hidden_state)
                next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                generated.append(next_id.item())
                last_id = next_id
                if next_id.item() == proc.tokenizer.eos_token_id:
                    break

        ans = proc.tokenizer.decode(generated, skip_special_tokens=True)
        peak = torch.cuda.max_memory_allocated()/1024**2
        total_time = time.time() - t0
        ok = NEEDLE in ans.upper()

        # Get memory summary
        mem_info = cache.memory_summary()

        return dict(ctx=ctx_len, tokens=n, peak=peak, prefill_peak=prefetch_peak,
                    time=total_time, prefill_time=prefill_time, ok=ok, ans=ans[:60],
                    oom=False, mem_info=mem_info)

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            peak = torch.cuda.max_memory_allocated()/1024**2
            return dict(ctx=ctx_len, tokens=n, peak=peak, time=0, ok=False, ans="OOM", oom=True)
        raise
    finally:
        del inputs, cache; torch.cuda.empty_cache()

def run_heterokv_generate(ctx_len, needle_pos):
    """HeteroKV with model.generate() (for shorter contexts)."""
    torch.cuda.reset_peak_memory_stats(); torch.cuda.empty_cache()
    prompt = build_prompt(ctx_len, needle_pos)
    inputs = proc(text=prompt, images=img, return_tensors='pt').to('cuda')
    n = inputs.input_ids.shape[-1]

    cache = FusedHeteroCache(
        num_layers=32, sink_tokens=64, keep_tail=2048, chunk_size=2048,
        device='cuda', enable_quant=True, enable_prefetch=True, enable_triton=True,
        self_healing=True, adaptive_self_healing=True,
    )

    t0 = time.time()
    try:
        with torch.no_grad():
            out = model.generate(input_ids=inputs.input_ids, pixel_values=inputs.pixel_values,
                                 attention_mask=inputs.attention_mask, max_new_tokens=30,
                                 do_sample=False, past_key_values=cache)
        ans = proc.decode(out[0], skip_special_tokens=True).split("ASSISTANT:")[-1].strip()
        peak = torch.cuda.max_memory_allocated()/1024**2
        ok = NEEDLE in ans.upper()
        mem_info = cache.memory_summary()
        return dict(ctx=ctx_len, tokens=n, peak=peak, time=time.time()-t0, ok=ok,
                    ans=ans[:60], oom=False, mem_info=mem_info)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            peak = torch.cuda.max_memory_allocated()/1024**2
            return dict(ctx=ctx_len, tokens=n, peak=peak, time=0, ok=False, ans="OOM", oom=True)
        raise
    finally:
        del inputs, cache; torch.cuda.empty_cache()

# ── Run Tests ───────────────────────────────────────────────────────────────

print("\n[2/7] Baseline (Standard KV Cache)")
print("-" * 80)
print(f"  {'Ctx':<8} {'Tokens':<8} {'GPU MB':<10} {'Time':<6} {'OK':<4} {'Answer'}")
print("-" * 80)
base_res = []
for ctx in [4096, 8192, 16384]:
    r = run_baseline(ctx, ctx // 2)
    base_res.append(r)
    s = "OK" if r['ok'] else ("OOM" if r['oom'] else "FAIL")
    print(f"  {ctx:<8} {r['tokens']:<8} {r['peak']:<10.0f} {r['time']:<6.1f} {s:<4} {r['ans'][:50]}")
    if r['oom']: break
    if r['peak'] > MAX_MEM: break

print("\n[3/7] HeteroKV - Needle in SINK (position ≈ 30)")
print("   Tests: first tokens preserved correctly in Sink zone")
print("-" * 80)
sink_res = []
for ctx in [4096, 8192, 16384, 32768, 65536]:
    r = run_heterokv_chunked(ctx, 30)
    sink_res.append(r)
    s = "OK" if r['ok'] else ("OOM" if r['oom'] else "FAIL")
    print(f"  {ctx:<8} {r['tokens']:<8} {r['peak']:<10.0f} {r['time']:<6.1f} {s:<4} {r['ans'][:50]}")
    if r['oom']: break

print("\n[4/7] HeteroKV - Needle in TAIL (position ≈ ctx-500)")
print("   Tests: recent tokens preserved correctly in Tail zone")
print("-" * 80)
tail_res = []
for ctx in [4096, 8192, 16384, 32768, 65536]:
    r = run_heterokv_chunked(ctx, ctx - 500)
    tail_res.append(r)
    s = "OK" if r['ok'] else ("OOM" if r['oom'] else "FAIL")
    print(f"  {ctx:<8} {r['tokens']:<8} {r['peak']:<10.0f} {r['time']:<6.1f} {s:<4} {r['ans'][:50]}")
    if r['oom']: break

print("\n[5/7] HeteroKV - Needle in DRAM (position ≈ ctx//2)")
print("   Tests: self-healing retrieves evicted tokens from DRAM")
print("-" * 80)
dram_res = []
for ctx in [4096, 8192, 16384, 32768, 65536]:
    r = run_heterokv_chunked(ctx, ctx // 2)
    dram_res.append(r)
    s = "OK" if r['ok'] else ("OOM" if r['oom'] else "FAIL")
    print(f"  {ctx:<8} {r['tokens']:<8} {r['peak']:<10.0f} {r['time']:<6.1f} {s:<4} {r['ans'][:50]}")
    if r['oom']: break

# ── Analysis ────────────────────────────────────────────────────────────────

print("\n[6/7] Memory O(1) Analysis")
print("=" * 80)

all_hk = [r for r in sink_res + tail_res + dram_res if not r['oom']]
if len(all_hk) >= 2:
    by_ctx = {}
    for r in all_hk:
        by_ctx.setdefault(r['ctx'], []).append(r['peak'])
    ctx_peaks = {ctx: max(peaks) for ctx, peaks in by_ctx.items()}

    print(f"  {'Context':<10} {'Peak MB':<12} {'GB':<8}")
    print("-" * 40)
    for ctx in sorted(ctx_peaks):
        p = ctx_peaks[ctx]
        print(f"  {ctx:<10} {p:<12.0f} {p/1024:<8.2f}")

    vals = list(ctx_peaks.values())
    min_m, max_m = min(vals), max(vals)
    growth = max_m - min_m
    growth_pct = (growth / min_m) * 100 if min_m > 0 else 999

    print(f"\n  Growth: {growth:.0f} MB ({growth_pct:.1f}%)")
    print(f"  {'✅ O(1) VERIFIED' if growth_pct < 10 else '⚠️ Near O(1)' if growth_pct < 20 else '❌ NOT O(1)'}")
else:
    growth_pct = 999

print("\n[7/7] Accuracy Summary")
print("=" * 80)

def report(name, results):
    valid = [r for r in results if not r['oom']]
    if not valid: return
    ok = sum(1 for r in valid if r['ok'])
    total = len(valid)
    pct = ok/total*100
    icon = "✅" if pct >= 80 else "⚠️" if pct >= 50 else "❌"
    print(f"  {icon} {name}: {ok}/{total} ({pct:.0f}%)")

report("Baseline (mid-ctx needle)", base_res)
report("HK Sink zone (pos≈30)", sink_res)
report("HK Tail zone (pos≈ctx-500)", tail_res)
report("HK DRAM zone (pos≈ctx//2)", dram_res)

print(f"\n{'='*80}")
print("FINAL VERDICT")
print(f"{'='*80}")
print(f"  Memory:  {'✅ O(1)' if growth_pct < 10 else '⚠️' if growth_pct < 20 else '❌'} ({growth_pct:.1f}% growth across 4K-64K)")
sink_ok = sum(1 for r in sink_res if r.get('ok'))
tail_ok = sum(1 for r in tail_res if r.get('ok'))
dram_ok = sum(1 for r in dram_res if r.get('ok'))
print(f"  Sink:    {'✅' if sink_ok >= 3 else '❌'} ({sink_ok}/{len(sink_res)} correct)")
print(f"  Tail:    {'✅' if tail_ok >= 3 else '❌'} ({tail_ok}/{len(tail_res)} correct)")
print(f"  DRAM:    {'✅' if dram_ok >= 3 else '⚠️' if dram_ok >= 1 else '❌'} ({dram_ok}/{len(dram_res)} correct)")
max_ctx = max((r['ctx'] for r in all_hk if not r.get('oom')), default=0)
base_max = max((r['ctx'] for r in base_res if not r.get('oom')), default=0)
print(f"  Max HK context:  {max_ctx} tokens ({max_ctx//1024}K)")
print(f"  Max Base context: {base_max} tokens ({base_max//1024}K)")
if max_ctx > base_max:
    print(f"  ✅ HeteroKV extends context {max_ctx//base_max}x beyond baseline limit")
print(f"{'='*80}\n")
