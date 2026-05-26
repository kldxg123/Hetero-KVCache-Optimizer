#!/usr/bin/env python3
"""
Fixed Method D vs C Comparison — Sufficient Context Length
==========================================================
Uses WikiText-2 train split (much larger) to ensure 8K+ token contexts.
"""

import torch, time, sys, os, gc
sys.path.insert(0, '/home/app-ahr/Hetero-KVCache-Optimizer/src')
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from transformers import AutoProcessor, LlavaForConditionalGeneration
from core.engine_wrapper import FusedHeteroCache
from core.fused_attention_patch import patch_model_for_fused_attention

NEEDLE = "HETEROKV2026"

print("=" * 80)
print("  Fixed Method D vs C — Sufficient Context")
print("=" * 80)

# Load model
print("\n[1/3] Loading model...")
mp = "/home/app-ahr/.cache/huggingface/hub/models--llava-hf--llava-1.5-7b-hf/snapshots"
snaps = sorted([d for d in os.listdir(mp) if os.path.isdir(os.path.join(mp, d))])
mp = os.path.join(mp, snaps[-1])
proc = AutoProcessor.from_pretrained(mp)
model = LlavaForConditionalGeneration.from_pretrained(mp, torch_dtype=torch.float16, device_map="cuda")
model.eval()

# Load WikiText-2 TRAIN split (much more data)
print("\n[2/3] Loading WikiText-2 train split...")
from datasets import load_dataset
wiki = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", trust_remote_code=False)

passages = []
for ex in wiki:
    t = ex['text'].strip()
    if len(t) > 100:
        passages.append(t)
    if len(passages) >= 500:
        break

corpus = " ".join(passages)
print(f"   Corpus: {len(corpus)} chars (~{len(corpus)//4} tokens)")

# Pre-tokenize to get exact token positions
print("\n   Pre-tokenizing corpus...")
all_ids = proc.tokenizer(corpus, return_tensors='pt').input_ids[0]
total_tokens_available = all_ids.shape[0]
print(f"   Total tokens available: {total_tokens_available}")


def build_needle_ids(ctx_target, needle_pos):
    """Build token IDs with needle at exact position using real tokenizer."""
    needle_str = f"The secret code is {NEEDLE}. Remember this code."
    needle_ids = proc.tokenizer(needle_str, add_special_tokens=False).input_ids
    needle_len = len(needle_ids)

    # Extract context from corpus
    ctx_ids = all_ids[:ctx_target].clone()

    # Insert needle at exact position
    if needle_pos + needle_len < ctx_target:
        # Replace tokens at needle_pos with needle
        ctx_ids[needle_pos:needle_pos + needle_len] = torch.tensor(needle_ids)

    # Add question at end
    question = "\n\nQuestion: What is the secret code mentioned in the text?\nAnswer: The secret code is"
    question_ids = proc.tokenizer(question, add_special_tokens=False).input_ids

    full_ids = torch.cat([ctx_ids, torch.tensor(question_ids)])
    return full_ids, needle_len


def run_test(method_name, enable_method_d, use_patch, ctx, needle_pos):
    """Run a single test with exact token positions."""
    input_ids, needle_len = build_needle_ids(ctx, needle_pos)

    n_tokens = input_ids.shape[0]

    # Create attention mask
    attention_mask = torch.ones(1, n_tokens, dtype=torch.long)

    # Calculate zones
    sink, tail = 1024, 2048
    if needle_pos < sink:
        zone = "Sink"
    elif needle_pos >= ctx - tail:
        zone = "Tail"
    else:
        zone = "DRAM"

    print(f"\n  {method_name}: zone={zone} | total_tokens={n_tokens} | needle_pos={needle_pos}")

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    gc.collect()

    input_ids_dev = input_ids.unsqueeze(0).to('cuda')
    attn_mask_dev = attention_mask.to('cuda')

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
                    attention_mask=attn_mask_dev,
                    max_new_tokens=30,
                    do_sample=False,
                    past_key_values=cache,
                )
        else:
            out = model.generate(
                input_ids=input_ids_dev,
                attention_mask=attn_mask_dev,
                max_new_tokens=30,
                do_sample=False,
                past_key_values=cache,
            )

        elapsed = time.time() - t0
        peak = torch.cuda.max_memory_allocated() / 1024**2

        # Decode only the generated part
        generated = out[0][n_tokens:]
        raw = proc.tokenizer.decode(generated, skip_special_tokens=True)
        ok = NEEDLE in raw.upper()

        result = dict(
            method=method_name, ctx=ctx, needle_pos=needle_pos, zone=zone,
            tokens=n_tokens, peak=peak, time=elapsed, ok=ok,
            ans=raw.strip()[:80], oom=False,
        )

        icon = "✅" if ok else "❌"
        print(f"    {icon} {method_name:25} | {zone:5} | tok={n_tokens:5} | "
              f"peak={peak:7.0f}MB | {elapsed:5.1f}s | {raw.strip()[:50]}")
        return result

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            peak = torch.cuda.max_memory_allocated() / 1024**2
            result = dict(
                method=method_name, ctx=ctx, needle_pos=needle_pos, zone=zone,
                tokens=n_tokens, peak=peak, time=0, ok=False, ans="OOM", oom=True,
            )
            print(f"    ❌ {method_name:25} | OOM at {peak:.0f}MB")
            return result
        raise
    finally:
        del input_ids_dev, attn_mask_dev, cache
        gc.collect()
        torch.cuda.empty_cache()


# ── Run Tests ─────────────────────────────────────────────────────────────────
print("\n[3/3] Running comparison tests...")
print("=" * 80)

results = []

test_configs = [
    # (ctx, needle_pos, expected_zone)
    (8192, 4096, "DRAM"),   # 8K, needle squarely in DRAM
    (6144, 3072, "DRAM"),   # 6K, needle in DRAM
    (8192, 500, "Sink"),    # 8K, needle in Sink (should always work)
]

for ctx, needle_pos, expected_zone in test_configs:
    print(f"\n{'─'*70}")
    print(f"  TEST: ctx={ctx}, needle_pos={needle_pos}, expected={expected_zone}")

    # Verify zone calculation
    sink, tail = 1024, 2048
    dram_start = sink
    dram_end = ctx - tail
    print(f"  Zones: Sink[0:{sink}] DRAM[{dram_start}:{dram_end}] Tail[{dram_end}:{ctx}]")
    print(f"  Needle at {needle_pos} → zone={expected_zone}")

    if total_tokens_available < ctx + 100:
        print(f"  ⚠️ Skipping: not enough tokens ({total_tokens_available} < {ctx})")
        continue

    # Method C with patch
    r_c = run_test("Method C (Triton+Patch)", False, True, ctx, needle_pos)
    results.append(r_c)

    # Method D without patch (it doesn't need it)
    r_d = run_test("Method D (Query-aware)", True, False, ctx, needle_pos)
    results.append(r_d)

    if r_c['oom'] and r_d['oom']:
        print("  Both OOM, skipping remaining tests")
        break

# ── Analysis ───────────────────────────────────────────────────────────────────
print(f"\n{'='*80}")
print("ANALYSIS")
print(f"{'='*80}")

c_results = [r for r in results if 'Method C' in r['method']]
d_results = [r for r in results if 'Method D' in r['method']]

c_ok = sum(1 for r in c_results if r['ok'])
d_ok = sum(1 for r in d_results if r['ok'])
c_valid = [r for r in c_results if not r['oom']]
d_valid = [r for r in d_results if not r['oom']]

print(f"\n  Method C (Triton+Patch):  {c_ok}/{len(c_results)} accuracy")
print(f"  Method D (Query-aware):   {d_ok}/{len(d_results)} accuracy")

if c_valid:
    print(f"  Method C avg peak: {sum(r['peak'] for r in c_valid)/len(c_valid):.0f}MB")
    print(f"  Method C avg time: {sum(r['time'] for r in c_valid)/len(c_valid):.1f}s")
if d_valid:
    print(f"  Method D avg peak: {sum(r['peak'] for r in d_valid)/len(d_valid):.0f}MB")
    print(f"  Method D avg time: {sum(r['time'] for r in d_valid)/len(d_valid):.1f}s")

# Per-zone
for zone in ['Sink', 'DRAM', 'Tail']:
    cz = [r for r in c_results if r.get('zone') == zone]
    dz = [r for r in d_results if r.get('zone') == zone]
    if cz or dz:
        c_z_ok = sum(1 for r in cz if r['ok'])
        d_z_ok = sum(1 for r in dz if r['ok'])
        print(f"\n  {zone}: C={c_z_ok}/{len(cz)} | D={d_z_ok}/{len(dz)}")

print(f"\n{'='*80}\n")
