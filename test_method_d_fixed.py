#!/usr/bin/env python3
"""
Fixed Method D vs C Comparison
===============================
Uses patch_model_for_fused_attention for Method C (Triton path),
and WikiText-2 natural text (not repetitive patterns).
"""

import torch, time, sys, os, gc
sys.path.insert(0, '/home/app-ahr/Hetero-KVCache-Optimizer/src')
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from transformers import AutoProcessor, LlavaForConditionalGeneration
from core.engine_wrapper import FusedHeteroCache
from core.fused_attention_patch import patch_model_for_fused_attention

NEEDLE = "HETEROKV2026"

print("=" * 80)
print("  Fixed Method D vs C Comparison (with Patch + Natural Text)")
print("=" * 80)

# Load model
print("\n[1/3] Loading model...")
mp = "/home/app-ahr/.cache/huggingface/hub/models--llava-hf--llava-1.5-7b-hf/snapshots"
snaps = sorted([d for d in os.listdir(mp) if os.path.isdir(os.path.join(mp, d))])
mp = os.path.join(mp, snaps[-1])
proc = AutoProcessor.from_pretrained(mp)
model = LlavaForConditionalGeneration.from_pretrained(mp, torch_dtype=torch.float16, device_map="cuda")
model.eval()

# Load WikiText-2
print("\n[2/3] Loading WikiText-2...")
from datasets import load_dataset
wiki = load_dataset("wikitext", "wikitext-2-raw-v1", split="test", trust_remote_code=False)

passages = []
for ex in wiki:
    t = ex['text'].strip()
    if len(t) > 200:
        passages.append(t)
corpus = " ".join(passages[:30])
print(f"   Corpus: {len(corpus)} chars")

def build_needle_prompt(ctx_target, needle_pos):
    """Build prompt with needle using natural WikiText-2 text."""
    needle_str = f" The secret code is {NEEDLE}. Remember this code."
    chars_per_token = 4
    total_chars = ctx_target * chars_per_token
    needle_char_pos = int(needle_pos * chars_per_token)

    text = corpus[:total_chars]
    if 0 < needle_char_pos < len(text):
        text = text[:needle_char_pos] + needle_str + text[needle_char_pos:]
    text = text[:total_chars]

    prompt = f"{text}\n\nQuestion: What is the secret code mentioned in the text above?\nAnswer: The secret code is"
    return prompt

def run_test(method_name, enable_method_d, enable_triton, use_patch, ctx, needle_pos):
    """Run a single test."""
    prompt = build_needle_prompt(ctx, needle_pos)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    gc.collect()

    inputs = proc(text=prompt, return_tensors='pt').to('cuda')
    n_tokens = inputs.input_ids.shape[-1]

    # Calculate zones
    sink, tail = 1024, 2048
    if needle_pos < sink:
        zone = "Sink"
    elif needle_pos >= n_tokens - tail:
        zone = "Tail"
    else:
        zone = "DRAM"

    cache = FusedHeteroCache(
        num_layers=32, sink_tokens=sink, keep_tail=tail, chunk_size=2048,
        device='cuda', enable_quant=True, enable_prefetch=True, enable_triton=enable_triton,
        self_healing=True, adaptive_self_healing=True,
        enable_method_d=enable_method_d,
        method_d_alpha=1.0,
    )

    t0 = time.time()
    try:
        # Use patch context manager for Method C (Triton path)
        if use_patch:
            with patch_model_for_fused_attention(model, cache, enable_fused=True):
                out = model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=30,
                    do_sample=False,
                    past_key_values=cache,
                )
        else:
            out = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=30,
                do_sample=False,
                past_key_values=cache,
            )

        elapsed = time.time() - t0
        peak = torch.cuda.max_memory_allocated() / 1024**2
        raw = proc.decode(out[0][-30:], skip_special_tokens=True)
        ok = NEEDLE in raw.upper()

        result = dict(
            method=method_name, ctx=ctx, needle_pos=needle_pos, zone=zone,
            tokens=n_tokens, peak=peak, time=elapsed, ok=ok,
            ans=raw.strip()[:80], oom=False,
        )

        icon = "✅" if ok else "❌"
        print(f"  {icon} {method_name:20} | {zone:5} | tok={n_tokens:4} | "
              f"peak={peak:7.0f}MB | time={elapsed:4.1f}s | {raw[:40]}")
        return result

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            peak = torch.cuda.max_memory_allocated() / 1024**2
            result = dict(
                method=method_name, ctx=ctx, needle_pos=needle_pos, zone=zone,
                tokens=n_tokens, peak=peak, time=0, ok=False, ans="OOM", oom=True,
            )
            print(f"  ❌ {method_name:20} | {zone:5} | OOM at {peak:.0f}MB")
            return result
        raise
    finally:
        del inputs, cache
        gc.collect()
        torch.cuda.empty_cache()

# Run tests
print("\n[3/3] Running tests...")
print("-" * 80)

results = []
test_configs = [
    # (ctx, needle_pos, expected_zone)
    (8192, 4096, "DRAM"),  # 8K, needle in DRAM (multi-chunk)
    (8192, 500, "Sink"),   # 8K, needle in Sink
    (8192, 7000, "Tail"),  # 8K, needle in Tail
]

for ctx, needle_pos, expected_zone in test_configs:
    print(f"\n  ctx={ctx}, needle_pos={needle_pos}, expected_zone={expected_zone}")

    # Method C: Triton path (REQUIRES patch)
    r_c = run_test("Method C (Triton+Patch)", False, True, True, ctx, needle_pos)
    results.append(r_c)

    # Method D: Query-aware path (does NOT require patch)
    r_d = run_test("Method D (Query-aware)", True, False, False, ctx, needle_pos)
    results.append(r_d)

# Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

for zone in ['Sink', 'DRAM', 'Tail']:
    c_results = [r for r in results if 'Method C' in r['method'] and r.get('zone') == zone]
    d_results = [r for r in results if 'Method D' in r['method'] and r.get('zone') == zone]

    if c_results or d_results:
        c_ok = sum(1 for r in c_results if r['ok'])
        d_ok = sum(1 for r in d_results if r['ok'])
        c_total = len(c_results)
        d_total = len(d_results)

        print(f"\n  {zone} zone:")
        if c_total > 0:
            print(f"    Method C (Triton+Patch): {c_ok}/{c_total} ({c_ok/c_total*100:.0f}%)")
        if d_total > 0:
            print(f"    Method D (Query-aware):  {d_ok}/{d_total} ({d_ok/d_total*100:.0f}%)")

print("\n" + "=" * 80 + "\n")
