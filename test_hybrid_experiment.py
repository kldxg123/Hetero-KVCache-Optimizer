#!/usr/bin/env python3
"""
Controlled Experiment: Prove Query-Aware + Tiered Precision is Better
=====================================================================
Uses real WikiText-2 text, multiple context lengths, and measures:
  - Perplexity (PPL) — gold standard for KV cache quality
  - Needle-in-haystack accuracy
  - Peak memory + latency
"""

import torch, time, sys, os, gc, math
import numpy as np

sys.path.insert(0, '/home/app-ahr/Hetero-KVCache-Optimizer/src')
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from transformers import AutoProcessor, LlavaForConditionalGeneration, AutoTokenizer, AutoModelForCausalLM
from core.engine_wrapper import FusedHeteroCache

NEEDLE = "HETEROKV2026"

print("=" * 80)
print("  Controlled Experiment: Query-Aware + Tiered Precision")
print("=" * 80)

# ── Load Model ────────────────────────────────────────────────────────────────
print("\n[1/4] Loading model...")
mp = "/home/app-ahr/.cache/huggingface/hub/models--llava-hf--llava-1.5-7b-hf/snapshots"
snaps = sorted([d for d in os.listdir(mp) if os.path.isdir(os.path.join(mp, d))])
mp = os.path.join(mp, snaps[-1])
proc = AutoProcessor.from_pretrained(mp)
model = LlavaForConditionalGeneration.from_pretrained(mp, torch_dtype=torch.float16, device_map="cuda")
model.eval()
print(f"   Model: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

# ── Load WikiText-2 ────────────────────────────────────────────────────────────
print("\n[2/4] Loading WikiText-2...")
from datasets import load_dataset
wiki = load_dataset("wikitext", "wikitext-2-raw-v1", split="test", trust_remote_code=False)

# Collect natural text passages
passages = []
for ex in wiki:
    t = ex['text'].strip()
    if len(t) > 200:  # Longer passages for better quality
        passages.append(t)

# Build a large natural text corpus
corpus = " ".join(passages[:50])
print(f"   Corpus: {len(corpus)} chars from {len(passages[:50])} passages")


def build_needle_prompt(ctx_tokens_target, needle_pos):
    """Build prompt with needle at specific position using natural text."""
    needle_str = f"The unique identifier code is {NEEDLE}. Please remember this identifier. "

    chars_per_token = 4
    total_chars = ctx_tokens_target * chars_per_token
    needle_char_pos = int(needle_pos * chars_per_token)

    text = corpus[:total_chars]

    if 0 < needle_char_pos < len(text):
        text = text[:needle_char_pos] + needle_str + text[needle_char_pos:]

    text = text[:total_chars]

    prompt = f"{text}\n\nBased on the text above, what is the unique identifier code mentioned?\nThe unique identifier code is"
    return prompt


def build_continuation_prompt(ctx_tokens_target):
    """Build prompt for next-token prediction (PPL measurement)."""
    chars_per_token = 4
    total_chars = ctx_tokens_target * chars_per_token
    text = corpus[:total_chars]

    # Split: use 90% as context, 10% as target
    split_pos = int(len(text) * 0.9)
    context = text[:split_pos]
    target = text[split_pos:split_pos + 200]  # ~50 tokens target

    return context, target


# ── Test Functions ─────────────────────────────────────────────────────────────
def run_needle_test(method_name, enable_method_d, ctx, needle_pos):
    """Needle-in-haystack accuracy test."""
    prompt = build_needle_prompt(ctx, needle_pos)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    gc.collect()

    inputs = proc(text=prompt, return_tensors='pt').to('cuda')
    n_tokens = inputs.input_ids.shape[-1]

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
                max_new_tokens=30,
                do_sample=False,
                past_key_values=cache,
            )
        elapsed = time.time() - t0
        peak = torch.cuda.max_memory_allocated() / 1024**2
        raw = proc.decode(out[0][-30:], skip_special_tokens=True)
        ok = NEEDLE in raw.upper()

        result = dict(
            method=method_name, ctx=ctx, tokens=n_tokens,
            peak=peak, time=elapsed, ok=ok,
            ans=raw.strip()[:80], oom=False,
        )
        return result

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            peak = torch.cuda.max_memory_allocated() / 1024**2
            return dict(
                method=method_name, ctx=ctx, tokens=n_tokens,
                peak=peak, time=0, ok=False, ans="OOM", oom=True,
            )
        raise
    finally:
        del inputs, cache
        gc.collect()
        torch.cuda.empty_cache()


def run_ppl_test(method_name, enable_method_d, ctx_tokens):
    """Perplexity test using next-token prediction."""
    context, target = build_continuation_prompt(ctx_tokens)
    full_text = context + target

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    gc.collect()

    inputs = proc(text=full_text, return_tensors='pt').to('cuda')
    input_ids = inputs.input_ids
    n_tokens = input_ids.shape[-1]

    # Target is last portion
    target_len = min(50, n_tokens // 10)
    target_start = n_tokens - target_len

    cache = FusedHeteroCache(
        num_layers=32, sink_tokens=1024, keep_tail=2048, chunk_size=2048,
        device='cuda', enable_quant=True, enable_prefetch=True, enable_triton=True,
        self_healing=True, adaptive_self_healing=True,
        enable_method_d=enable_method_d,
        method_d_alpha=1.0,
    )

    try:
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=inputs.attention_mask,
                past_key_values=cache,
                use_cache=True,
            )
            logits = outputs.logits

            # Compute PPL on target tokens
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()

            # Only compute on target portion
            target_logits = shift_logits[..., target_start:, :]
            target_labels = shift_labels[..., target_start:]

            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(
                target_logits.view(-1, target_logits.size(-1)),
                target_labels.view(-1)
            )
            ppl = math.exp(loss.item())

        peak = torch.cuda.max_memory_allocated() / 1024**2

        return dict(
            method=method_name, ctx=ctx_tokens, tokens=n_tokens,
            ppl=ppl, loss=loss.item(), peak=peak, oom=False,
        )

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            peak = torch.cuda.max_memory_allocated() / 1024**2
            return dict(method=method_name, ctx=ctx_tokens, ppl=float('inf'), oom=True, peak=peak)
        raise
    finally:
        del inputs, cache, outputs if 'outputs' in dir() else None
        gc.collect()
        torch.cuda.empty_cache()


# ── Run Experiments ─────────────────────────────────────────────────────────────
print("\n[3/4] Running experiments...")
print("-" * 80)

# Experiment 1: PPL comparison at different context lengths
print("\n  Experiment 1: Perplexity (PPL)")
print(f"  {'Config':12} {'Ctx':>5} {'Tokens':>6} {'PPL':>8} {'Peak MB':>9} {'OOM':>4}")
print("  " + "-" * 50)

ppl_results = []
for ctx in [4096, 8192]:
    for name, enable_d in [("Method C", False), ("Method D", True)]:
        r = run_ppl_test(name, enable_d, ctx)
        ppl_results.append(r)
        if not r['oom']:
            print(f"  {name:12} {ctx:5} {r['tokens']:6} {r['ppl']:8.2f} {r['peak']:9.0f} {'No':>4}")
        else:
            print(f"  {name:12} {ctx:5} {'OOM':>6} {'---':>8} {r['peak']:9.0f} {'Yes':>4}")

# Experiment 2: Needle-in-haystack at different positions
print("\n\n  Experiment 2: Needle-in-Haystack Accuracy")
print(f"  {'Config':12} {'Ctx':>5} {'Zone':>6} {'Tokens':>6} {'Found':>6} {'Peak MB':>9} {'Time':>5}")
print("  " + "-" * 60)

needle_results = []
test_configs = [
    # (ctx, needle_pos, zone)
    (8192, 4096, "DRAM"),     # 8K, needle in DRAM zone (multi-chunk)
    (8192, 500, "Sink"),      # 8K, needle in sink
    (8192, 7000, "Tail"),     # 8K, needle in tail
]

for ctx, needle_pos, zone in test_configs:
    for name, enable_d in [("Method C", False), ("Method D", True)]:
        r = run_needle_test(name, enable_d, ctx, needle_pos)
        r['zone'] = zone
        needle_results.append(r)
        found = "YES" if r['ok'] else "NO"
        if not r['oom']:
            print(f"  {name:12} {ctx:5} {zone:6} {r['tokens']:6} {found:6} {r['peak']:9.0f} {r['time']:4.1f}s")
        else:
            print(f"  {name:12} {ctx:5} {zone:6} {'OOM':>6}")

# ── Analysis ────────────────────────────────────────────────────────────────────
print("\n[4/4] Analysis")
print("=" * 80)

# PPL comparison
print("\n  Perplexity Comparison:")
for ctx in [4096, 8192]:
    c_ppl = [r for r in ppl_results if r['method'] == 'Method C' and r.get('ctx') == ctx and not r['oom']]
    d_ppl = [r for r in ppl_results if r['method'] == 'Method D' and r.get('ctx') == ctx and not r['oom']]

    if c_ppl and d_ppl:
        c = c_ppl[0]
        d = d_ppl[0]
        diff = d['ppl'] - c['ppl']
        pct = diff / c['ppl'] * 100
        if abs(pct) < 5:
            verdict = "Comparable (within 5%)"
        elif pct > 0:
            verdict = "Method D slightly worse"
        else:
            verdict = "Method D slightly better"

        print(f"    ctx={ctx}: Method C PPL={c['ppl']:.2f} | Method D PPL={d['ppl']:.2f} | Diff={diff:+.2f} ({pct:+.1f}%) | {verdict}")

# Needle comparison
print("\n  Needle-in-Haystack Comparison:")
for zone in ['Sink', 'DRAM', 'Tail']:
    c_results = [r for r in needle_results if r['method'] == 'Method C' and r.get('zone') == zone]
    d_results = [r for r in needle_results if r['method'] == 'Method D' and r.get('zone') == zone]

    c_ok = sum(1 for r in c_results if r['ok'])
    d_ok = sum(1 for r in d_results if r['ok'])
    c_total = len(c_results)
    d_total = len(d_results)

    if c_total > 0 or d_total > 0:
        print(f"    {zone}: Method C={c_ok}/{c_total} | Method D={d_ok}/{d_total}")

# ── Final Verdict ──────────────────────────────────────────────────────────────
print(f"\n{'='*80}")
print("CONCLUSIONS & NEXT STEPS")
print(f"{'='*80}")

print("""
  To prove query-aware + tiered precision (8/4-bit) is better, we need:

  1. Multi-chunk DRAM scenario:
     - Context ≥ 8K tokens → multiple DRAM chunks
     - Method D can selectively retrieve relevant chunks
     - Method C retrieves based on historical attention only

  2. Tiered precision experiment:
     - Important chunks (high query similarity): 8-bit quantization
     - Unimportant chunks: 4-bit quantization
     - Expected: 8/4-bit + Method D > 4-bit + Method C in accuracy
     - Expected: memory overhead < 30% (only ~30% chunks at 8-bit)

  3. Key metrics to report:
     - PPL degradation: should be < 2% with hybrid approach
     - Needle accuracy: should improve in DRAM zone
     - Memory efficiency: accuracy-per-byte should improve

  The next step is implementing the tiered quantization in HeteroKVManager.
""")
print("=" * 80 + "\n")
