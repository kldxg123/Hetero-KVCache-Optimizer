#!/usr/bin/env python3
"""
Method D vs C — Simple Narrative Text
======================================
Uses a simple generated story instead of WikiText-2 to ensure model understands the task.
"""

import torch, time, sys, os, gc
sys.path.insert(0, '/home/app-ahr/Hetero-KVCache-Optimizer/src')
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from transformers import AutoProcessor, LlavaForConditionalGeneration
from core.engine_wrapper import FusedHeteroCache
from core.fused_attention_patch import patch_model_for_fused_attention

CODE = "XY789"

print("=" * 80)
print("  Method D vs C — Simple Narrative Test")
print("=" * 80)

# Load model
print("\n[1/2] Loading model...")
mp = "/home/app-ahr/.cache/huggingface/hub/models--llava-hf--llava-1.5-7b-hf/snapshots"
snaps = sorted([d for d in os.listdir(mp) if os.path.isdir(os.path.join(mp, d))])
mp = os.path.join(mp, snaps[-1])
proc = AutoProcessor.from_pretrained(mp)
model = LlavaForConditionalGeneration.from_pretrained(mp, torch_dtype=torch.float16, device_map="cuda")
model.eval()

# Generate a simple narrative story
print("\n[2/2] Building story...")
story_parts = [
    "Once upon a time, there was a young detective named Alex who loved solving mysteries.",
    "Alex lived in a small town called Maplewood where everyone knew each other.",
    "One sunny morning, Alex found a mysterious note with a secret code: " + CODE + ".",
    "The note was hidden inside an old book at the town library.",
    "Alex decided to investigate this strange discovery and find out what the code meant.",
    "Alex's best friend Sam agreed to help with the investigation.",
    "Together, they visited the library and talked to the librarian, Mrs. Johnson.",
    "Mrs. Johnson remembered that the book had been donated by an elderly gentleman.",
    "The gentleman, Professor Williams, was a former mathematics teacher.",
    "Alex and Sam went to visit Professor Williams at his house.",
    "Professor Williams was surprised to see them and curious about their visit.",
    "When Alex mentioned the secret code, his eyes widened with surprise.",
    "Professor Williams explained that the code was part of a treasure hunt.",
    "The treasure hunt had been created many years ago for his students.",
    "He gave Alex a map showing locations where clues were hidden.",
    "The first clue was located near the old oak tree in the park.",
    "Alex and Sam rushed to the park and found another note there.",
    "The note contained a riddle that they had to solve to proceed.",
    "After thinking for a while, Alex figured out the answer to the riddle.",
    "The answer led them to the town museum where the next clue awaited.",
    "At the museum, the curator showed them an ancient artifact.",
    "The artifact had symbols that matched the secret code they had found.",
    "Alex realized that each clue was bringing them closer to solving the mystery.",
    "The adventure continued as they followed more clues around the town.",
    "Each new discovery revealed more about the town's hidden history.",
    "Alex was determined to find out what the secret code really meant.",
]

# Repeat story to reach target length
story = " ".join(story_parts)
target_len = 35000  # Aim for ~8K+ tokens
while len(story) < target_len:
    story = story + " " + story

print(f"   Story length: {len(story)} chars")

# Tokenize to find positions
story_ids = proc.tokenizer(story, return_tensors='pt').input_ids[0]
print(f"   Story tokens: {story_ids.shape[0]}")


def build_prompt(ctx_tokens, needle_token_pos):
    """Build prompt with needle at specific token position."""
    # Find the code in the story
    code_ids = proc.tokenizer(CODE, add_special_tokens=False).input_ids

    # Build context
    ctx_ids = story_ids[:ctx_tokens].clone()

    # Verify needle position is valid
    if needle_token_pos + len(code_ids) < ctx_tokens:
        ctx_ids[needle_token_pos:needle_token_pos + len(code_ids)] = torch.tensor(code_ids)

    question = "\n\nQuestion: In the story above, what is the secret code? Answer:"
    question_ids = proc.tokenizer(question, add_special_tokens=False).input_ids

    full_ids = torch.cat([ctx_ids, torch.tensor(question_ids)])
    return full_ids


def run_test(method_name, enable_method_d, use_patch, ctx, needle_pos):
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
                    max_new_tokens=20,
                    do_sample=False,
                    past_key_values=cache,
                )
        else:
            out = model.generate(
                input_ids=input_ids_dev,
                attention_mask=attention_mask,
                max_new_tokens=20,
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

        icon = "✅" if ok else "❌"
        print(f"  {icon} {method_name:20} | {zone:5} | tok={n_tokens:5} | "
              f"peak={peak:7.0f}MB | {elapsed:4.1f}s | {raw.strip()[:50]}")
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

# Find where the code appears in the tokenized story
code_span = proc.tokenizer(CODE, add_special_tokens=False).input_ids
code_str = "secret code: " + CODE
code_start_tokens = proc.tokenizer(code_str, add_special_tokens=False).input_ids

# Find first occurrence
import bisect
target_ids = proc.tokenizer("One sunny morning, Alex found a mysterious note with a secret code: " + CODE,
                            add_special_tokens=False).input_ids

test_configs = [
    # (ctx, needle_pos, zone, description)
    (6144, 3072, "DRAM", "DRAM zone (middle)"),
    (6144, 500, "Sink", "Sink zone (beginning)"),
]

for ctx, needle_pos, expected_zone, desc in test_configs:
    print(f"\n  Test: {desc}")
    print(f"    ctx={ctx}, needle_pos={needle_pos}, zone={expected_zone}")
    print(f"    Zones: Sink[0:1024] DRAM[1024:{ctx-2048}] Tail[{ctx-2048}:{ctx}]")

    if story_ids.shape[0] < ctx:
        print(f"    ⚠️  Not enough tokens, skipping")
        continue

    # Method C with patch
    r_c = run_test("Method C (Triton)", False, True, ctx, needle_pos)
    results.append(r_c)

    # Method D without patch
    r_d = run_test("Method D (Query)", True, False, ctx, needle_pos)
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

c_total = len(c_results) if c_results else 1
d_total = len(d_results) if d_results else 1

print(f"\n  Method C (Triton+Patch): {c_ok}/{len(c_results)} ({c_ok/c_total*100:.0f}%)")
print(f"  Method D (Query-aware):  {d_ok}/{len(d_results)} ({d_ok/d_total*100:.0f}%)")

for zone in ['Sink', 'DRAM']:
    cz = [r for r in c_results if r.get('zone') == zone]
    dz = [r for r in d_results if r.get('zone') == zone]
    if cz or dz:
        c_z_ok = sum(1 for r in cz if r['ok'])
        d_z_ok = sum(1 for r in dz if r['ok'])
        print(f"  {zone}: C={c_z_ok}/{len(cz)} | D={d_z_ok}/{len(dz)}")

if c_valid := [r for r in c_results if not r['oom']]:
    print(f"\n  Method C avg: {sum(r['peak'] for r in c_valid)/len(c_valid):.0f}MB, "
          f"{sum(r['time'] for r in c_valid)/len(c_valid):.1f}s")
if d_valid := [r for r in d_results if not r['oom']]:
    print(f"  Method D avg: {sum(r['peak'] for r in d_valid)/len(d_valid):.0f}MB, "
          f"{sum(r['time'] for r in d_valid)/len(d_valid):.1f}s")

print(f"\n{'='*80}\n")
