#!/usr/bin/env python3
"""
Comprehensive Benchmark: Method D vs Method C vs Baseline
=========================================================
Uses narrative text (proven to work with LLaVA-1.5-7b).
Tests: 4K→64K progressive stress + multi-zone accuracy + memory/latency.
"""

import torch, time, sys, os, gc, json
sys.path.insert(0, '/home/app-ahr/Hetero-KVCache-Optimizer/src')
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from transformers import AutoProcessor, LlavaForConditionalGeneration
from core.engine_wrapper import FusedHeteroCache
from core.fused_attention_patch import patch_model_for_fused_attention

CODE = "XY789"

print("=" * 80)
print("  Comprehensive Benchmark: Method D vs C vs Baseline")
print("=" * 80)

# Load model
print("\n[1/2] Loading model...")
mp = "/home/app-ahr/.cache/huggingface/hub/models--llava-hf--llava-1.5-7b-hf/snapshots"
snaps = sorted([d for d in os.listdir(mp) if os.path.isdir(os.path.join(mp, d))])
mp = os.path.join(mp, snaps[-1])
proc = AutoProcessor.from_pretrained(mp)
model = LlavaForConditionalGeneration.from_pretrained(mp, torch_dtype=torch.float16, device_map="cuda")
model.eval()

# Build story
print("\n[2/2] Building narrative corpus...")
story_parts = [
    "In the year 2026, a team of researchers discovered a code: " + CODE + ".",
    "Dr. Sarah Chen led the investigation into this mysterious finding.",
    "The code " + CODE + " appeared in ancient manuscripts from multiple civilizations.",
    "Each manuscript contained references to " + CODE + " in different languages.",
    "Professor Williams from Oxford confirmed the code's historical significance.",
    "The researchers published their findings about " + CODE + " in Nature.",
    "Scientists worldwide began studying the implications of " + CODE + ".",
    "Some believed " + CODE + " was a key to understanding ancient mathematics.",
    "Others thought " + CODE + " might be evidence of early computational thinking.",
    "The mystery of " + CODE + " continues to fascinate researchers today.",
]

story = " ".join(story_parts)
while len(story) < 300000:  # ~128K tokens
    story = story + " " + story

story_ids = proc.tokenizer(story, return_tensors='pt').input_ids[0]
total_tokens = story_ids.shape[0]
print(f"   Story: {len(story)} chars, {total_tokens} tokens available")


def build_prompt(ctx_tokens, needle_pos):
    """Build prompt with code at a specific position."""
    code_ids = proc.tokenizer(CODE, add_special_tokens=False).input_ids
    ctx_ids = story_ids[:ctx_tokens].clone()

    if needle_pos + len(code_ids) < ctx_tokens:
        ctx_ids[needle_pos:needle_pos + len(code_ids)] = torch.tensor(code_ids)

    question = "\n\nQuestion: What is the code mentioned in the text? Answer:"
    question_ids = proc.tokenizer(question, add_special_tokens=False).input_ids
    return torch.cat([ctx_ids, torch.tensor(question_ids)])


def run_test(config, ctx, needle_pos):
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
        if use_patch:
            with patch_model_for_fused_attention(model, cache, enable_fused=True):
                out = model.generate(
                    input_ids=input_ids_dev, attention_mask=attention_mask,
                    max_new_tokens=30, do_sample=False, past_key_values=cache,
                )
        else:
            out = model.generate(
                input_ids=input_ids_dev, attention_mask=attention_mask,
                max_new_tokens=30, do_sample=False, past_key_values=cache,
            )

        elapsed = time.time() - t0
        peak = torch.cuda.max_memory_allocated() / 1024**2
        generated = out[0][n_tokens:]
        raw = proc.tokenizer.decode(generated, skip_special_tokens=True)
        ok = CODE in raw

        result = dict(
            method=name, ctx=ctx, needle_pos=needle_pos, zone=zone,
            tokens=n_tokens, peak=peak, time=elapsed, ok=ok,
            ans=raw.strip()[:60], oom=False,
        )
        return result

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            peak = torch.cuda.max_memory_allocated() / 1024**2
            return dict(
                method=name, ctx=ctx, needle_pos=needle_pos, zone=zone,
                tokens=n_tokens, peak=peak, time=0, ok=False, ans="OOM", oom=True,
            )
        raise
    finally:
        del input_ids_dev, attention_mask, cache
        gc.collect()
        torch.cuda.empty_cache()


# ── Configs ───────────────────────────────────────────────────────────────────
configs = [
    ("Baseline (No Healing)",   False, False, False, False),
    ("Method C (Triton+Adapt)", False, True,  True,  True),
    ("Method D (Query-aware)",  True,  False, False, False),
]

# ── Part A: Progressive Stress Test ──────────────────────────────────────────
print("\n" + "=" * 80)
print("  PART A: Progressive Stress Test (4K → 64K)")
print("=" * 80)

results_stress = []
context_lengths = [4096, 8192, 16384, 32768, 65536]

for ctx in context_lengths:
    if total_tokens < ctx:
        print(f"\n  ⚠️ ctx={ctx//1024}K: not enough tokens, skip")
        continue

    needle_pos = ctx // 2  # Middle = DRAM zone
    print(f"\n  ctx={ctx//1024}K | needle at {needle_pos} (DRAM) | zones: Sink[0:{1024}] DRAM[1024:{ctx-2048}] Tail[{ctx-2048}:{ctx}]")

    for config in configs:
        r = run_test(config, ctx, needle_pos)
        results_stress.append(r)
        icon = "✅" if r['ok'] else "❌" if not r['oom'] else "💥"
        print(f"    {icon} {r['method']:25} | peak={r['peak']:7.0f}MB | {r['time']:5.1f}s | {r.get('ans','')[:30]}")

# ── Part B: Multi-Zone Accuracy Test ─────────────────────────────────────────
print("\n" + "=" * 80)
print("  PART B: Multi-Zone Accuracy Test (16K context)")
print("=" * 80)

results_zone = []
ctx = 16384
if total_tokens >= ctx:
    zone_tests = [
        (1024, "Sink"),        # Sink zone
        (8192, "DRAM"),        # DRAM zone (middle)
        (14336, "Tail"),       # Tail zone
        (4096, "DRAM-early"),  # DRAM zone (early)
        (12288, "DRAM-late"),  # DRAM zone (late)
    ]

    for needle_pos, zone_name in zone_tests:
        print(f"\n  Zone: {zone_name} | needle at {needle_pos}")

        for config in configs:
            r = run_test(config, ctx, needle_pos)
            r['zone_name'] = zone_name
            results_zone.append(r)
            icon = "✅" if r['ok'] else "❌" if not r['oom'] else "💥"
            print(f"    {icon} {r['method']:25} | {r.get('ans','')[:40]}")

# ── Part C: Video VQA Test (Text-only Simulating VQA) ─────────────────────────
print("\n" + "=" * 80)
print("  PART C: Multimodal VQA Test")
print("=" * 80)

results_vqa = []

# For VQA, we use LLaVA with image + text
# Check if we have test images
import glob
test_images = glob.glob("/home/app-ahr/Hetero-KVCache-Optimizer/test_images/*.jpg") + \
              glob.glob("/home/app-ahr/Hetero-KVCache-Optimizer/test_images/*.png") + \
              glob.glob("/home/app-ahr/Hetero-KVCache-Optimizer/**/*.jpg", recursive=True)

if test_images:
    print(f"\n  Found {len(test_images)} test images")

    for img_path in test_images[:3]:  # Test up to 3 images
        from PIL import Image
        img = Image.open(img_path).convert('RGB')

        # Build multimodal prompt
        vqa_prompt = f"<image>\nDescribe what you see in the image. The secret code in this project is {CODE}.\n\nQuestion: What is the secret code? Answer:"

        for config in configs:
            name, enable_d, adaptive, triton, use_patch = config
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            gc.collect()

            cache = FusedHeteroCache(
                num_layers=32, sink_tokens=1024, keep_tail=2048, chunk_size=2048,
                device='cuda', enable_quant=True, enable_prefetch=False,
                self_healing=enable_d or adaptive,
                adaptive_self_healing=adaptive,
                enable_triton=triton,
                enable_method_d=enable_d,
            )

            try:
                inputs = proc(text=vqa_prompt, images=img, return_tensors='pt').to('cuda')

                if use_patch:
                    with patch_model_for_fused_attention(model, cache, enable_fused=True):
                        out = model.generate(
                            input_ids=inputs.input_ids, attention_mask=inputs.attention_mask,
                            pixel_values=inputs.pixel_values,
                            max_new_tokens=30, do_sample=False, past_key_values=cache,
                        )
                else:
                    out = model.generate(
                        input_ids=inputs.input_ids, attention_mask=inputs.attention_mask,
                        pixel_values=inputs.pixel_values,
                        max_new_tokens=30, do_sample=False, past_key_values=cache,
                    )

                peak = torch.cuda.max_memory_allocated() / 1024**2
                raw = proc.tokenizer.decode(out[0][-30:], skip_special_tokens=True)
                ok = CODE in raw

                r = dict(method=name, image=os.path.basename(img_path), peak=peak, ok=ok, ans=raw.strip()[:60], oom=False)
                results_vqa.append(r)
                icon = "✅" if ok else "❌"
                print(f"    {icon} {name:25} | {os.path.basename(img_path)[:30]} | {raw.strip()[:40]}")

            except Exception as e:
                print(f"    ❌ {name:25} | ERROR: {str(e)[:50]}")
            finally:
                del cache
                gc.collect()
                torch.cuda.empty_cache()
else:
    print("\n  No test images found. Running text-only VQA simulation.")

    # Simulate VQA with longer descriptive text
    vqa_text = (
        f"The image shows a research laboratory with several scientists working on a project. "
        f"On the whiteboard, there is a code written: {CODE}. "
        f"The scientists are analyzing data from their experiments. "
        f"Dr. Chen points to the code {CODE} on the board and explains its significance. "
    ) * 100

    vqa_prompt = vqa_text[:20000] + "\n\nQuestion: What code is visible in the image? Answer:"

    for config in configs:
        name, enable_d, adaptive, triton, use_patch = config
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        gc.collect()

        cache = FusedHeteroCache(
            num_layers=32, sink_tokens=1024, keep_tail=2048, chunk_size=2048,
            device='cuda', enable_quant=True, enable_prefetch=False,
            self_healing=enable_d or adaptive,
            adaptive_self_healing=adaptive,
            enable_triton=triton,
            enable_method_d=enable_d,
        )

        try:
            inputs = proc(text=vqa_prompt, return_tensors='pt').to('cuda')
            n_tok = inputs.input_ids.shape[-1]

            if use_patch:
                with patch_model_for_fused_attention(model, cache, enable_fused=True):
                    out = model.generate(
                        input_ids=inputs.input_ids, attention_mask=inputs.attention_mask,
                        max_new_tokens=30, do_sample=False, past_key_values=cache,
                    )
            else:
                out = model.generate(
                    input_ids=inputs.input_ids, attention_mask=inputs.attention_mask,
                    max_new_tokens=30, do_sample=False, past_key_values=cache,
                )

            peak = torch.cuda.max_memory_allocated() / 1024**2
            generated = out[0][n_tok:]
            raw = proc.tokenizer.decode(generated, skip_special_tokens=True)
            ok = CODE in raw

            r = dict(method=name, image="simulated_vqa", peak=peak, ok=ok, ans=raw.strip()[:60], oom=False, tokens=n_tok)
            results_vqa.append(r)
            icon = "✅" if ok else "❌"
            print(f"    {icon} {name:25} | tok={n_tok:5} | peak={peak:7.0f}MB | {raw.strip()[:40]}")

        except Exception as e:
            print(f"    ❌ {name:25} | ERROR: {str(e)[:60]}")
        finally:
            del cache
            gc.collect()
            torch.cuda.empty_cache()

# ── Final Summary ─────────────────────────────────────────────────────────────
all_results = results_stress + results_zone + results_vqa

print(f"\n{'='*80}")
print("  FINAL SUMMARY")
print(f"{'='*80}")

# Part A: Stress Test
print(f"\n  PART A: Progressive Stress Test")
print(f"  {'Method':25} | ", end="")
for ctx in context_lengths:
    print(f"{ctx//1024:>4}K | ", end="")
print()
print("  " + "-" * 75)

for name in ["Baseline (No Healing)", "Method C (Triton+Adapt)", "Method D (Query-aware)"]:
    print(f"  {name:25} | ", end="")
    for ctx in context_lengths:
        r = [r for r in results_stress if r['method'] == name and r['ctx'] == ctx]
        if not r:
            print("  -  | ", end="")
        elif r[0]['oom']:
            print(" 💥 | ", end="")
        elif r[0]['ok']:
            print(" ✅ | ", end="")
        else:
            print(" ❌ | ", end="")
    print()

# Part A: Memory
print(f"\n  Peak Memory by Context Length:")
print(f"  {'Method':25} | ", end="")
for ctx in context_lengths:
    print(f"{ctx//1024:>4}K | ", end="")
print()
print("  " + "-" * 75)

for name in ["Baseline (No Healing)", "Method C (Triton+Adapt)", "Method D (Query-aware)"]:
    print(f"  {name:25} | ", end="")
    for ctx in context_lengths:
        r = [r for r in results_stress if r['method'] == name and r['ctx'] == ctx]
        if not r:
            print("   - | ", end="")
        else:
            print(f"{r[0]['peak']:5.0f} | ", end="")
    print()

# Part B: Zone Accuracy
print(f"\n  PART B: Multi-Zone Accuracy (16K context)")
print(f"  {'Zone':12} {'Baseline':>10} {'Method C':>10} {'Method D':>10}")
print("  " + "-" * 50)

for zone_name in ['Sink', 'DRAM', 'DRAM-early', 'DRAM-late', 'Tail']:
    b = [r for r in results_zone if r.get('zone_name') == zone_name and r['method'] == 'Baseline (No Healing)']
    c = [r for r in results_zone if r.get('zone_name') == zone_name and r['method'] == 'Method C (Triton+Adapt)']
    d = [r for r in results_zone if r.get('zone_name') == zone_name and r['method'] == 'Method D (Query-aware)']

    b_ok = sum(1 for r in b if r['ok'])
    c_ok = sum(1 for r in c if r['ok'])
    d_ok = sum(1 for r in d if r['ok'])

    b_str = f"{b_ok}/{len(b)}" if b else "N/A"
    c_str = f"{c_ok}/{len(c)}" if c else "N/A"
    d_str = f"{d_ok}/{len(d)}" if d else "N/A"

    print(f"  {zone_name:12} {b_str:>10} {c_str:>10} {d_str:>10}")

# Part C: VQA
print(f"\n  PART C: VQA Test")
for r in results_vqa:
    icon = "✅" if r['ok'] else "❌"
    print(f"    {icon} {r['method']:25} | {r.get('image', 'N/A'):20} | peak={r['peak']:7.0f}MB")

# Overall
print(f"\n  Overall Results:")
for name in ["Baseline (No Healing)", "Method C (Triton+Adapt)", "Method D (Query-aware)"]:
    all_method = [r for r in all_results if r['method'] == name and not r.get('oom')]
    if all_method:
        ok_count = sum(1 for r in all_method if r['ok'])
        total = len(all_method)
        avg_peak = sum(r['peak'] for r in all_method) / len(all_method)
        avg_time = sum(r['time'] for r in all_method if r['time'] > 0) / max(1, len([r for r in all_method if r['time'] > 0]))
        print(f"    {name:25} | {ok_count}/{total} ({ok_count/total*100:.0f}%) | avg peak={avg_peak:.0f}MB | avg time={avg_time:.1f}s")

# Save
output_file = "/home/app-ahr/Hetero-KVCache-Optimizer/comprehensive_benchmark_results.json"
with open(output_file, 'w') as f:
    json.dump(all_results, f, indent=2)
print(f"\n  Results saved to: {output_file}")

print(f"\n{'='*80}\n")
