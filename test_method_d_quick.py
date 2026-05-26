#!/usr/bin/env python3
"""Quick Method D vs C verification"""

import torch, time, sys
sys.path.insert(0, '/home/app-ahr/Hetero-KVCache-Optimizer/src')

from transformers import AutoProcessor, LlavaForConditionalGeneration
from core.engine_wrapper import FusedHeteroCache

os = __import__('os')
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

NEEDLE = "HETEROKV2026"

print("=" * 80)
print("  Quick Method D vs C Test")
print("=" * 80)

# Load model
print("\n[1/3] Loading model...")
mp = "/home/app-ahr/.cache/huggingface/hub/models--llava-hf--llava-1.5-7b-hf/snapshots"
snaps = sorted([d for d in os.listdir(mp) if os.path.isdir(os.path.join(mp, d))])
mp = os.path.join(mp, snaps[-1])
proc = AutoProcessor.from_pretrained(mp)
model = LlavaForConditionalGeneration.from_pretrained(mp, torch_dtype=torch.float16, device_map="cuda")
model.eval()

# Simple test: needle in DRAM zone
print("\n[2/3] Building test prompt...")
ctx_len = 4096
needle_pos = 2048

# Create simple text with needle
text = "The quick brown fox jumps over the lazy dog. " * 400  # ~15K chars
needle_str = f"The secret passcode is {NEEDLE}. Remember this!"
needle_char_pos = int(needle_pos * 4)  # Approximate
text = text[:needle_char_pos] + needle_str + text[needle_char_pos:]
text = text[:ctx_len * 4]

prompt = f"{text}\n\nQuestion: What is the secret passcode?\nAnswer: The secret passcode is"

# Calculate zones accurately
total_tokens = n_tokens
sink_zone = 1024
tail_zone = 2048
dram_zone = total_tokens - sink_zone - tail_zone

if needle_pos < sink_zone:
    zone = "Sink"
elif needle_pos >= total_tokens - tail_zone:
    zone = "Tail"
else:
    zone = "DRAM"

print(f"  Prompt length: {len(prompt)} chars")
print(f"  Total tokens: {total_tokens}")
print(f"  Zone distribution: Sink={sink_zone}, DRAM={dram_zone}, Tail={tail_zone}")
print(f"  Needle position: ~{needle_pos} tokens → Zone: {zone}")

# Test Method C
print("\n[3/3] Running tests...")
print("-" * 80)

results = []

for method_name, enable_d in [("Method C", False), ("Method D", True)]:
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    inputs = proc(text=prompt, return_tensors='pt').to('cuda')
    n_tokens = inputs.input_ids.shape[-1]

    cache = FusedHeteroCache(
        num_layers=32, sink_tokens=1024, keep_tail=2048, chunk_size=2048,
        device='cuda', enable_quant=True, enable_prefetch=True, enable_triton=True,
        self_healing=True, adaptive_self_healing=True,
        enable_method_d=enable_d,
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

        icon = "✅" if ok else "❌"
        print(f"{icon} {method_name:12} | tokens={n_tokens:4} | peak={peak:7.0f}MB | time={elapsed:4.1f}s | {ans[:50]}")

        results.append({
            'method': method_name,
            'ok': ok,
            'peak': peak,
            'time': elapsed,
            'ans': ans[:50]
        })

    except Exception as e:
        print(f"❌ {method_name:12} | ERROR: {str(e)[:60]}")
        results.append({'method': method_name, 'ok': False, 'error': str(e)})
    finally:
        del inputs, cache
        torch.cuda.empty_cache()

# Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

if len(results) == 2:
    c_ok = results[0]['ok']
    d_ok = results[1]['ok']

    if c_ok and d_ok:
        print("✅ Both methods succeeded - no difference in this test")
    elif d_ok and not c_ok:
        print("✅ Method D outperformed Method C")
    elif c_ok and not d_ok:
        print("⚠️  Method C outperformed Method D (unexpected)")
    else:
        print("❌ Both methods failed")

    if 'peak' in results[0] and 'peak' in results[1]:
        mem_diff = results[1]['peak'] - results[0]['peak']
        time_diff = results[1]['time'] - results[0]['time']
        print(f"\nMemory difference: {mem_diff:+.0f}MB")
        print(f"Time difference: {time_diff:+.1f}s")

print("=" * 80 + "\n")
