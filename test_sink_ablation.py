#!/usr/bin/env python3
"""
Verify that larger sink_tokens fixes the accuracy issue.
The hypothesis: with sink=64, only 64/576 image tokens are preserved,
causing the model to lose image information during decode.
"""

import torch
import sys
import os
from PIL import Image
import numpy as np

sys.path.insert(0, '/home/app-ahr/Hetero-KVCache-Optimizer/src')

from transformers import AutoProcessor, LlavaForConditionalGeneration
from core.engine_wrapper import FusedHeteroCache

print("=" * 70)
print("Sink Size Ablation: 验证 sink_tokens 对准确率的影响")
print("=" * 70)

# Load model
model_path = "/home/app-ahr/.cache/huggingface/hub/models--llava-hf--llava-1.5-7b-hf/snapshots"
snapshots = [d for d in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, d))]
latest = sorted(snapshots)[-1]
model_path = os.path.join(model_path, latest)

processor = AutoProcessor.from_pretrained(model_path)
model = LlavaForConditionalGeneration.from_pretrained(
    model_path, torch_dtype=torch.float16, device_map="cuda"
)
model.eval()

# Create test image (red circle on white background)
img_arr = np.ones((224, 224, 3), dtype=np.uint8) * 255
y, x = np.ogrid[:224, :224]
mask = (x - 112)**2 + (y - 112)**2 <= 50**2
img_arr[mask] = [255, 0, 0]
test_image = Image.fromarray(img_arr)

question = "What color is the circle in the image?"
expected = "red"

def normalize(text):
    return text.lower().strip().replace(".", "").replace(",", "").replace("!", "")

def run_test(sink_tokens, ctx_len, use_self_healing=False):
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    padding = "The quick brown fox jumps over the lazy dog. " * max(1, ctx_len // 12)
    prompt = f"USER: <image>\n{padding}\n{question}\nASSISTANT:"
    inputs = processor(text=prompt, images=test_image, return_tensors='pt').to('cuda')
    num_tokens = inputs.input_ids.shape[-1]

    cache = FusedHeteroCache(
        num_layers=32,
        sink_tokens=sink_tokens,
        keep_tail=2048,
        chunk_size=2048,
        device='cuda',
        enable_quant=True,
        enable_triton=True if use_self_healing else False,
        self_healing=use_self_healing,
        adaptive_self_healing=use_self_healing,
    )

    try:
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs.input_ids,
                pixel_values=inputs.pixel_values,
                attention_mask=inputs.attention_mask,
                max_new_tokens=30,
                do_sample=False,
                past_key_values=cache,
            )

        response = processor.decode(outputs[0], skip_special_tokens=True)
        answer = response.split("ASSISTANT:")[-1].strip()
        correct = expected in normalize(answer)
        peak_mb = torch.cuda.max_memory_allocated() / 1024**2
        status = "✅" if correct else "❌"
        print(f"  {status} sink={sink_tokens:>4} ctx={ctx_len:>5} | {num_tokens:>5} tokens | {peak_mb:>8.0f} MB | {answer[:50]}")
        return correct
    except Exception as e:
        peak_mb = torch.cuda.max_memory_allocated() / 1024**2
        print(f"  ❌ sink={sink_tokens:>4} ctx={ctx_len:>5} | {num_tokens:>5} tokens | {peak_mb:>8.0f} MB | ERR: {str(e)[:60]}")
        return False
    finally:
        del inputs, cache
        torch.cuda.empty_cache()

# First get baseline
print("\n[Baseline - 标准 KV cache]")
torch.cuda.reset_peak_memory_stats()
torch.cuda.empty_cache()
prompt = f"USER: <image>\n{question}\nASSISTANT:"
inputs = processor(text=prompt, images=test_image, return_tensors='pt').to('cuda')
with torch.no_grad():
    outputs = model.generate(
        input_ids=inputs.input_ids,
        pixel_values=inputs.pixel_values,
        attention_mask=inputs.attention_mask,
        max_new_tokens=30,
        do_sample=False,
    )
response = processor.decode(outputs[0], skip_special_tokens=True)
answer = response.split("ASSISTANT:")[-1].strip()
print(f"  Baseline answer: {answer}")
del inputs
torch.cuda.empty_cache()

print("\n[测试 1: 不同 sink_tokens 大小 @ ctx=2000 (2591 tokens)]")
print("  假设：image tokens ≈ 576, sink=64 太小无法覆盖所有 image tokens")
print("-" * 70)

for sink in [64, 256, 576, 768, 1024]:
    run_test(sink, 2000)

print("\n[测试 2: self_healing 能否弥补 sink 不足?]")
print("-" * 70)

for sink in [64, 576]:
    run_test(sink, 2000, use_self_healing=True)

print("\n[测试 3: 不同上下文长度 @ sink=768]")
print("-" * 70)

for ctx in [0, 500, 1000, 2000, 4000]:
    run_test(768, ctx)

print("\n" + "=" * 70)
print("结论：如果 sink≥576 准确率恢复正常，说明问题是 sink 太小导致 image tokens 丢失")
print("=" * 70)
