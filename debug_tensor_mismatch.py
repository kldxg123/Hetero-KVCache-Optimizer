#!/usr/bin/env python3
"""
Debug script to pinpoint the exact location of the tensor mismatch error.
"""

import torch
import sys
import traceback

sys.path.insert(0, '/home/app-ahr/Hetero-KVCache-Optimizer/src')

from PIL import Image
import numpy as np
from transformers import AutoProcessor, LlavaForConditionalGeneration

print("[加载模型] LLaVA-1.5-7B...")
model_path = "/home/app-ahr/.cache/huggingface/hub/models--llava-hf--llava-1.5-7b-hf/snapshots"
import os
snapshots = [d for d in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, d))]
latest = sorted(snapshots)[-1]
model_path = os.path.join(model_path, latest)

processor = AutoProcessor.from_pretrained(model_path)
model = LlavaForConditionalGeneration.from_pretrained(
    model_path, torch_dtype=torch.float16, device_map="cuda"
)
model.eval()

# Create test image
img_arr = np.ones((224, 224, 3), dtype=np.uint8) * 255
y, x = np.ogrid[:224, :224]
mask = (x - 112)**2 + (y - 112)**2 <= 50**2
img_arr[mask] = [255, 0, 0]
test_image = Image.fromarray(img_arr)

# Build prompt with context that triggers eviction
context_len = 2000
question = "What color is the circle in the image?"
padding = "The quick brown fox jumps over the lazy dog. " * max(1, context_len // 12)
prompt = f"USER: <image>\n{padding}\n{question}\nASSISTANT:"

inputs = processor(text=prompt, images=test_image, return_tensors='pt').to('cuda')

print(f"Input shape: {inputs.input_ids.shape}")

# Initialize HeteroKV cache
from core.engine_wrapper import FusedHeteroCache
cache = FusedHeteroCache(
    num_layers=32, sink_tokens=64, keep_tail=2048,
    chunk_size=2048, device='cuda',
    enable_quant=True, enable_triton=False,
    self_healing=False, adaptive_self_healing=False,
)

print("\n[开始 generate] 预期会出现 tensor mismatch error...")

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
except Exception as e:
    print(f"\n{'='*60}")
    print(f"❌ 错误捕获: {type(e).__name__}")
    print(f"{'='*60}")
    print(f"错误信息: {e}")
    print(f"\n完整堆栈跟踪:")
    print(f"{'='*60}")
    traceback.print_exc()

    # Also print the shape of KV tensors for debugging
    if cache._manager is not None:
        print(f"\n\n[调试信息] Manager状态:")
        print(f"  Sink K shape: {cache._manager._sink_k[0].shape if cache._manager._sink_k[0] is not None else None}")
        print(f"  Tail K shape: {cache._manager._tail_k[0].shape if cache._manager._tail_k[0] is not None else None}")
        print(f"  HH K shape: {cache._manager._heavyhitter_k[0].shape if cache._manager._heavyhitter_k[0] is not None else None}")
