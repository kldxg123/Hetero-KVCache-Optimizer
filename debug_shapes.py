#!/usr/bin/env python3
"""
Debug script to print tensor shapes before and after eviction.
"""

import torch
import sys

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

# Monkey-patch manager._decode_update to print shapes
from memory.manager import HeteroKVManager

original_decode_update = HeteroKVManager._decode_update

def patched_decode_update(self, layer_idx, key_states, value_states, seq_offset=0):
    print(f"\n[DEBUG _decode_update] layer_idx={layer_idx}")
    print(f"  key_states shape: {key_states.shape}")
    print(f"  value_states shape: {value_states.shape}")

    # Check if tail exists
    if self._tail_k[layer_idx] is not None:
        print(f"  existing tail_k shape: {self._tail_k[layer_idx].shape}")
        print(f"  tail_k[:, 1:, :] shape: {self._tail_k[layer_idx][:, 1:, :].shape}")

    return original_decode_update(self, layer_idx, key_states, value_states, seq_offset)

HeteroKVManager._decode_update = patched_decode_update

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
            max_new_tokens=3,  # Only generate a few tokens
            do_sample=False,
            past_key_values=cache,
        )
except Exception as e:
    print(f"\n{'='*60}")
    print(f"❌ 错误: {e}")
    print(f"{'='*60}")
