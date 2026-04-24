#!/usr/bin/env python3
"""
Quick Llama-3 availability check
"""

import torch
import sys
sys.path.insert(0, '.')
from tests.llama3_generalization_benchmark import set_memory_limit

print("\n1. Checking if Llama-3 is available...")
from transformers import AutoTokenizer, AutoModelForCausalLM

# Try different possible paths
paths = [
    'models/Llama-3.1-8B-Instruct',
    '/home/app-ahr/.cache/huggingface/hub/models--NousResearch--Meta-Llama-3.1-8B-Instruct',
    '/home/app-ahr/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-8B-Instruct',
    'NousResearch/Meta-Llama-3.1-8B-Instruct',
    'meta-llama/Meta-Llama-3.1-8B-Instruct'
]

model_path = None
for path in paths:
    try:
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True)
        print(f"  ✓ Found tokenizer at: {path}")
        model_path = path
        break
    except Exception as e:
        print(f"  ✗ {path}: {str(e)[:100]}")

if model_path:
    print(f"\nUsing model path: {model_path}")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map='cuda:0',
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
        )
        model.eval()
        print("  ✓ Model loaded successfully!")

        # Basic model info
        print(f"\nModel info:")
        print(f"  Type: {type(model)}")
        print(f"  Device: {next(model.parameters()).device}")
        num_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {num_params / 1e9:.2f}B")

        # Try a simple generation
        print("\n2. Testing basic generation...")
        inputs = tokenizer("Hello, this is a test", return_tensors="pt").to('cuda')
        outputs = model.generate(**inputs, max_new_tokens=10)
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Generated: {generated}")

        print("\n✓ Llama-3 is ready for benchmark!")

    except Exception as e:
        print(f"  ✗ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
else:
    print("\n✗ No working Llama-3 installation found")
    sys.exit(1)