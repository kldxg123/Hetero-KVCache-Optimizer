#!/usr/bin/env python3
"""
test_kv_compressor_fix.py
==========================
Regression test for the reshape bug in KVCompressor when the total number
of elements is not divisible by group_size (common in MLLM visual tokens).
"""

import torch
import sys
import os

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.quantization.kv_compressor import KVCompressor


def test_case(shape, group_size, dtype, device):
    compressor = KVCompressor(group_size=group_size, bits=4)
    tensor = torch.randn(shape, dtype=dtype, device=device)

    q, scales, zps = compressor.compress(tensor)
    restored = compressor.decompress(q, scales, zps, target_dtype=dtype)

    assert q.shape == tensor.shape, f"Quantized shape mismatch: {q.shape} vs {tensor.shape}"
    assert restored.shape == tensor.shape, f"Restored shape mismatch: {restored.shape} vs {tensor.shape}"
    assert scales.dim() == 1, f"Scales should be 1D, got {scales.dim()}D"
    assert zps.dim() == 1, f"Zero points should be 1D, got {zps.dim()}D"

    mse = compressor.mse_error(tensor, q, scales, zps)
    rel = compressor.relative_error(tensor, q, scales, zps)

    # Sanity bounds (asymmetric 4-bit is inherently lossy; 15% rel err is acceptable)
    assert mse < 0.1, f"MSE too high: {mse} for shape {shape}"
    assert rel < 15.0, f"Relative error too high: {rel}% for shape {shape}"

    return mse, rel


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running KVCompressor fix tests on {device}\n")

    test_shapes = [
        # Original working case: exact multiple of 128
        (1, 32, 16, 128),
        # MLLM common case: head_dim = 64 (not divisible by 128)
        (1, 4, 1984, 64),
        (1, 8, 512, 64),
        # Arbitrary seq_len not divisible by group_size
        (1, 8, 123, 128),
        (1, 8, 1, 128),
        (1, 8, 1999, 64),
        # Flatten-friendly but total numel not divisible by 128
        (1, 3, 7, 11),
    ]

    all_pass = True
    results = []

    for shape in test_shapes:
        for group_size in [64, 128]:
            for dtype in [torch.float16, torch.bfloat16]:
                try:
                    mse, rel = test_case(shape, group_size, dtype, device)
                    results.append((shape, group_size, dtype, "PASS", mse, rel))
                    print(f"  PASS | shape={shape} group={group_size} dtype={dtype} | MSE={mse:.6f} Rel={rel:.4f}%")
                except Exception as e:
                    all_pass = False
                    results.append((shape, group_size, dtype, "FAIL", str(e), ""))
                    print(f"  FAIL | shape={shape} group={group_size} dtype={dtype} | {e}")

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    passed = sum(1 for r in results if r[3] == "PASS")
    total = len(results)
    print(f"Passed: {passed}/{total}")

    if all_pass:
        print("\n[OK] KVCompressor reshape bug fixed — arbitrary-length 4-bit quantization is stable.")
        sys.exit(0)
    else:
        print("\n[ERROR] Some test cases failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
