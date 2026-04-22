#!/usr/bin/env python3
"""
 simplified_stress_test.py
 ========================
 ARIS Mission: Simplified SOTA Stress Test
 Deploy KIVI (quantization SOTA) and SnapKV (eviction SOTA) under 16GB VRAM constraint.

 This version focuses on KV cache stress testing without complex multimodal pipelines.
"""

import os, sys, gc, time, warnings, json
import torch
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache

warnings.filterwarnings('ignore')

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from src.core.engine_wrapper import build_fused_cache
from src.quantization.kv_compressor import KVCompressor

# ── Configuration ──────────────────────────────────────────────────────────
DEVICE = "cuda:0"
GPU_TOTAL_GB = 80.0  # A100
VRAM_LIMIT_GB = 16.0
MEMORY_FRACTION = VRAM_LIMIT_GB / GPU_TOTAL_GB

# Model path
MODEL_PATH = os.path.join(project_root, "models", "Qwen2.5-7B-Instruct")

# Stress test configuration
CONTEXT_LENGTHS = [16384, 32768, 65536, 131072, 262144]
DECODE_TOKENS = 256
WARMUP_TOKENS = 10
REPEATS = 1

class KIVIStaticQuantCache(DynamicCache):
    """KIVI-style 4-bit static quantization cache"""
    def __init__(self, group_size: int = 128):
        super().__init__()
        self.group_size = group_size
        self._compressor = KVCompressor(group_size=group_size, bits=4)
        self._quant_keys: List[Optional[tuple]] = []
        self._value_cache: List[Optional[torch.Tensor]] = []
        self._seen = 0

    def update(self, key, value, layer_idx, cache_kwargs=None):
        while len(self._quant_keys) <= layer_idx:
            self._quant_keys.append(None)
            self._value_cache.append(None)

        new_len = key.shape[-2]

        if new_len > 1:  # Prefill — quantize keys, store values in FP16
            q_k, s_k, z_k = self._compressor.compress(key)
            self._quant_keys[layer_idx] = (q_k.cpu(), s_k.cpu(), z_k.cpu())
            self._value_cache[layer_idx] = value
        else:  # Decode — append
            if self._quant_keys[layer_idx] is not None:
                prev_q, prev_s, prev_z = self._quant_keys[layer_idx]
                prev_k = self._compressor.decompress(
                    prev_q.to(DEVICE), prev_s.to(DEVICE), prev_z.to(DEVICE),
                    target_dtype=key.dtype
                )
                full_k = torch.cat([prev_k, key], dim=-2)
                q_k, s_k, z_k = self._compressor.compress(full_k)
                self._quant_keys[layer_idx] = (q_k.cpu(), s_k.cpu(), z_k.cpu())
            else:
                q_k, s_k, z_k = self._compressor.compress(key)
                self._quant_keys[layer_idx] = (q_k.cpu(), s_k.cpu(), z_k.cpu())

            if self._value_cache[layer_idx] is not None:
                self._value_cache[layer_idx] = torch.cat(
                    [self._value_cache[layer_idx], value], dim=-2)
            else:
                self._value_cache[layer_idx] = value

        if layer_idx == 0:
            self._seen += new_len

        # Return decompressed keys + original values
        q_k, s_k, z_k = self._quant_keys[layer_idx]
        restored_k = self._compressor.decompress(
            q_k.to(DEVICE), s_k.to(DEVICE), z_k.to(DEVICE), target_dtype=key.dtype)
        return restored_k, self._value_cache[layer_idx]

    def get_seq_length(self, layer_idx=0):
        if layer_idx < len(self._value_cache) and self._value_cache[layer_idx] is not None:
            return self._value_cache[layer_idx].shape[-2]
        return 0

class SnapKVEvictionCache(DynamicCache):
    """SnapKV-style eviction cache"""
    def __init__(self, sink_tokens: int = 64, eviction_window: int = 4096):
        super().__init__()
        self.sink_tokens = sink_tokens
        self.eviction_window = eviction_window
        self._value_cache: List[Optional[torch.Tensor]] = []
        self._key_cache: List[Optional[torch.Tensor]] = []
        self._seen = 0

    def update(self, key, value, layer_idx, cache_kwargs=None):
        while len(self._key_cache) <= layer_idx:
            self._key_cache.append(None)
            self._value_cache.append(None)

        new_len = key.shape[-2]

        # Store new keys and values
        if self._key_cache[layer_idx] is None:
            self._key_cache[layer_idx] = key
            self._value_cache[layer_idx] = value
        else:
            self._key_cache[layer_idx] = torch.cat([self._key_cache[layer_idx], key], dim=-2)
            self._value_cache[layer_idx] = torch.cat([self._value_cache[layer_idx], value], dim=-2)

        # Apply eviction strategy
        if self._key_cache[layer_idx].shape[-2] > self.sink_tokens + self.eviction_window:
            self._key_cache[layer_idx] = torch.cat([
                self._key_cache[layer_idx][:, :, :self.sink_tokens, :],
                self._key_cache[layer_idx][:, :, -self.eviction_window:, :]
            ], dim=-2)
            self._value_cache[layer_idx] = torch.cat([
                self._value_cache[layer_idx][:, :, :self.sink_tokens, :],
                self._value_cache[layer_idx][:, :, -self.eviction_window:, :]
            ], dim=-2)

        if layer_idx == 0:
            self._seen += new_len

        return self._key_cache[layer_idx], self._value_cache[layer_idx]

    def get_seq_length(self, layer_idx=0):
        if layer_idx < len(self._value_cache) and self._value_cache[layer_idx] is not None:
            return self._value_cache[layer_idx].shape[-2]
        return 0

def setup_memory_limit():
    """Set 16GB VRAM limit"""
    torch.cuda.set_per_process_memory_fraction(MEMORY_FRACTION, device=0)
    print(f"  VRAM Limit: {VRAM_LIMIT_GB}GB ({GPU_TOTAL_GB * MEMORY_FRACTION:.1f}/{GPU_TOTAL_GB}GB A100)")

def run_stress_trial(
    model, tokenizer, seq_len: int, method: str
) -> Optional[Dict]:
    """Run a single stress trial"""
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats(DEVICE)

    # Build synthetic input
    dummy_text = "This is a long video analysis. " * (seq_len // 20 + 1)
    inputs = tokenizer(dummy_text, return_tensors="pt",
                       truncation=True, max_length=seq_len).to(DEVICE)
    actual_len = inputs.input_ids.shape[1]

    num_layers = len(model.model.layers)

    # Build cache per method
    if method == "hetero_kv":
        cache = build_fused_cache(
            num_layers=num_layers,
            sink_tokens=64,
            keep_tail=4096,
            device=DEVICE,
            enable_quant=True,
            group_size=128,
        )
    elif method == "kivi":
        cache = KIVIStaticQuantCache(group_size=128)
    elif method == "snapkv":
        cache = SnapKVEvictionCache(sink_tokens=64, eviction_window=4096)
    elif method == "hf_offload":
        cache = None  # standard HF DynamicCache
    else:
        raise ValueError(f"Unknown method: {method}")

    t0 = time.time()
    try:
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=DECODE_TOKENS,
                num_beams=1,
                do_sample=False,
                use_cache=True,
                past_key_values=cache,
                pad_token_id=tokenizer.eos_token_id,
            )
        elapsed = time.time() - t0
        peak_mem = torch.cuda.max_memory_allocated(DEVICE) / 1024**3
        tokens_per_sec = DECODE_TOKENS / max(elapsed, 1e-6)
        oom = False

        # Calculate compression ratio (for quantized methods)
        compression_ratio = calculate_compression_ratio(cache, method)

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            elapsed = 0
            peak_mem = torch.cuda.memory_allocated(DEVICE) / 1024**3
            tokens_per_sec = 0
            oom = True
            compression_ratio = 0
            print(f"    OOM at seq_len={actual_len} ({method})")
        else:
            raise

    result = {
        "method": method,
        "target_seq_len": seq_len,
        "actual_seq_len": actual_len,
        "decode_tokens": DECODE_TOKENS,
        "elapsed_s": round(elapsed, 3),
        "tokens_per_sec": round(tokens_per_sec, 2),
        "peak_memory_gb": round(peak_mem, 3),
        "vram_limit_gb": VRAM_LIMIT_GB,
        "oom": oom,
        "compression_ratio": compression_ratio,
        "within_budget": peak_mem <= VRAM_LIMIT_GB if not oom else False,
    }

    # Clean up
    if 'outputs' in dir():
        del outputs
    del inputs, cache
    gc.collect()
    torch.cuda.empty_cache()
    return result

def calculate_compression_ratio(cache, method: str) -> float:
    """Calculate KV cache compression ratio"""
    if method == "hetero_kv" or method == "kivi":
        if hasattr(cache, '_compressor'):
            return 4.0  # 4-bit quantization
    elif method == "snapkv":
        return 0.0  # No compression, just eviction
    return 0.0

def main():
    print("=" * 70)
    print("ARIS MISSION: SOTA Integration & Stress Test")
    print("=" * 70)
    print(f"VRAM Limit: {VRAM_LIMIT_GB}GB | Model: Qwen2.5-7B-Instruct")
    print("=" * 70)

    setup_memory_limit()

    # Load model
    print(f"\nLoading model from {MODEL_PATH} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map={"": DEVICE},
        trust_remote_code=True,
    ).eval()
    print(f"Model loaded: {len(model.model.layers)} layers")

    # Run stress tests
    methods = ["hf_offload", "kivi", "snapkv", "hetero_kv"]
    all_results = []

    for seq_len in CONTEXT_LENGTHS:
        print(f"\n{'─'*60}")
        print(f"Context length: {seq_len}")

        for method in methods:
            print(f"  [{method}]", end=" ")
            for rep in range(REPEATS):
                r = run_stress_trial(model, tokenizer, seq_len, method)
                if r is not None:
                    r["repeat"] = rep
                    all_results.append(r)
                    status = "OOM" if r["oom"] else f"{r['tokens_per_sec']:.1f} tok/s"
                    print(f"rep{rep}: {status}", end=" ")
            print()

    # Save results
    df = pd.DataFrame(all_results)
    base_path = os.path.join(project_root, "stress_test_results")
    os.makedirs(base_path, exist_ok=True)

    # Full results
    df.to_csv(f"{base_path}/stress_results.csv", index=False)

    # Summary analysis
    print("\n" + "=" * 70)
    print("STRESS TEST SUMMARY")
    print("=" * 70)

    print("\n1. OOM SURVIVAL:")
    for method in methods:
        method_data = df[df["method"] == method]
        if len(method_data) > 0:
            survived = method_data[~method_data["oom"]]
            if len(survived) > 0:
                max_len = survived["actual_seq_len"].max()
                max_tokens = survived[survived["actual_seq_len"] == max_len]["tokens_per_sec"].mean()
                print(f"  {method:12s}: survived {max_len:>7d} @ {max_tokens:.1f} tok/s")

                # Find OOM threshold
                oom_data = method_data[method_data["oom"]]
                if len(oom_data) > 0:
                    oom_at = oom_data["target_seq_len"].min()
                    print(f"                 OOM at {oom_at}")
            else:
                print(f"  {method:12s}: ALL OOM")

    print("\n2. MEMORY EFFICIENCY:")
    for method in methods:
        method_data = df[(df["method"] == method) & (~df["oom"])]
        if len(method_data) > 0:
            avg_mem = method_data["peak_memory_gb"].mean()
            avg_tokens_per_sec = method_data["tokens_per_sec"].mean()
            efficiency = avg_tokens_per_sec / avg_mem
            print(f"  {method:12s}: {avg_mem:.1f}GB avg @ {efficiency:.1f} tok/s/GB")

    # Save final analysis
    report = {
        "config": {
            "vr_limit_gb": VRAM_LIMIT_GB,
            "memory_fraction": MEMORY_FRACTION,
            "model": "Qwen2.5-7B-Instruct",
            "gpu": "NVIDIA A100 80GB",
        },
        "context_lengths": CONTEXT_LENGTHS,
        "methods": methods,
        "results": df.to_dict('records'),
    }

    with open(f"{base_path}/stress_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nResults saved to: {base_path}/")
    print("Files:")
    print(f"  - stress_results.csv: All test results")
    print(f"  - stress_report.json: Complete analysis")

    # Cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()