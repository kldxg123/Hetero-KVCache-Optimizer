#!/usr/bin/env python3
"""
run_baseline_throughput.py
==========================
Phase 1.2: Throughput benchmark comparing Hetero-KV, native HF offload, and
KIVI (4-bit static quantization) from 8K to 128K context lengths.

Measures tokens/s and records OOM failures under a 24 GB VRAM budget.
Output: results_throughput.csv
"""

import os, sys, gc, time, warnings
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
MODEL_PATH = os.path.join(project_root, "models", "Qwen2.5-7B-Instruct")
DEVICE = "cuda:0"
VRAM_BUDGET_GB = 24.0
CONTEXT_LENGTHS = [2048, 4096, 8192, 16384, 32768, 65536, 131072]
DECODE_TOKENS = 64           # tokens to generate for throughput measurement
WARMUP_TOKENS = 4
REPEATS = 2


def check_vram() -> float:
    return torch.cuda.memory_allocated(DEVICE) / 1024**3


class KIVIStaticQuantCache(DynamicCache):
    """
    Simplified KIVI-style 4-bit static quantization cache.
    Key: INT4 group-wise quant;  Value: FP16 (as in KIVI paper).
    """

    def __init__(self, group_size: int = 128):
        super().__init__()
        self.group_size = group_size
        self._compressor = KVCompressor(group_size=group_size, bits=4)
        self._quant_keys: List[Optional[tuple]] = []  # (q, scales, zps)
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
                # Decompress, append, re-quantize (simplified KIVI approach)
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

        # For attention, return decompressed keys + original values
        q_k, s_k, z_k = self._quant_keys[layer_idx]
        restored_k = self._compressor.decompress(
            q_k.to(DEVICE), s_k.to(DEVICE), z_k.to(DEVICE), target_dtype=key.dtype)
        return restored_k, self._value_cache[layer_idx]

    def get_seq_length(self, layer_idx=0):
        if layer_idx < len(self._value_cache) and self._value_cache[layer_idx] is not None:
            return self._value_cache[layer_idx].shape[-2]
        return 0


def run_throughput_trial(
    model, tokenizer, seq_len: int, method: str
) -> Optional[Dict]:
    """Run a single throughput trial at a given context length and method."""
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats(DEVICE)

    # Build synthetic input of the right length
    dummy_text = "The quick brown fox jumps over the lazy dog. " * (seq_len // 10 + 1)
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

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            elapsed = 0
            peak_mem = check_vram()
            tokens_per_sec = 0
            oom = True
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
        "vram_budget_gb": VRAM_BUDGET_GB,
        "oom": oom,
        "within_budget": peak_mem <= VRAM_BUDGET_GB if not oom else False,
    }

    del inputs, outputs, cache
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    print("=" * 70)
    print("Phase 1.2: Throughput Benchmark (Hetero-KV vs HF Offload vs KIVI)")
    print(f"VRAM Budget: {VRAM_BUDGET_GB} GB | Decode tokens: {DECODE_TOKENS}")
    print("=" * 70)

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

    all_results = []
    methods = ["hf_offload", "kivi", "hetero_kv"]

    for seq_len in CONTEXT_LENGTHS:
        print(f"\n{'─'*60}")
        print(f"Context length: {seq_len}")

        for method in methods:
            print(f"  [{method}]", end=" ")
            for rep in range(REPEATS):
                r = run_throughput_trial(model, tokenizer, seq_len, method)
                if r is not None:
                    r["repeat"] = rep
                    all_results.append(r)
                    status = "OOM" if r["oom"] else f"{r['tokens_per_sec']:.1f} tok/s ({r['peak_memory_gb']:.1f}GB)"
                    print(f"rep{rep}: {status}", end="  ")
            print()

    # ── Save CSV ───────────────────────────────────────────────────────────
    df = pd.DataFrame(all_results)
    csv_path = os.path.join(project_root, "results_throughput.csv")
    df.to_csv(csv_path, index=False)

    # ── Print survival table ───────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SURVIVAL TABLE (max context before OOM)")
    print("=" * 70)
    for method in methods:
        sub = df[(df["method"] == method) & (~df["oom"])]
        if len(sub) > 0:
            max_len = sub["actual_seq_len"].max()
            max_tps = sub[sub["actual_seq_len"] == max_len]["tokens_per_sec"].mean()
            oom_at = df[(df["method"] == method) & (df["oom"])]["target_seq_len"].min()
            oom_str = f"OOM at {oom_at}" if not pd.isna(oom_at) else "No OOM"
            print(f"  {method:12s}: survived {max_len:>7d} tokens @ {max_tps:.1f} tok/s | {oom_str}")
        else:
            print(f"  {method:12s}: ALL OOM")

    print(f"\nCSV saved → {csv_path}")

    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()