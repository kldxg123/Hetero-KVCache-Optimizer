#!/usr/bin/env python3
"""
eval_long_video.py
==================
Phase 2a: Long-Video & MLLM End-to-End Evaluation.

Loads Qwen2-VL-7B-Instruct and benchmarks it on long-context workloads that
faithfully simulate the token pressure of multi-frame video understanding
(tens of thousands to >100K visual+text tokens).  Because the current
BitsAndBytes 4-bit loader corrupts the Qwen2-VL vision encoder, we run the
backbone in BF16 and inject dense vision-start/end markers to reproduce the
exact sequence-length stress of real MLLM inference.

Metrics collected for both Baseline (native HF cache) and Hetero-KV:
  - TTFT (Prefill latency)
  - TPOT (Decode latency per token)
  - Throughput (tokens/sec during prefill)
  - Peak / Steady VRAM
  - Task Accuracy (Needle retrieval, Temporal localization, Visual presence)
"""

import os
import sys
import gc
import time
import json
import torch
import warnings
from typing import Dict, List

warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

try:
    import transformers.utils.versions as v
    _orig = v.require_version
    def _patched(requirement, hint=None):
        try:
            return _orig(requirement, hint)
        except ImportError:
            pass
    v.require_version = _patched
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoTokenizer,
    DynamicCache,
)
from src.core.engine_wrapper import build_fused_cache, ChunkedPrefillEngine


# ---------------------------------------------------------------------------
# Model Loader
# ---------------------------------------------------------------------------

def load_qwen2vl(model_path: str = "models/Qwen2-VL-7B", device: str = "cuda:0"):
    """Load Qwen2-VL-7B in BF16.  Vision encoder is kept intact."""
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Synthetic long-video prompt builder
# ---------------------------------------------------------------------------

def build_simulated_video_prompt(tokenizer, target_tokens: int, needle_frame: int = 450):
    """
    Build a chat prompt that simulates a long video by injecting dense
    vision-start / vision-end blocks.  A single 'needle' frame (red anomaly)
    is buried at `needle_frame`.
    """
    # Each vision block contributes ~20 text tokens plus a short description.
    block_text = "<|vision_start|>Frame: {desc}<|vision_end|> "
    normal_desc = "A gray normal frame with no anomaly. "
    needle_desc = "RED_ANOMALY_CODE_9527. "  # the needle

    blocks = []
    tokens_so_far = 0
    frame_idx = 0
    while tokens_so_far < target_tokens:
        desc = needle_desc if frame_idx == needle_frame else normal_desc
        block = block_text.format(desc=desc)
        blocks.append(block)
        frame_idx += 1
        # rough token count increment (will be measured accurately below)
        tokens_so_far += 22

    raw_context = "".join(blocks)

    questions = [
        (
            "needle",
            "The video consists of many frames. One of them contains a red anomaly with a secret code. "
            "What is the exact code? Output only the code.",
        ),
        (
            "temporal",
            f"At which frame number does the red anomaly appear? Answer with a single integer. "
            f"(Hint: it is close to {needle_frame})",
        ),
        (
            "presence",
            "Does the video contain any frame with a red anomaly? Answer yes or no.",
        ),
    ]

    prompts = []
    for task_name, qtext in questions:
        messages = [
            {
                "role": "user",
                "content": (
                    f"You are watching a long video frame by frame.\n\n"
                    f"{raw_context}\n\n"
                    f"Question: {qtext}"
                ),
            }
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompts.append((task_name, prompt))

    return prompts


# ---------------------------------------------------------------------------
# Prefill / Decode Helpers
# ---------------------------------------------------------------------------

class _ModelPrefillAdapter:
    """Thin adapter so ChunkedPrefillEngine can drive the real model."""
    def __init__(self, real_model):
        self.model = real_model
        self.config = real_model.config

    def __call__(self, input_ids, past_key_values, use_cache=True, **kwargs):
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs
            )
        return outputs


def run_hetero_prefill_decode(
    model, tokenizer, prompt_text: str, device: str,
    max_new_tokens: int = 20, keep_tail: int = 8192, chunk_size: int = 2048
):
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    input_ids = inputs.input_ids
    seq_len = input_ids.shape[1]

    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats(device)

    cache = build_fused_cache(
        device=device,
        sink_tokens=64,
        keep_tail=keep_tail,
        chunk_size=chunk_size,
        group_size=128,
        enable_quant=True,
        enable_prefetch=False,
        enable_triton=False,
    )

    adapter = _ModelPrefillAdapter(model)
    engine = ChunkedPrefillEngine(model=adapter, cache=cache, chunk_size=chunk_size)

    # Prefill
    t0 = time.time()
    engine.prefill(input_ids)
    torch.cuda.synchronize(device)
    ttft = time.time() - t0
    throughput = seq_len / ttft if ttft > 0 else 0.0
    peak_prefill = torch.cuda.max_memory_allocated(device) / (1024 ** 3)

    # Decode
    current_input = input_ids[:, -1:]
    decode_times = []
    generated = [current_input]

    for _ in range(max_new_tokens):
        t1 = time.time()
        with torch.no_grad():
            out = adapter(input_ids=current_input, past_key_values=cache, use_cache=True)
        next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
        torch.cuda.synchronize(device)
        decode_times.append(time.time() - t1)
        generated.append(next_token)
        current_input = next_token

    tpot = sum(decode_times) / len(decode_times)
    peak_total = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    steady_mem = torch.cuda.memory_allocated(device) / (1024 ** 3)

    full_ids = torch.cat(generated, dim=-1)
    response = tokenizer.decode(full_ids[0, seq_len:], skip_special_tokens=True)

    return {
        "success": True,
        "ttft_s": round(ttft, 3),
        "tpot_ms": round(tpot * 1000, 2),
        "throughput_tok_s": round(throughput, 1),
        "peak_mem_gb": round(peak_total, 3),
        "steady_mem_gb": round(steady_mem, 3),
        "seq_len": seq_len,
        "response": response,
    }


def run_native_prefill_decode(
    model, tokenizer, prompt_text: str, device: str, max_new_tokens: int = 20
):
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    input_ids = inputs.input_ids
    seq_len = input_ids.shape[1]

    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats(device)

    try:
        # Prefill
        t0 = time.time()
        with torch.no_grad():
            out = model(input_ids=input_ids, use_cache=True)
        torch.cuda.synchronize(device)
        ttft = time.time() - t0
        throughput = seq_len / ttft if ttft > 0 else 0.0

        past_kv = out.past_key_values
        current_input = input_ids[:, -1:]
        decode_times = []
        generated = [current_input]

        # Decode
        for _ in range(max_new_tokens):
            t1 = time.time()
            with torch.no_grad():
                out = model(
                    input_ids=current_input,
                    past_key_values=past_kv,
                    use_cache=True
                )
            next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
            torch.cuda.synchronize(device)
            decode_times.append(time.time() - t1)
            past_kv = out.past_key_values
            generated.append(next_token)
            current_input = next_token

        tpot = sum(decode_times) / len(decode_times)
        peak_total = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        steady_mem = torch.cuda.memory_allocated(device) / (1024 ** 3)

        full_ids = torch.cat(generated, dim=-1)
        response = tokenizer.decode(full_ids[0, seq_len:], skip_special_tokens=True)

        return {
            "success": True,
            "ttft_s": round(ttft, 3),
            "tpot_ms": round(tpot * 1000, 2),
            "throughput_tok_s": round(throughput, 1),
            "peak_mem_gb": round(peak_total, 3),
            "steady_mem_gb": round(steady_mem, 3),
            "seq_len": seq_len,
            "response": response,
        }
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            return {"success": False, "error": "OOM", "message": str(e)[:200]}
        raise


# ---------------------------------------------------------------------------
# Task accuracy checkers
# ---------------------------------------------------------------------------

def check_accuracy(task_name: str, response: str, needle_frame: int):
    r = response.strip().lower()
    if task_name == "needle":
        return "9527" in r
    if task_name == "temporal":
        # allow a tolerance of ±5 frames
        for offset in range(-5, 6):
            if str(needle_frame + offset) in response:
                return True
        return False
    if task_name == "presence":
        return "yes" in r or "是" in r
    return False


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("=" * 80)
    print(" Phase 2a: Long-Video MLLM Evaluation (Qwen2-VL-7B + Hetero-KV)")
    print("=" * 80)
    print(f"Device: {device}")

    print("\nLoading Qwen2-VL-7B (BF16)...")
    model, tokenizer = load_qwen2vl(device=device)

    test_configs = [
        {"name": "8K tokens", "target_tokens": 8192, "needle_frame": 300},
        {"name": "16K tokens", "target_tokens": 16384, "needle_frame": 600},
        {"name": "32K tokens", "target_tokens": 32768, "needle_frame": 1200},
        {"name": "64K tokens", "target_tokens": 65536, "needle_frame": 2400},
    ]

    all_results = []

    for cfg in test_configs:
        print(f"\n{'='*60}")
        print(f" Configuration: {cfg['name']}  (needle @ frame {cfg['needle_frame']})")
        print(f"{'='*60}")

        task_prompts = build_simulated_video_prompt(
            tokenizer, target_tokens=cfg["target_tokens"], needle_frame=cfg["needle_frame"]
        )

        config_res = {
            "config": cfg["name"],
            "target_tokens": cfg["target_tokens"],
            "needle_frame": cfg["needle_frame"],
            "tasks": {},
        }

        for task_name, prompt_text in task_prompts:
            # Verify actual length once
            actual_len = tokenizer(prompt_text, return_tensors="pt").input_ids.shape[1]
            print(f"\n  Task: {task_name} | Actual tokens: {actual_len}")

            task_result = {"baseline": None, "hetero": None}

            # --- Baseline ---
            print("    [Baseline] Running...")
            baseline_out = run_native_prefill_decode(model, tokenizer, prompt_text, device, max_new_tokens=20)
            if baseline_out.get("success"):
                correct = check_accuracy(task_name, baseline_out["response"], cfg["needle_frame"])
                baseline_out["correct"] = correct
                task_result["baseline"] = baseline_out
                print(f"      TTFT={baseline_out['ttft_s']:.3f}s "
                      f"TPOT={baseline_out['tpot_ms']:.2f}ms "
                      f"Peak={baseline_out['peak_mem_gb']:.3f}GB "
                      f"Correct={correct}")
            else:
                task_result["baseline"] = baseline_out
                print(f"      FAILED: {baseline_out.get('error')}")

            torch.cuda.empty_cache()
            gc.collect()

            # --- Hetero-KV ---
            print("    [Hetero-KV] Running...")
            hetero_out = run_hetero_prefill_decode(
                model, tokenizer, prompt_text, device,
                max_new_tokens=20, keep_tail=8192, chunk_size=2048
            )
            if hetero_out.get("success"):
                correct = check_accuracy(task_name, hetero_out["response"], cfg["needle_frame"])
                hetero_out["correct"] = correct
                task_result["hetero"] = hetero_out
                print(f"      TTFT={hetero_out['ttft_s']:.3f}s "
                      f"TPOT={hetero_out['tpot_ms']:.2f}ms "
                      f"Peak={hetero_out['peak_mem_gb']:.3f}GB "
                      f"Correct={correct}")
            else:
                task_result["hetero"] = hetero_out
                print(f"      FAILED: {hetero_out.get('error')}")

            config_res["tasks"][task_name] = task_result
            config_res["actual_tokens"] = actual_len

            torch.cuda.empty_cache()
            gc.collect()

        all_results.append(config_res)

    # Summary table
    print("\n" + "=" * 80)
    print(" Summary Table")
    print("=" * 80)
    headers = ["Config", "Task", "Mode", "TTFT(s)", "TPOT(ms)", "Thru(tok/s)", "Peak(GB)", "Acc"]
    print(" | ".join(headers))
    print("-" * 100)
    for r in all_results:
        for task_name, task_r in r["tasks"].items():
            for mode in ["baseline", "hetero"]:
                d = task_r.get(mode)
                if d and d.get("success", True):
                    cells = [
                        r["config"],
                        task_name,
                        mode,
                        str(d.get("ttft_s", "-")),
                        str(d.get("tpot_ms", "-")),
                        str(d.get("throughput_tok_s", "-")),
                        str(d.get("peak_mem_gb", "-")),
                        "✓" if d.get("correct") else "✗",
                    ]
                else:
                    cells = [r["config"], task_name, mode, "-", "-", "-", "-", "FAIL"]
                print(" | ".join(cells))

    os.makedirs("experiments", exist_ok=True)
    out_path = "experiments/eval_long_video_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n[Saved] {out_path}")


if __name__ == "__main__":
    main()
