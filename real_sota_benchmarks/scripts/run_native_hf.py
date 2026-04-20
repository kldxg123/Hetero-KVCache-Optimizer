"""
scripts/run_native_hf.py
========================
Native HuggingFace Qwen2-VL-7B-Instruct (4-bit NF4) loading.
Tests dynamic KV growth up to OOM.
All measurements are wrapped with torch.cuda.synchronize();
VRAM is strictly separated into model weights vs KV Cache.
"""

import os
import sys
import json
import gc
import time
import torch
import warnings

warnings.filterwarnings("ignore")

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from utils.real_data_loader import build_multimodal_inputs
from utils.cuda_profiler import NativeHFProfiler, reset_memory_stats

MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(os.path.dirname(project_root), "models", "Qwen2-VL-7B"))
RESULTS_PATH = os.path.join(project_root, "results", "native_hf_results.json")
DEVICE = "cuda"


def run_native_hf():
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)

    print("[Native-HF] Loading Qwen2-VL-7B-Instruct (4-bit NF4)...")
    torch.cuda.empty_cache()
    gc.collect()
    reset_memory_stats(DEVICE)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map=DEVICE,
        trust_remote_code=True,
        local_files_only=True,
    )
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
    model.eval()

    profiler = NativeHFProfiler(model, DEVICE)
    print(f"[Native-HF] Model weights VRAM: {profiler.model_weights_gb:.2f} GB")

    results = {
        "method": "Native-HF-4bit-NF4",
        "model": "Qwen2-VL-7B-Instruct",
        "model_weights_gb": profiler.model_weights_gb,
        "tests": [],
    }

    targets = [4096, 8192, 12288, 16384]
    decode_tokens = 20

    for target in targets:
        print(f"\n[Native-HF] ====== Target: {target} tokens ======")
        test_res = {
            "target_tokens": target,
            "success": False,
            "oom": False,
        }

        try:
            data = build_multimodal_inputs(target, processor, device=DEVICE)
            inputs = data["inputs"]
            actual_tokens = data["actual_tokens"]
            test_res["actual_tokens"] = actual_tokens
            print(f"[Native-HF] Actual input tokens: {actual_tokens} | Frames: {data['num_frames']} | Video: {data['video_path']}")

            # Prefill (TTFT)
            outputs = profiler.run_prefill(inputs)
            print(f"[Native-HF] TTFT: {profiler.ttft:.3f}s | Prefill Peak: {profiler.prefill_peak_gb:.2f} GB | KV Cache: {profiler.kv_cache_gb:.2f} GB")

            # Decode (TPOT)
            profiler.run_decode(inputs["input_ids"], outputs.past_key_values, num_tokens=decode_tokens)
            throughput = decode_tokens / max(profiler.tpot * decode_tokens, 1e-6)
            print(f"[Native-HF] TPOT: {profiler.tpot * 1000:.2f}ms | Throughput: {throughput:.2f} tok/s | Decode Peak: {profiler.decode_peak_gb:.2f} GB | Steady: {profiler.steady_memory_gb:.2f} GB")

            test_res.update({
                "success": True,
                "ttft_s": profiler.ttft,
                "tpot_ms": profiler.tpot * 1000,
                "throughput_tok_s": throughput,
                "prefill_peak_gb": profiler.prefill_peak_gb,
                "decode_peak_gb": profiler.decode_peak_gb,
                "steady_memory_gb": profiler.steady_memory_gb,
                "kv_cache_gb": profiler.kv_cache_gb,
            })

            del outputs, inputs, data
            torch.cuda.empty_cache()
            gc.collect()

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"[Native-HF] OOM at {target} tokens: {str(e)[:120]}")
                test_res["oom"] = True
                test_res["error"] = str(e)
                torch.cuda.empty_cache()
                gc.collect()
            else:
                raise

        results["tests"].append(test_res)
        with open(RESULTS_PATH, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\n[Native-HF] Testing complete. Results saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    run_native_hf()
