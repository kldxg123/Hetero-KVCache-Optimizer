"""
scripts/run_sglang.py
===================
SGLang baseline (RadixAttention) with real multimodal inputs.
Measures TTFT, TPOT, Peak Memory, and throughput under a 16GB VRAM budget.
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

from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from utils.real_data_loader import build_multimodal_inputs
from utils.cuda_profiler import reset_memory_stats

MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(os.path.dirname(project_root), "models", "Qwen2-VL-7B"))
RESULTS_PATH = os.path.join(project_root, "results", "sglang_results.json")
DEVICE = "cuda"


def run_sglang():
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)

    print("[SGLang] Initializing SGLang engine (RadixAttention)...")
    torch.cuda.empty_cache()
    gc.collect()

    try:
        from sglang import Runtime
    except ImportError:
        print("[SGLang] sglang package not installed. Skipping.")
        with open(RESULTS_PATH, "w") as f:
            json.dump({"method": "SGLang-RadixAttention", "available": False}, f, indent=2)
        return

    reset_memory_stats(DEVICE)
    try:
        runtime = Runtime(
            model_path=MODEL_PATH,
            tp_size=1,
            mem_fraction_static=0.80,
            max_model_len=32768,
            dtype="bfloat16",
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"[SGLang] Engine initialization failed: {e}")
        with open(RESULTS_PATH, "w") as f:
            json.dump({"method": "SGLang-RadixAttention", "available": False, "error": str(e)}, f, indent=2)
        return

    engine_peak_gb = torch.cuda.max_memory_allocated(DEVICE) / (1024**3)
    print(f"[SGLang] Engine initialized. Engine peak memory: {engine_peak_gb:.2f} GB")

    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)

    results = {
        "method": "SGLang-RadixAttention",
        "model": "Qwen2-VL-7B-Instruct",
        "engine_peak_gb": engine_peak_gb,
        "tests": [],
    }

    targets = [4096, 8192, 12288, 16384]
    decode_tokens = 20

    for target in targets:
        print(f"\n[SGLang] ====== Target: {target} tokens ======")
        test_res = {"target_tokens": target, "success": False, "oom": False}

        try:
            data = build_multimodal_inputs(target, processor, device="cpu")
            messages = data["messages"]
            actual_tokens = data["actual_tokens"]
            test_res["actual_tokens"] = actual_tokens
            print(f"[SGLang] Actual tokens: {actual_tokens} | Frames: {data['num_frames']} | Video: {data['video_path']}")

            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)

            mm_data = {}
            if image_inputs is not None:
                mm_data["image"] = image_inputs
            if video_inputs is not None:
                mm_data["video"] = video_inputs

            prompt_payload = {"text": text, "multi_modal_data": mm_data}

            # TTFT: generate 1 token
            reset_memory_stats(DEVICE)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            state = runtime.submit(prompt_payload)
            _ = runtime.generate(state, max_new_tokens=1, temperature=0.1, top_p=0.8)
            torch.cuda.synchronize()
            ttft = time.perf_counter() - t0

            # Decode: generate decode_tokens
            reset_memory_stats(DEVICE)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            state = runtime.submit(prompt_payload)
            result = runtime.generate(state, max_new_tokens=decode_tokens, temperature=0.1, top_p=0.8)
            torch.cuda.synchronize()
            total_time = time.perf_counter() - t0
            tpot = (total_time - ttft) / max(1, decode_tokens - 1)

            decode_peak = torch.cuda.max_memory_allocated(DEVICE) / (1024**3)
            steady = torch.cuda.memory_allocated(DEVICE) / (1024**3)
            generated_text = result.get("text", "")
            throughput = decode_tokens / max(total_time - ttft, 1e-6)

            print(
                f"[SGLang] TTFT: {ttft:.3f}s | TPOT: {tpot * 1000:.2f}ms | "
                f"Throughput: {throughput:.2f} tok/s | Decode Peak: {decode_peak:.2f} GB | Steady: {steady:.2f} GB"
            )
            print(f"[SGLang] Output preview: {generated_text[:120]}...")

            test_res.update({
                "success": True,
                "ttft_s": ttft,
                "tpot_ms": tpot * 1000,
                "throughput_tok_s": throughput,
                "decode_peak_gb": decode_peak,
                "steady_memory_gb": steady,
                "output_preview": generated_text[:200],
            })

        except Exception as e:
            err_str = str(e).lower()
            if "out of memory" in err_str or "cuda" in err_str:
                print(f"[SGLang] OOM/Error at {target}: {str(e)[:150]}")
                test_res["oom"] = True
                test_res["error"] = str(e)
            else:
                print(f"[SGLang] Error at {target}: {str(e)[:150]}")
                test_res["error"] = str(e)
            torch.cuda.empty_cache()
            gc.collect()

        results["tests"].append(test_res)
        with open(RESULTS_PATH, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\n[SGLang] Testing complete. Results saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    run_sglang()
