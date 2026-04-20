"""
scripts/run_vllm.py
===================
vLLM engine (PagedAttention) with max_model_len=32768 and strict 16GB VRAM limit.
Uses real video data via vLLM offline inference API and records TTFT / TPOT / VRAM.
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
RESULTS_PATH = os.path.join(project_root, "results", "vllm_results.json")
DEVICE = "cuda"


def run_vllm():
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)

    print("[vLLM] Initializing vLLM engine (PagedAttention)...")
    torch.cuda.empty_cache()
    gc.collect()

    from vllm import LLM, SamplingParams

    reset_memory_stats(DEVICE)
    # Enforce a strict 16GB VRAM ceiling to emulate edge deployment
    total_vram_gb = torch.cuda.get_device_properties(DEVICE).total_memory / (1024**3)
    gpu_mem_util = min(0.90, 15.5 / total_vram_gb)
    llm = LLM(
        model=MODEL_PATH,
        trust_remote_code=True,
        max_model_len=32768,
        gpu_memory_utilization=gpu_mem_util,
        max_num_seqs=1,
        limit_mm_per_prompt={"image": 2000, "video": 10},
        dtype="bfloat16",
        swap_space=4,
    )
    model_weights_gb = torch.cuda.max_memory_allocated(DEVICE) / (1024**3)
    print(f"[vLLM] Engine initialized. Model weights + engine pre-allocation: {model_weights_gb:.2f} GB")

    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)

    results = {
        "method": "vLLM-PagedAttention",
        "model": "Qwen2-VL-7B-Instruct",
        "model_weights_gb": model_weights_gb,
        "tests": [],
    }

    targets = [4096, 8192, 12288, 16384]

    for target in targets:
        print(f"\n[vLLM] ====== Target: {target} tokens ======")
        test_res = {"target_tokens": target, "success": False, "oom": False}

        try:
            data = build_multimodal_inputs(target, processor, device="cpu")
            messages = data["messages"]
            actual_tokens = data["actual_tokens"]
            test_res["actual_tokens"] = actual_tokens
            print(f"[vLLM] Actual input tokens: {actual_tokens} | Frames: {data['num_frames']} | Video: {data['video_path']}")

            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)

            mm_data = {}
            if image_inputs is not None:
                mm_data["image"] = image_inputs
            if video_inputs is not None:
                mm_data["video"] = video_inputs

            llm_inputs = [{"prompt": text, "multi_modal_data": mm_data}]

            # TTFT: generate 1 token
            sp_ttft = SamplingParams(temperature=0.1, top_p=0.8, max_tokens=1)
            reset_memory_stats(DEVICE)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = llm.generate(llm_inputs, sampling_params=sp_ttft)
            torch.cuda.synchronize()
            ttft = time.perf_counter() - t0

            # Decode: generate 20 tokens
            sp_decode = SamplingParams(temperature=0.1, top_p=0.8, max_tokens=20)
            reset_memory_stats(DEVICE)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            outputs = llm.generate(llm_inputs, sampling_params=sp_decode)
            torch.cuda.synchronize()
            total_time = time.perf_counter() - t0
            tpot = (total_time - ttft) / max(1, sp_decode.max_tokens - 1)

            decode_peak = torch.cuda.max_memory_allocated(DEVICE) / (1024**3)
            steady = torch.cuda.memory_allocated(DEVICE) / (1024**3)
            generated_text = outputs[0].outputs[0].text
            throughput = 20 / max(total_time - ttft, 1e-6)

            print(
                f"[vLLM] TTFT: {ttft:.3f}s | TPOT: {tpot * 1000:.2f}ms | "
                f"Throughput: {throughput:.2f} tok/s | "
                f"Decode Peak: {decode_peak:.2f} GB | Steady: {steady:.2f} GB"
            )
            print(f"[vLLM] Output preview: {generated_text[:120]}...")

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
                print(f"[vLLM] OOM/Error at {target}: {str(e)[:150]}")
                test_res["oom"] = True
                test_res["error"] = str(e)
            else:
                print(f"[vLLM] Error at {target}: {str(e)[:150]}")
                test_res["error"] = str(e)
            torch.cuda.empty_cache()
            gc.collect()

        results["tests"].append(test_res)
        with open(RESULTS_PATH, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\n[vLLM] Testing complete. Results saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    run_vllm()
