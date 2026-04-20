"""
scripts/run_streamingllm.py
===========================
StreamingLLM baseline (retain 64 Sink + 4096 Local) with real generation and recall testing.
Based on Qwen2-VL-7B-Instruct 4-bit NF4, using real video data.
"""

import os
import sys
import json
import gc
import time
import torch
import warnings
from transformers.cache_utils import DynamicCache

warnings.filterwarnings("ignore")

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
from utils.real_data_loader import build_multimodal_inputs
from utils.cuda_profiler import NativeHFProfiler, reset_memory_stats

MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(os.path.dirname(project_root), "models", "Qwen2-VL-7B"))
RESULTS_PATH = os.path.join(project_root, "results", "streaming_llm_results.json")
DEVICE = "cuda"

SINK_TOKENS = 64
LOCAL_WINDOW = 4096


class StreamingLLMCache(DynamicCache):
    """
    StreamingLLM Cache implementation: only keep the first sink_tokens as Sink Tokens
    and the last local_window as Local Tokens; middle tokens are physically discarded.
    """

    def __init__(self, sink_tokens: int = 64, local_window: int = 4096):
        super().__init__()
        self.sink_tokens = sink_tokens
        self.local_window = local_window
        self._seen_tokens = 0

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            new_k = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            new_v = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
            cur_len = new_k.shape[-2]
            max_len = self.sink_tokens + self.local_window

            if cur_len > max_len:
                k_sink = new_k[..., :self.sink_tokens, :]
                v_sink = new_v[..., :self.sink_tokens, :]
                k_local = new_k[..., -self.local_window:, :]
                v_local = new_v[..., -self.local_window:, :]
                self.key_cache[layer_idx] = torch.cat([k_sink, k_local], dim=-2)
                self.value_cache[layer_idx] = torch.cat([v_sink, v_local], dim=-2)
                del k_sink, v_sink, k_local, v_local
            else:
                self.key_cache[layer_idx] = new_k
                self.value_cache[layer_idx] = new_v

            del new_k, new_v

        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx=0):
        return self._seen_tokens


def _build_needle_inputs(processor, device):
    """Build Needle-in-Video test inputs using the project's needle_video_5min.mp4."""
    repo_root = os.path.dirname(project_root)
    needle_video = os.path.join(repo_root, "needle_video_5min.mp4")
    if not os.path.exists(needle_video):
        raise FileNotFoundError(f"Needle video not found: {needle_video}")

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": needle_video,
                    "fps": 1.0,
                    "max_pixels": 100352,
                },
                {
                    "type": "text",
                    "text": (
                        "This long video is mostly normal gray frames. But at one moment, "
                        "the screen turned red and an anomalous code appeared. Please recall: "
                        "what is the exact content of that code? Output the code directly."
                    ),
                },
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    return inputs, needle_video


def run_streamingllm():
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)

    print("[StreamingLLM] Loading Qwen2-VL-7B-Instruct (4-bit NF4)...")
    torch.cuda.empty_cache()
    gc.collect()

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

    model_weights_gb = torch.cuda.memory_allocated(DEVICE) / (1024**3)
    print(f"[StreamingLLM] Model weights VRAM: {model_weights_gb:.2f} GB")

    results = {
        "method": "StreamingLLM",
        "config": {"sink_tokens": SINK_TOKENS, "local_window": LOCAL_WINDOW},
        "model": "Qwen2-VL-7B-Instruct-4bit",
        "model_weights_gb": model_weights_gb,
        "tests": [],
    }

    # ============================================================
    # 1. Video recall test (Needle-in-Video)
    # ============================================================
    print("\n[StreamingLLM] ====== Video recall test (Needle-in-Video) ======")
    try:
        needle_inputs, needle_video = _build_needle_inputs(processor, DEVICE)
        needle_len = int(needle_inputs["input_ids"].shape[1])
        print(f"[StreamingLLM] Needle video token count: {needle_len}")

        cache = StreamingLLMCache(sink_tokens=SINK_TOKENS, local_window=LOCAL_WINDOW)
        reset_memory_stats(DEVICE)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                **needle_inputs,
                max_new_tokens=20,
                past_key_values=cache,
            )
        torch.cuda.synchronize()
        total_time = time.perf_counter() - t0

        generated_ids = outputs[0][len(needle_inputs["input_ids"][0]):]
        resp = processor.batch_decode([generated_ids], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        peak_gb = torch.cuda.max_memory_allocated(DEVICE) / (1024**3)

        recall_hit = "9527" in resp or "ANOMALY" in resp.upper()
        print(f"[StreamingLLM] Output: {resp}")
        print(f"[StreamingLLM] Recall: {'HIT' if recall_hit else 'MISS'} | Peak: {peak_gb:.2f} GB | Time: {total_time:.2f}s")

        results["video_recall_test"] = {
            "video_path": needle_video,
            "input_tokens": needle_len,
            "output_text": resp,
            "recall_hit": recall_hit,
            "peak_memory_gb": peak_gb,
            "total_time_s": total_time,
        }

        del outputs, cache, needle_inputs
        torch.cuda.empty_cache()
        gc.collect()
    except Exception as e:
        print(f"[StreamingLLM] Needle test exception: {e}")
        results["video_recall_test"] = {"error": str(e)}

    # ============================================================
    # 2. Performance benchmark (4K / 8K / 12K / 16K)
    # ============================================================
    targets = [4096, 8192, 12288, 16384]
    decode_tokens = 20

    for target in targets:
        print(f"\n[StreamingLLM] ====== Performance test: {target} tokens ======")
        test_res = {"target_tokens": target, "success": False, "oom": False}

        try:
            data = build_multimodal_inputs(target, processor, device=DEVICE)
            inputs = data["inputs"]
            actual = data["actual_tokens"]
            test_res["actual_tokens"] = actual
            print(f"[StreamingLLM] Actual tokens: {actual} | Frames: {data['num_frames']} | Video: {data['video_path']}")

            cache = StreamingLLMCache(sink_tokens=SINK_TOKENS, local_window=LOCAL_WINDOW)
            profiler = NativeHFProfiler(model, DEVICE)

            # Prefill
            reset_memory_stats(DEVICE)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                prefill_out = model(**inputs, past_key_values=cache, use_cache=True)
            torch.cuda.synchronize()
            ttft = time.perf_counter() - t0
            prefill_peak = torch.cuda.max_memory_allocated(DEVICE) / (1024**3)
            kv_cache_gb = torch.cuda.memory_allocated(DEVICE) / (1024**3)

            # Decode
            profiler.run_decode_with_cache_obj(inputs["input_ids"], cache, num_tokens=decode_tokens)

            throughput = decode_tokens / max(profiler.tpot * decode_tokens, 1e-6)
            print(
                f"[StreamingLLM] TTFT: {ttft:.3f}s | TPOT: {profiler.tpot * 1000:.2f}ms | "
                f"Throughput: {throughput:.2f} tok/s | "
                f"Prefill Peak: {prefill_peak:.2f} GB | Decode Peak: {profiler.decode_peak_gb:.2f} GB | "
                f"Steady: {profiler.steady_memory_gb:.2f} GB | Physical KV len: {cache.key_cache[0].shape[-2] if len(cache.key_cache) > 0 else 0}"
            )

            test_res.update({
                "success": True,
                "ttft_s": ttft,
                "tpot_ms": profiler.tpot * 1000,
                "throughput_tok_s": throughput,
                "prefill_peak_gb": prefill_peak,
                "decode_peak_gb": profiler.decode_peak_gb,
                "steady_memory_gb": profiler.steady_memory_gb,
                "kv_cache_gb": kv_cache_gb,
                "physical_kv_len": int(cache.key_cache[0].shape[-2]) if len(cache.key_cache) > 0 else 0,
            })

            del prefill_out, cache, inputs, data
            torch.cuda.empty_cache()
            gc.collect()

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"[StreamingLLM] OOM at {target}: {str(e)[:120]}")
                test_res["oom"] = True
                test_res["error"] = str(e)
                torch.cuda.empty_cache()
                gc.collect()
            else:
                raise

        results["tests"].append(test_res)
        with open(RESULTS_PATH, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\n[StreamingLLM] Testing complete. Results saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    run_streamingllm()
