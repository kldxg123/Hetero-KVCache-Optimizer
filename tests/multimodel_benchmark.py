"""
tests/multimodel_benchmark.py
==============================
Multi-model rigorous benchmark for Hetero-KVCache-Optimizer.

Supports:
  - meta-llama/Llama-3.1-8B-Instruct  (text LLM)
  - OpenGVLab/InternVL2-8B            (MLLM)

Evaluates both native HF and Hetero-KV paths with identical prompts.
"""

import os
import sys
import json
import time
import gc
import argparse
import warnings

warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Patched version check
def _try_patch_transformers():
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

_try_patch_transformers()

import torch
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.core.engine_wrapper import build_fused_cache, ChunkedPrefillEngine
from src.simulation.pcie_throttle_sim import BandwidthLimiter


class ModelAdapter:
    """Thin wrapper to unify HF model calling for ChunkedPrefillEngine."""

    def __init__(self, real_model):
        self.model = real_model
        self.config = real_model.config

    def __call__(self, input_ids, past_key_values, use_cache=True, **kwargs):
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs,
            )
        return outputs


def load_model(model_path: str, device: str = "cuda"):
    """Load a causal LM or MLLM with 4-bit quantization."""
    print(f"[MultiModel] Loading {model_path} ...")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    gc.collect()

    local_only = os.path.exists(model_path)
    if local_only:
        print(f"[MultiModel] Using local model at {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=local_only
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Special-case Qwen2-VL which requires its own model class
    if "Qwen2-VL" in model_path or "Qwen2-VL" in str(tokenizer.__class__):
        try:
            from transformers import Qwen2VLForConditionalGeneration
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_path,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                local_files_only=local_only,
                torch_dtype=torch.bfloat16,
            )
        except Exception as e:
            print(f"[MultiModel] Qwen2VLForConditionalGeneration failed ({e}), falling back...")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                local_files_only=local_only,
                torch_dtype=torch.bfloat16,
            )
    else:
        # Try causal LM first, then vision-enabled variant
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                local_files_only=local_only,
                torch_dtype=torch.bfloat16,
            )
        except Exception as e:
            print(f"[MultiModel] AutoModelForCausalLM failed ({e}), trying generic from_pretrained ...")
            from transformers import AutoModel
            model = AutoModel.from_pretrained(
                model_path,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                local_files_only=local_only,
                torch_dtype=torch.bfloat16,
            )

    model.eval()
    load_mem = torch.cuda.memory_allocated() / (1024 ** 3)
    print(f"[MultiModel] Model loaded | Weight memory: {load_mem:.2f} GB")
    return model, tokenizer


def create_prompt(tokenizer, target_tokens: int, model_family: str = "llama"):
    """Generate a long text-only prompt of approximately target_tokens length."""
    base = "The quick brown fox jumps over the lazy dog. "
    if model_family.lower().startswith("internvl"):
        # Simulate visual tokens with special markers for InternVL
        base = "<image>\nDescribe the image in detail. "

    repeat = (target_tokens // (len(base.split()) // 2)) + 1
    long_text = (base * repeat)[: target_tokens * 6]

    messages = [{"role": "user", "content": long_text}]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = long_text

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=target_tokens + 128,
    )
    if "input_ids" not in inputs:
        raise RuntimeError("Tokenizer did not return input_ids")
    inputs = {k: v.to(model_device()) for k, v in inputs.items()}
    actual = inputs["input_ids"].shape[1]
    print(f"[MultiModel] Prompt actual tokens: {actual}")
    return inputs, actual


def model_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def run_hetero(
    model,
    tokenizer,
    inputs,
    min_new_tokens: int = 20,
    chunk_size: int = 2048,
    sink_tokens: int = 64,
    keep_tail: int = 8192,
    bandwidth_gbps: Optional[float] = None,
):
    """Run inference with Hetero-KVCache."""
    input_ids = inputs["input_ids"]
    device = input_ids.device
    seq_len = input_ids.shape[1]
    print(f"\n[MultiModel] Hetero-KV | seq_len={seq_len}")

    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()

    limiter = None
    if bandwidth_gbps is not None:
        limiter = BandwidthLimiter(max_bandwidth_gbps=bandwidth_gbps)
        print(f"[MultiModel] PCIe throttle enabled: {bandwidth_gbps} GB/s")

    cache = build_fused_cache(
        num_layers=getattr(model.config, "num_hidden_layers", None),
        device=str(device),
        sink_tokens=sink_tokens,
        keep_tail=keep_tail,
        chunk_size=chunk_size,
        group_size=128,
        enable_quant=True,
        enable_prefetch=False,
        enable_triton=False,
        bandwidth_limiter=limiter,
    )

    adapter = ModelAdapter(model)
    engine = ChunkedPrefillEngine(model=adapter, cache=cache, chunk_size=chunk_size)

    try:
        # Prefill
        t0 = time.time()
        engine.prefill(input_ids)
        torch.cuda.synchronize()
        ttft = time.time() - t0
        peak_prefill = torch.cuda.max_memory_allocated() / (1024 ** 3)
        print(f"[MultiModel] Hetero Prefill | TTFT={ttft:.3f}s Peak={peak_prefill:.3f}GB")

        # Decode
        current_input = input_ids[:, -1:]
        decode_times = []
        for _ in range(min_new_tokens):
            t1 = time.time()
            with torch.no_grad():
                outputs = adapter(input_ids=current_input, past_key_values=cache, use_cache=True)
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
            torch.cuda.synchronize()
            decode_times.append(time.time() - t1)
            current_input = next_token

        tpot = sum(decode_times) / len(decode_times)
        peak_total = torch.cuda.max_memory_allocated() / (1024 ** 3)
        steady_mem = torch.cuda.memory_allocated() / (1024 ** 3)

        print(
            f"[MultiModel] Hetero Decode | TPOT={tpot * 1000:.2f}ms "
            f"Peak={peak_total:.3f}GB Steady={steady_mem:.3f}GB"
        )

        return {
            "success": True,
            "ttft": ttft,
            "tpot": tpot,
            "peak_memory_gb": peak_total,
            "steady_memory_gb": steady_mem,
            "seq_length": cache.get_seq_length(),
            "dram_entries": len(cache.dram_table),
        }
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            return {"success": False, "error": "OOM"}
        raise
    finally:
        del cache, engine, adapter


def run_native(model, tokenizer, inputs, min_new_tokens: int = 20):
    """Run inference with native HF cache."""
    input_ids = inputs["input_ids"]
    seq_len = input_ids.shape[1]
    print(f"\n[MultiModel] Native HF | seq_len={seq_len}")

    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()

    try:
        t0 = time.time()
        with torch.no_grad():
            outputs = model(input_ids=input_ids, use_cache=True)
        torch.cuda.synchronize()
        ttft = time.time() - t0
        peak_prefill = torch.cuda.max_memory_allocated() / (1024 ** 3)
        print(f"[MultiModel] Native Prefill | TTFT={ttft:.3f}s Peak={peak_prefill:.3f}GB")

        past_key_values = outputs.past_key_values
        current_input = input_ids[:, -1:]
        decode_times = []
        for _ in range(min_new_tokens):
            t1 = time.time()
            with torch.no_grad():
                outputs = model(
                    input_ids=current_input,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
            torch.cuda.synchronize()
            decode_times.append(time.time() - t1)
            past_key_values = outputs.past_key_values
            current_input = next_token

        tpot = sum(decode_times) / len(decode_times)
        peak_total = torch.cuda.max_memory_allocated() / (1024 ** 3)
        steady_mem = torch.cuda.memory_allocated() / (1024 ** 3)

        print(
            f"[MultiModel] Native Decode | TPOT={tpot * 1000:.2f}ms "
            f"Peak={peak_total:.3f}GB Steady={steady_mem:.3f}GB"
        )

        return {
            "success": True,
            "ttft": ttft,
            "tpot": tpot,
            "peak_memory_gb": peak_total,
            "steady_memory_gb": steady_mem,
        }
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            return {"success": False, "error": "OOM"}
        raise


def benchmark_model(model_name: str, model_path: str, configs, args):
    """Run full benchmark suite for a single model."""
    results = {
        "model_name": model_name,
        "model_path": model_path,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "tests": [],
    }

    try:
        model, tokenizer = load_model(model_path, device=model_device())
        family = "qwen"

        for cfg in configs:
            print(f"\n{'='*60}")
            print(f" {model_name} | {cfg['name']}")
            print(f"{'='*60}")

            inputs, actual_tokens = create_prompt(tokenizer, cfg["target_tokens"], family)
            test_result = {
                "name": cfg["name"],
                "target_tokens": cfg["target_tokens"],
                "actual_tokens": actual_tokens,
                "hetero": None,
                "native": None,
            }

            # Hetero run
            hetero_res = run_hetero(
                model,
                tokenizer,
                inputs,
                min_new_tokens=args.min_new_tokens,
                chunk_size=args.chunk_size,
                sink_tokens=args.sink_tokens,
                keep_tail=args.keep_tail,
                bandwidth_gbps=args.bandwidth_gbps,
            )
            test_result["hetero"] = hetero_res
            torch.cuda.empty_cache()
            gc.collect()

            # Native run (skip if expected OOM)
            if cfg.get("skip_native", False) or actual_tokens > args.native_max_tokens:
                test_result["native"] = {"skipped": True, "reason": "expected OOM"}
            else:
                native_res = run_native(
                    model, tokenizer, inputs, min_new_tokens=args.min_new_tokens
                )
                test_result["native"] = native_res
                torch.cuda.empty_cache()
                gc.collect()

            results["tests"].append(test_result)

            # Save incremental results
            os.makedirs("experiments", exist_ok=True)
            with open(f"experiments/multimodel_{model_name.replace('/', '_')}.json", "w") as f:
                json.dump(results, f, indent=2)

            del inputs

    except Exception as e:
        print(f"[MultiModel] ERROR benchmarking {model_name}: {e}")
        import traceback
        traceback.print_exc()
        results["error"] = str(e)

    return results


def main():
    parser = argparse.ArgumentParser(description="Multi-model Hetero-KV benchmark")
    parser.add_argument("--models", nargs="+", default=["qwen_text", "qwen_vl"],
                        help="Models to benchmark: qwen_text, qwen_vl")
    parser.add_argument("--llama_path", default="models/Qwen2.5-7B-Instruct",
                        help="Path or HF hub name for text LLM")
    parser.add_argument("--internvl_path", default="models/Qwen2-VL-7B",
                        help="Path or HF hub name for MLLM")
    parser.add_argument("--token_targets", nargs="+", type=int, default=[4096, 8192, 12000],
                        help="Target token lengths")
    parser.add_argument("--native_max_tokens", type=int, default=8000,
                        help="Max tokens to attempt native HF baseline")
    parser.add_argument("--min_new_tokens", type=int, default=20)
    parser.add_argument("--chunk_size", type=int, default=2048)
    parser.add_argument("--sink_tokens", type=int, default=64)
    parser.add_argument("--keep_tail", type=int, default=8192)
    parser.add_argument("--bandwidth_gbps", type=float, default=None,
                        help="Simulate PCIe bandwidth limit (GB/s). e.g. 16.0 for RTX 4060 Ti x8")
    args = parser.parse_args()

    model_registry = {
        "qwen_text": ("Qwen2.5-7B", args.llama_path),
        "qwen_vl": ("Qwen2-VL-7B", args.internvl_path),
    }

    configs = [
        {"name": f"{t//1000}K_context", "target_tokens": t}
        for t in args.token_targets
    ]

    all_results = {}
    for key in args.models:
        if key not in model_registry:
            print(f"[MultiModel] Unknown model key: {key}, skipping.")
            continue
        name, path = model_registry[key]
        all_results[name] = benchmark_model(name, path, configs, args)

    os.makedirs("experiments", exist_ok=True)
    with open("experiments/multimodel_benchmark.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 60)
    print(" Multi-model benchmark complete")
    print(" Results: experiments/multimodel_benchmark.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
