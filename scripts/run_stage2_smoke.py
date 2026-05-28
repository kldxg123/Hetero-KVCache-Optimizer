#!/usr/bin/env python3
"""Stage-2 real-model smoke tests under a 4090-like memory cap."""

import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def query_gpu(gpu_index: int):
    proc = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    for line in proc.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 6 and int(parts[0]) == gpu_index:
            return {
                "index": int(parts[0]),
                "uuid": parts[1],
                "name": parts[2],
                "memory_total_mb": int(parts[3]),
                "memory_used_mb": int(parts[4]),
                "utilization_gpu": int(parts[5]),
            }
    raise RuntimeError(f"GPU {gpu_index} not found")


def parse_memory_mb(value: str) -> int:
    digits = "".join(ch for ch in value if ch.isdigit())
    return int(digits) if digits else 0


def query_compute_apps():
    proc = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    apps = []
    for line in proc.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4:
            apps.append(
                {
                    "gpu_uuid": parts[0],
                    "pid": parts[1],
                    "process_name": parts[2],
                    "used_memory_mb": parse_memory_mb(parts[3]),
                }
            )
    return apps


def assert_cuda_visible_matches(gpu):
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible:
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES must be set before running this script directly; "
            f"expected first visible device to map to physical GPU{gpu['index']}."
        )
    first = visible.split(",")[0].strip()
    if first not in {str(gpu["index"]), gpu["uuid"]}:
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES first device does not match the safety-checked GPU: "
            f"first={first}, expected GPU{gpu['index']} or {gpu['uuid']}."
        )
    return {
        "cuda_visible_devices": visible,
        "logical_device": "cuda:0",
        "physical_gpu_index": gpu["index"],
        "physical_gpu_uuid": gpu["uuid"],
    }


def assert_safe(gpu_index: int, allow_busy: bool):
    gpu = query_gpu(gpu_index)
    visibility = assert_cuda_visible_matches(gpu)
    apps = query_compute_apps()
    target_apps = [app for app in apps if app["gpu_uuid"] == gpu["uuid"]]
    busy = bool(target_apps) or gpu["memory_used_mb"] > 1024 or gpu["utilization_gpu"] > 5
    if busy and not allow_busy:
        raise RuntimeError(
            "skipped due to shared-server safety: "
            f"GPU{gpu_index} used={gpu['memory_used_mb']}MB "
            f"util={gpu['utilization_gpu']}%, target_apps={target_apps}, all_apps={apps}"
        )
    return {"gpu": gpu, "cuda_visibility": visibility, "target_apps": target_apps, "all_apps": apps}


def set_seed(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass


def set_cap(cap_gib: float):
    import torch

    total = torch.cuda.get_device_properties(0).total_memory
    fraction = min(1.0, (cap_gib * 1024**3) / float(total))
    torch.cuda.set_per_process_memory_fraction(fraction, 0)
    return {"cap_gib": cap_gib, "total_gib": total / 1024**3, "fraction": fraction}


def build_prompt(target_tokens: int) -> str:
    filler = (
        "This is a controlled long-context smoke test for HeteroKV. "
        "The model should preserve basic instruction following while the cache "
        "manager keeps only a short active KV state. "
    )
    approx_chars = max(256, target_tokens * 4)
    repeated = (filler * ((approx_chars // len(filler)) + 1))[:approx_chars]
    return (
        repeated
        + "\n\nQuestion: Reply with exactly the phrase HETEROKV_SMOKE_OK.\nAnswer:"
    )


def run_smoke(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.core.engine_wrapper import build_fused_cache
    from src.core.fused_attention_patch import patch_qwen2_attention_for_heterokv

    cap = set_cap(args.cap_gib)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    model_path = PROJECT_ROOT / "models" / "Qwen2.5-7B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()
    patched = patch_qwen2_attention_for_heterokv(model)

    results = {
        "cap": cap,
        "model_path": str(model_path),
        "patched_attention_modules": patched,
        "lengths": [],
    }

    for target in args.lengths:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        prompt = build_prompt(target)
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
        cache = build_fused_cache(
            num_layers=getattr(model.config, "num_hidden_layers", None),
            device="cuda:0",
            sink_tokens=args.sink_tokens,
            keep_tail=args.keep_tail,
            chunk_size=args.chunk_size,
            enable_quant=True,
            enable_prefetch=False,
            enable_triton=False,
            self_healing=True,
            adaptive_self_healing=False,
            enable_method_d=True,
        )
        start = time.time()
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                use_cache=True,
                past_key_values=cache,
                pad_token_id=tokenizer.eos_token_id,
            )
        elapsed = time.time() - start
        text = tokenizer.decode(output[0, inputs.input_ids.shape[-1] :], skip_special_tokens=True)
        summary = cache.memory_summary()
        results["lengths"].append(
            {
                "target_tokens": target,
                "actual_input_tokens": int(inputs.input_ids.shape[-1]),
                "generated": text.strip()[:200],
                "ok": "HETEROKV_SMOKE_OK" in text,
                "elapsed_sec": elapsed,
                "max_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
                "max_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
                "memory_summary": summary,
            }
        )
        del cache, inputs, output
        torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-index", type=int, default=1)
    parser.add_argument("--cap-gib", type=float, default=22.0)
    parser.add_argument("--allow-busy", action="store_true")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--lengths", type=int, nargs="+", default=[2048, 4096, 8192])
    parser.add_argument("--sink-tokens", type=int, default=64)
    parser.add_argument("--keep-tail", type=int, default=2048)
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--output", default="experiments/stage2_smoke.json")
    args = parser.parse_args()

    result = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "gpu_index": args.gpu_index,
        "cap_gib": args.cap_gib,
        "seed": args.seed,
        "lengths_requested": args.lengths,
    }
    try:
        set_seed(args.seed)
        result["safety"] = assert_safe(args.gpu_index, args.allow_busy)
        result["smoke"] = run_smoke(args)
        result["status"] = "ok"
    except Exception as exc:
        result["status"] = "skipped" if "skipped due to shared-server safety" in str(exc) else "failed"
        result["reason"] = str(exc)

    out = PROJECT_ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result["status"] == "ok":
        return 0
    if result["status"] == "skipped":
        return 3
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
