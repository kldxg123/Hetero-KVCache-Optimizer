#!/usr/bin/env python3
"""Safety-gated validation runner for 4090-24G survival experiments."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def run(cmd):
    return subprocess.run(cmd, check=False, capture_output=True, text=True)


def nvidia_smi_compute_apps():
    proc = run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    apps = []
    for line in proc.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4:
            digits = "".join(ch for ch in parts[3] if ch.isdigit())
            apps.append(
                {
                    "gpu_uuid": parts[0],
                    "pid": parts[1],
                    "process_name": parts[2],
                    "used_memory_mb": int(digits) if digits else 0,
                }
            )
    return apps


def nvidia_smi_gpus():
    proc = run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    rows = []
    for line in proc.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 6:
            rows.append(
                {
                    "index": int(parts[0]),
                    "uuid": parts[1],
                    "name": parts[2],
                    "memory_total_mb": int(parts[3]),
                    "memory_used_mb": int(parts[4]),
                    "utilization_gpu": int(parts[5]),
                }
            )
    return rows


def assert_target_gpu_safe(gpu_index: int, allow_busy: bool = False):
    gpus = nvidia_smi_gpus()
    target = next((g for g in gpus if g["index"] == gpu_index), None)
    if target is None:
        raise RuntimeError(f"GPU {gpu_index} not found")
    apps = nvidia_smi_compute_apps()
    target_apps = [app for app in apps if app["gpu_uuid"] == target["uuid"]]
    target_busy = bool(target_apps) or target["memory_used_mb"] > 1024 or target["utilization_gpu"] > 5
    if target_busy and not allow_busy:
        raise RuntimeError(
            "skipped due to shared-server safety: "
            f"GPU{gpu_index} used={target['memory_used_mb']}MB "
            f"util={target['utilization_gpu']}%, target_apps={target_apps}, all_apps={apps}"
        )
    return {"target": target, "target_apps": target_apps, "all_apps": apps}


def set_4090_cap(target_gib: float):
    import torch

    total = torch.cuda.get_device_properties(0).total_memory
    fraction = min(1.0, (target_gib * 1024**3) / float(total))
    torch.cuda.set_per_process_memory_fraction(fraction, device=0)
    return {
        "target_gib": target_gib,
        "a100_total_gib": total / 1024**3,
        "fraction": fraction,
    }


def static_check(args):
    import inspect

    from src.core.engine_wrapper import FusedHeteroCache
    from src.memory import query_aware_retriever

    src = inspect.getsource(query_aware_retriever)
    forbidden = ["mean_k_embedding", "cosine_similarity", "F.cosine_similarity"]
    return {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "fused_cache_default_triton": inspect.signature(FusedHeteroCache).parameters[
            "enable_triton"
        ].default,
        "query_retriever_forbidden_terms": {
            term: (term in src) for term in forbidden
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-index", type=int, default=1)
    parser.add_argument("--cap-gib", type=float, default=22.0)
    parser.add_argument("--allow-busy", action="store_true")
    parser.add_argument("--stage", choices=["static", "safety"], default="safety")
    parser.add_argument("--output", default="experiments/4090_24g_validation.json")
    args = parser.parse_args()

    result = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "stage": args.stage,
        "gpu_index": args.gpu_index,
        "cap_gib": args.cap_gib,
    }
    try:
        result["safety"] = assert_target_gpu_safe(args.gpu_index, args.allow_busy)
        if args.stage == "static":
            result["static"] = static_check(args)
    except Exception as exc:
        result["status"] = "skipped" if "skipped due to shared-server safety" in str(exc) else "failed"
        result["reason"] = str(exc)
    else:
        result["status"] = "ok"

    out = PROJECT_ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if result["status"] == "ok":
        return 0
    if result["status"] == "skipped":
        return 3
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
