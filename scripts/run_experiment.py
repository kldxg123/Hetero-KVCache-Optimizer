#!/usr/bin/env python3
"""Workflow 1.5 experiment orchestrator.

This runner intentionally separates cheap sanity checks from GPU deployment.
It parses refine-logs/EXPERIMENT_PLAN.md, runs code-review gates by default,
executes minimal sanity tests, and updates an experiment tracker.
"""

import argparse
import json
import logging
import os
import random
import re
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = PROJECT_ROOT / "refine-logs" / "EXPERIMENT_PLAN.md"
TRACKER_PATH = PROJECT_ROOT / "experiments" / "experiment_tracker.json"
TRACKER_JSONL = PROJECT_ROOT / "experiments" / "experiment_tracker.jsonl"


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def run_command(
    cmd: List[str],
    timeout: int,
    env: Optional[Dict[str, str]] = None,
    cwd: Path = PROJECT_ROOT,
) -> Dict[str, object]:
    logging.info("running: %s", " ".join(cmd))
    start = time.time()
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    elapsed = time.time() - start
    result = {
        "cmd": cmd,
        "returncode": proc.returncode,
        "elapsed_sec": elapsed,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }
    if proc.returncode != 0:
        logging.warning("command failed rc=%s: %s", proc.returncode, " ".join(cmd))
    return result


def parse_memory_mb(value: str) -> int:
    digits = "".join(ch for ch in value if ch.isdigit())
    return int(digits) if digits else 0


def query_target_gpu_uuid(gpu_index: int) -> Optional[str]:
    proc = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"nvidia-smi GPU query failed: {proc.stderr.strip()}")
    for line in proc.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[0] == str(gpu_index):
            return parts[1]
    raise RuntimeError(f"GPU {gpu_index} not found in nvidia-smi output")


def process_group_pids(root_pid: int) -> List[int]:
    try:
        pgid = os.getpgid(root_pid)
    except Exception:
        return [root_pid]
    proc = subprocess.run(
        ["ps", "-eo", "pid=,pgid="],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return [root_pid]
    pids = []
    for line in proc.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
            if int(parts[1]) == pgid:
                pids.append(int(parts[0]))
    return pids or [root_pid]


def query_gpu_process_memory(gpu_index: int, pids: List[int]) -> Dict[str, object]:
    target_uuid = query_target_gpu_uuid(gpu_index)
    pid_set = {str(pid) for pid in pids}
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
    if proc.returncode != 0:
        raise RuntimeError(f"nvidia-smi compute-app query failed: {proc.stderr.strip()}")
    entries = []
    total_mb = 0
    for line in proc.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        entry = {
            "gpu_uuid": parts[0],
            "pid": parts[1],
            "process_name": parts[2],
            "used_memory_mb": parse_memory_mb(parts[3]),
        }
        if parts[0] == target_uuid and parts[1] in pid_set:
            entries.append(entry)
            total_mb += entry["used_memory_mb"]
    return {
        "gpu_index": gpu_index,
        "gpu_uuid": target_uuid,
        "pids": pids,
        "process_memory_mb": total_mb,
        "entries": entries,
    }


def terminate_process_group(proc: subprocess.Popen) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except Exception:
        try:
            proc.terminate()
        except Exception:
            pass
    try:
        proc.wait(timeout=15)
    except Exception:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass


def run_monitored_command(
    cmd: List[str],
    timeout: int,
    env: Optional[Dict[str, str]],
    gpu_index: int,
    max_vram_gib: float,
    monitor_interval_sec: float,
    cwd: Path = PROJECT_ROOT,
) -> Dict[str, object]:
    logging.info(
        "running with GPU monitor: %s | max_vram_gib=%.2f",
        " ".join(cmd),
        max_vram_gib,
    )
    start = time.time()
    samples = []
    max_process_mb = 0
    killed_by_monitor = False
    kill_kind = None
    kill_reason = None
    with tempfile.NamedTemporaryFile("w+", encoding="utf-8") as stdout_file, tempfile.NamedTemporaryFile(
        "w+", encoding="utf-8"
    ) as stderr_file:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=stdout_file,
            stderr=stderr_file,
            text=True,
            start_new_session=True,
        )
        deadline = start + timeout
        threshold_mb = int(max_vram_gib * 1024)
        while proc.poll() is None:
            try:
                pids = process_group_pids(proc.pid)
                sample = query_gpu_process_memory(gpu_index, pids)
            except Exception as exc:
                killed_by_monitor = True
                kill_kind = "monitor_error"
                kill_reason = str(exc)
                logging.error("GPU monitor failed closed: %s", kill_reason)
                terminate_process_group(proc)
                break
            sample["elapsed_sec"] = time.time() - start
            samples.append(sample)
            max_process_mb = max(max_process_mb, int(sample["process_memory_mb"]))
            if int(sample["process_memory_mb"]) > threshold_mb:
                killed_by_monitor = True
                kill_kind = "vram"
                kill_reason = (
                    f"process GPU memory {sample['process_memory_mb']}MB exceeded "
                    f"{threshold_mb}MB fuse"
                )
                logging.error(kill_reason)
                terminate_process_group(proc)
                break
            if time.time() > deadline:
                killed_by_monitor = True
                kill_kind = "timeout"
                kill_reason = f"timeout after {timeout}s"
                logging.error(kill_reason)
                terminate_process_group(proc)
                break
            time.sleep(max(0.5, monitor_interval_sec))

        returncode = proc.poll()
        stdout_file.seek(0)
        stderr_file.seek(0)
        stdout = stdout_file.read()
        stderr = stderr_file.read()

    elapsed = time.time() - start
    result = {
        "cmd": cmd,
        "returncode": returncode if returncode is not None else 124,
        "elapsed_sec": elapsed,
        "stdout_tail": stdout[-4000:],
        "stderr_tail": stderr[-4000:],
        "monitor": {
            "enabled": True,
            "gpu_index": gpu_index,
            "max_vram_gib": max_vram_gib,
            "monitor_interval_sec": monitor_interval_sec,
            "killed_by_monitor": killed_by_monitor,
            "kill_kind": kill_kind,
            "kill_reason": kill_reason,
            "max_process_memory_gib": max_process_mb / 1024,
            "sample_count": len(samples),
            "samples_tail": samples[-20:],
        },
    }
    if result["returncode"] != 0:
        logging.warning("command failed rc=%s: %s", result["returncode"], " ".join(cmd))
    return result


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def parse_plan(path: Path = PLAN_PATH) -> Dict[str, object]:
    text = path.read_text(encoding="utf-8")
    headings = []
    for line in text.splitlines():
        match = re.match(r"^(#{1,3})\s+(.+)$", line)
        if match:
            headings.append({"level": len(match.group(1)), "title": match.group(2)})
    stages = [h["title"] for h in headings if str(h["title"]).startswith("Stage ")]
    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "heading_count": len(headings),
        "stages": stages,
        "has_deferred_tests": "Deferred Tests" in text,
    }


def read_json_result(relative_path: str) -> Optional[Dict[str, object]]:
    path = PROJECT_ROOT / relative_path
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def is_shared_safety_skip(payload: Optional[Dict[str, object]]) -> bool:
    if not payload:
        return False
    return payload.get("status") == "skipped" and "shared-server safety" in str(
        payload.get("reason", "")
    )


def status_from_stage_result(result: Dict[str, object]) -> str:
    if result.get("passed"):
        return "ok"
    if result.get("status") == "skipped":
        return "skipped"
    return "failed"


def code_review_gates(args) -> Dict[str, object]:
    """Deterministic local review gates; external GPT review is a documented hook."""
    py_files = [
        "src/memory/manager.py",
        "src/memory/query_aware_retriever.py",
        "src/memory/attention_competition_queue.py",
        "src/memory/dram_storage.py",
        "src/core/engine_wrapper.py",
        "src/core/fused_attention_patch.py",
        "tests/test_heterokv_stage1.py",
        "scripts/validate_4090_24g_survival.py",
        "scripts/run_stage2_smoke.py",
        "scripts/run_niah_eval.py",
        "scripts/run_stage2_ablation.py",
        "scripts/run_experiment.py",
    ]
    results = []
    results.append(
        run_command(
            [sys.executable, "-m", "py_compile", *py_files],
            timeout=args.timeout,
        )
    )
    results.append(run_command(["git", "diff", "--check"], timeout=args.timeout))

    forbidden = {
        "full_prefill_return": "return key_states, value_states",
        "mean_k_embedding": "mean_k_embedding",
        "cosine_similarity": "cosine_similarity",
        "triton_default_true": "enable_triton: bool = True",
    }
    file_text = "\n".join(
        (PROJECT_ROOT / p).read_text(encoding="utf-8", errors="replace")
        for p in [
            "src/memory/manager.py",
            "src/memory/query_aware_retriever.py",
            "src/core/engine_wrapper.py",
        ]
    )
    forbidden_hits = {name: (pattern in file_text) for name, pattern in forbidden.items()}

    passed = all(r["returncode"] == 0 for r in results) and not any(forbidden_hits.values())
    return {
        "name": "local_static_code_review",
        "external_review_hook": "GPT-5.4/GPT-5.5 review should inspect this tracker entry before GPU-heavy runs.",
        "passed": passed,
        "commands": results,
        "forbidden_hits": forbidden_hits,
    }


def stage_sanity(args) -> Dict[str, object]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)
    safety_output = "experiments/4090_24g_safety_gate.json"
    commands = [
        [
            sys.executable,
            "scripts/validate_4090_24g_survival.py",
            "--stage",
            "safety",
            "--gpu-index",
            str(args.gpu_index),
            "--output",
            safety_output,
        ],
        [sys.executable, "-m", "pytest", "-q", "tests/test_heterokv_stage1.py"],
    ]
    results = [run_command(cmd, timeout=args.timeout, env=env) for cmd in commands]
    safety_payload = read_json_result(safety_output)
    safety_ok = results[0]["returncode"] == 0 or is_shared_safety_skip(safety_payload)
    pytest_ok = results[-1]["returncode"] == 0
    return {
        "name": "sanity",
        "passed": safety_ok and pytest_ok,
        "status": "ok" if safety_ok and pytest_ok else "failed",
        "safety_gate_status": safety_payload.get("status") if safety_payload else "missing",
        "safety_skip_allowed": is_shared_safety_skip(safety_payload),
        "commands": results,
    }


def stage_stage2(args) -> Dict[str, object]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)
    output_path = "experiments/stage2_smoke.json"
    cmd = [
        sys.executable,
        "scripts/run_stage2_smoke.py",
        "--gpu-index",
        str(args.gpu_index),
        "--cap-gib",
        str(args.cap_gib),
        "--attn-implementation",
        args.niah_attn_implementation,
        "--seed",
        str(args.seed),
        "--output",
        output_path,
    ]
    if args.allow_busy:
        cmd.append("--allow-busy")
    result = run_monitored_command(
        cmd,
        timeout=args.timeout,
        env=env,
        gpu_index=args.gpu_index,
        max_vram_gib=args.max_vram_gib,
        monitor_interval_sec=args.monitor_interval_sec,
    )
    payload = read_json_result(output_path)
    child_status = payload.get("status") if payload else "missing"
    monitor = result.get("monitor", {})
    if monitor.get("killed_by_monitor"):
        child_status = "failed"
    return {
        "name": "stage2_smoke",
        "passed": result["returncode"] == 0,
        "status": child_status,
        "safety_skip_allowed": is_shared_safety_skip(payload),
        "monitor_max_process_memory_gib": monitor.get("max_process_memory_gib"),
        "monitor_killed_by_monitor": monitor.get("killed_by_monitor"),
        "monitor_kill_reason": monitor.get("kill_reason"),
        "result_path": output_path,
        "commands": [result],
    }


def stage_niah(args) -> Dict[str, object]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)
    output_path = "experiments/niah_eval.json"
    cmd = [
        sys.executable,
        "scripts/run_niah_eval.py",
        "--gpu-index",
        str(args.gpu_index),
        "--cap-gib",
        str(args.cap_gib),
        "--seed",
        str(args.seed),
        "--lengths",
        *[str(v) for v in args.niah_lengths],
        "--depths",
        *[str(v) for v in args.niah_depths],
        "--trials",
        str(args.niah_trials),
        "--modes",
        *args.niah_modes,
        "--primary-mode",
        args.niah_primary_mode,
        "--sink-tokens",
        str(args.niah_sink_tokens),
        "--keep-tail",
        str(args.niah_keep_tail),
        "--chunk-size",
        str(args.niah_chunk_size),
        "--method-d-gate-margin",
        str(args.niah_method_d_gate_margin),
        "--method-d-token-window",
        str(args.niah_method_d_token_window),
        "--method-d-layer-min",
        str(args.niah_method_d_layer_min),
        "--method-d-retrieval-bias",
        str(args.niah_method_d_retrieval_bias),
        "--method-d-score-reduce",
        args.niah_method_d_score_reduce,
        "--method-d-top-r",
        str(args.niah_method_d_top_r),
        "--method-d-query-history-tokens",
        str(args.niah_method_d_query_history_tokens),
        "--method-d-consensus-boost",
        str(args.niah_method_d_consensus_boost),
        "--method-d-min-position",
        str(args.niah_method_d_min_position),
        "--method-d-tail-guard-tokens",
        str(args.niah_method_d_tail_guard_tokens),
        "--method-d-focus-radius",
        str(args.niah_method_d_focus_radius),
        "--method-d-source-token-boost",
        str(args.niah_method_d_source_token_boost),
        "--method-d-source-query-tokens",
        str(args.niah_method_d_source_query_tokens),
        "--method-d-focus-bias",
        str(args.niah_method_d_focus_bias),
        "--method-d-nonfocus-penalty",
        str(args.niah_method_d_nonfocus_penalty),
        "--method-d-source-fusion-alpha",
        str(args.niah_method_d_source_fusion_alpha),
        "--method-d-source-fusion-low-alpha",
        str(args.niah_method_d_source_fusion_low_alpha),
        "--method-d-source-fusion-source-threshold",
        str(args.niah_method_d_source_fusion_source_threshold),
        "--method-d-source-cue-answer-tokens",
        str(args.niah_method_d_source_cue_answer_tokens),
        "--method-d-retrieve-focus-only" if args.niah_method_d_retrieve_focus_only else None,
        "--method-d-retrieve-focus-context-tokens",
        str(args.niah_method_d_retrieve_focus_context_tokens),
        "--method-d-reuse-ttl-tokens",
        str(args.niah_method_d_reuse_ttl_tokens),
        "--method-d-reuse-source-threshold",
        str(args.niah_method_d_reuse_source_threshold),
        "--method-d-triton-scoring-batch-chunks",
        str(args.niah_method_d_triton_scoring_batch_chunks),
        "--max-new-tokens",
        str(args.niah_max_new_tokens),
        "--heterokv-decode-suffix-tokens",
        str(args.niah_decode_suffix_tokens),
        "--output",
        output_path,
    ]
    if args.niah_prefill_keep_tail is not None:
        cmd.extend(["--prefill-keep-tail", str(args.niah_prefill_keep_tail)])
    if args.niah_decode_keep_tail is not None:
        cmd.extend(["--decode-keep-tail", str(args.niah_decode_keep_tail)])
    if args.niah_method_d_layer_max is not None:
        cmd.extend(["--method-d-layer-max", str(args.niah_method_d_layer_max)])
    if args.niah_method_d_top_k is not None:
        cmd.extend(["--method-d-top-k", str(args.niah_method_d_top_k)])
    if args.niah_method_d_require_source_overlap:
        cmd.append("--method-d-require-source-overlap")
    if args.niah_method_d_allow_source_before_min_position:
        cmd.append("--method-d-allow-source-before-min-position")
    if args.niah_method_d_source_fusion_focus_only:
        cmd.append("--method-d-source-fusion-focus-only")
    if args.niah_method_d_source_cue_focus:
        cmd.append("--method-d-source-cue-focus")
    cmd = [item for item in cmd if item is not None]
    if args.niah_method_d_reuse_kv_cache:
        cmd.append("--method-d-reuse-kv-cache")
    if args.niah_method_d_triton_scoring:
        cmd.append("--method-d-triton-scoring")
    if args.niah_no_attention_mask:
        cmd.append("--heterokv-no-attention-mask")
    if args.niah_diagnostic_bf16_dram:
        cmd.append("--diagnostic-bf16-dram")
    if args.niah_fullkv_manual_decode:
        cmd.append("--fullkv-manual-decode")
    if args.allow_busy:
        cmd.append("--allow-busy")
    result = run_monitored_command(
        cmd,
        timeout=args.timeout,
        env=env,
        gpu_index=args.gpu_index,
        max_vram_gib=args.max_vram_gib,
        monitor_interval_sec=args.monitor_interval_sec,
    )
    payload = read_json_result(output_path)
    child_status = payload.get("status") if payload else "missing"
    monitor = result.get("monitor", {})
    if monitor.get("killed_by_monitor"):
        child_status = "failed"
    niah = payload.get("niah", {}) if payload else {}
    return {
        "name": "niah_eval",
        "passed": result["returncode"] == 0,
        "status": child_status,
        "accuracy": niah.get("accuracy"),
        "passed_accuracy": niah.get("passed_accuracy"),
        "correct": niah.get("correct"),
        "total": niah.get("total"),
        "safety_skip_allowed": is_shared_safety_skip(payload),
        "monitor_max_process_memory_gib": monitor.get("max_process_memory_gib"),
        "monitor_killed_by_monitor": monitor.get("killed_by_monitor"),
        "monitor_kill_reason": monitor.get("kill_reason"),
        "result_path": output_path,
        "commands": [result],
    }


def stage_ablation(args) -> Dict[str, object]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)
    output_path = "experiments/stage2_ablation.json"
    cmd = [
        sys.executable,
        "scripts/run_stage2_ablation.py",
        "--gpu-index",
        str(args.gpu_index),
        "--cap-gib",
        str(args.cap_gib),
        "--seed",
        str(args.seed),
        "--lengths",
        *[str(v) for v in args.ablation_lengths],
        "--output",
        output_path,
    ]
    if args.allow_busy:
        cmd.append("--allow-busy")
    result = run_monitored_command(
        cmd,
        timeout=args.timeout,
        env=env,
        gpu_index=args.gpu_index,
        max_vram_gib=args.max_vram_gib,
        monitor_interval_sec=args.monitor_interval_sec,
    )
    payload = read_json_result(output_path)
    child_status = payload.get("status") if payload else "missing"
    monitor = result.get("monitor", {})
    if monitor.get("killed_by_monitor"):
        child_status = "failed"
    ablation = payload.get("ablation", {}) if payload else {}
    return {
        "name": "stage2_ablation",
        "passed": result["returncode"] == 0,
        "status": child_status,
        "summary": ablation.get("summary"),
        "safety_skip_allowed": is_shared_safety_skip(payload),
        "monitor_max_process_memory_gib": monitor.get("max_process_memory_gib"),
        "monitor_killed_by_monitor": monitor.get("killed_by_monitor"),
        "monitor_kill_reason": monitor.get("kill_reason"),
        "result_path": output_path,
        "commands": [result],
    }


def stage_gpu(args) -> Dict[str, object]:
    if args.suite == "stage2":
        return stage_stage2(args)
    if args.suite == "niah":
        return stage_niah(args)
    if args.suite == "ablation":
        return stage_ablation(args)
    if args.suite != "stage2":
        return {
            "name": "gpu_deploy",
            "passed": False,
            "status": "not_implemented",
            "reason": (
                f"suite={args.suite} is intentionally not launched by this lightweight "
                "orchestrator yet. Add the dedicated 16K/32K/128K runner first."
            ),
        }
    return stage_stage2(args)


def load_tracker() -> Dict[str, object]:
    if TRACKER_PATH.exists():
        return json.loads(TRACKER_PATH.read_text(encoding="utf-8"))
    return {"runs": []}


def update_tracker(entry: Dict[str, object]) -> None:
    TRACKER_PATH.parent.mkdir(parents=True, exist_ok=True)
    TRACKER_JSONL.parent.mkdir(parents=True, exist_ok=True)
    tracker = load_tracker()
    tracker.setdefault("runs", []).append(entry)
    tracker["last_updated"] = entry["timestamp"]
    TRACKER_PATH.write_text(json.dumps(tracker, indent=2, ensure_ascii=False), encoding="utf-8")
    with TRACKER_JSONL.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Workflow 1.5 experiment stages.")
    parser.add_argument(
        "--stage",
        choices=["parse-plan", "sanity", "stage2", "gpu"],
        default="sanity",
    )
    parser.add_argument("--suite", choices=["stage2", "ablation", "survival", "niah", "ppl", "latency"], default="stage2")
    parser.add_argument("--gpu-index", type=int, default=1)
    parser.add_argument("--cap-gib", type=float, default=22.0)
    parser.add_argument("--max-vram-gib", type=float, default=30.0)
    parser.add_argument("--monitor-interval-sec", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--niah-attn-implementation",
        choices=["eager", "sdpa", "flash_attention_2"],
        default="eager",
    )
    parser.add_argument("--niah-lengths", type=int, nargs="+", default=[4096, 8192])
    parser.add_argument("--niah-depths", type=float, nargs="+", default=[0.25, 0.5, 0.75, 0.9])
    parser.add_argument("--niah-trials", type=int, default=1)
    parser.add_argument(
        "--niah-modes",
        nargs="+",
        choices=[
            "full_kv_baseline",
            "heterokv_no_retrieval",
            "heterokv_dotproduct",
            "heterokv_oracle_retrieval",
        ],
        default=["heterokv_dotproduct"],
    )
    parser.add_argument("--niah-primary-mode", default="heterokv_dotproduct")
    parser.add_argument("--niah-sink-tokens", type=int, default=64)
    parser.add_argument("--niah-keep-tail", type=int, default=2048)
    parser.add_argument("--niah-prefill-keep-tail", type=int, default=None)
    parser.add_argument("--niah-decode-keep-tail", type=int, default=None)
    parser.add_argument("--niah-chunk-size", type=int, default=2048)
    parser.add_argument("--niah-method-d-gate-margin", type=float, default=1.10)
    parser.add_argument("--niah-method-d-token-window", type=int, default=0)
    parser.add_argument("--niah-method-d-layer-min", type=int, default=0)
    parser.add_argument("--niah-method-d-layer-max", type=int, default=None)
    parser.add_argument("--niah-method-d-top-k", type=int, default=None)
    parser.add_argument("--niah-method-d-retrieval-bias", type=float, default=0.0)
    parser.add_argument(
        "--niah-method-d-score-reduce",
        choices=[
            "max",
            "top_r_mean",
            "head_mean_max",
            "head_mean_top_r_mean",
            "query_top_r_mean",
            "query_mean_max",
            "z_score_max",
            "peak_contrast",
        ],
        default="max",
    )
    parser.add_argument("--niah-method-d-top-r", type=int, default=8)
    parser.add_argument("--niah-method-d-query-history-tokens", type=int, default=1)
    parser.add_argument("--niah-method-d-consensus-boost", type=float, default=0.0)
    parser.add_argument("--niah-method-d-min-position", type=int, default=0)
    parser.add_argument("--niah-method-d-tail-guard-tokens", type=int, default=0)
    parser.add_argument("--niah-method-d-focus-radius", type=int, default=0)
    parser.add_argument("--niah-method-d-source-token-boost", type=float, default=0.0)
    parser.add_argument("--niah-method-d-source-query-tokens", type=int, default=64)
    parser.add_argument("--niah-method-d-require-source-overlap", action="store_true")
    parser.add_argument("--niah-method-d-allow-source-before-min-position", action="store_true")
    parser.add_argument("--niah-method-d-focus-bias", type=float, default=0.0)
    parser.add_argument("--niah-method-d-nonfocus-penalty", type=float, default=0.0)
    parser.add_argument("--niah-method-d-source-fusion-alpha", type=float, default=0.0)
    parser.add_argument("--niah-method-d-source-fusion-low-alpha", type=float, default=0.0)
    parser.add_argument("--niah-method-d-source-fusion-source-threshold", type=float, default=0.0)
    parser.add_argument("--niah-method-d-source-fusion-focus-only", action="store_true")
    parser.add_argument("--niah-method-d-source-cue-focus", action="store_true")
    parser.add_argument("--niah-method-d-source-cue-answer-tokens", type=int, default=8)
    parser.add_argument("--niah-method-d-retrieve-focus-only", action="store_true")
    parser.add_argument("--niah-method-d-retrieve-focus-context-tokens", type=int, default=0)
    parser.add_argument("--niah-method-d-reuse-ttl-tokens", type=int, default=0)
    parser.add_argument("--niah-method-d-reuse-source-threshold", type=float, default=0.0)
    parser.add_argument("--niah-method-d-reuse-kv-cache", action="store_true")
    parser.add_argument("--niah-method-d-triton-scoring", action="store_true")
    parser.add_argument("--niah-method-d-triton-scoring-batch-chunks", type=int, default=8)
    parser.add_argument("--niah-no-attention-mask", action="store_true")
    parser.add_argument("--niah-diagnostic-bf16-dram", action="store_true")
    parser.add_argument("--niah-max-new-tokens", type=int, default=24)
    parser.add_argument("--niah-fullkv-manual-decode", action="store_true")
    parser.add_argument("--niah-decode-suffix-tokens", type=int, default=1)
    parser.add_argument("--ablation-lengths", type=int, nargs="+", default=[4096, 8192])
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--allow-busy", action="store_true")
    parser.add_argument("--code-review", dest="code_review", action="store_true", default=True)
    parser.add_argument("--no-code-review", dest="code_review", action="store_false")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--tracker", default=str(TRACKER_PATH))
    parser.add_argument("--tracker-jsonl", default=None)
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    global TRACKER_PATH, TRACKER_JSONL
    TRACKER_PATH = Path(args.tracker)
    TRACKER_JSONL = Path(args.tracker_jsonl) if args.tracker_jsonl else TRACKER_PATH.with_suffix(".jsonl")

    setup_logging(args.log_level)
    set_seed(args.seed)

    entry: Dict[str, object] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "stage": args.stage,
        "suite": args.suite,
        "seed": args.seed,
        "gpu_index": args.gpu_index,
        "cap_gib": args.cap_gib,
        "max_vram_gib": args.max_vram_gib,
        "monitor_interval_sec": args.monitor_interval_sec,
        "code_review_enabled": args.code_review,
        "plan": parse_plan(),
        "results": [],
    }

    try:
        if args.code_review and args.stage != "parse-plan":
            review = code_review_gates(args)
            entry["results"].append(review)
            if not review["passed"]:
                entry["status"] = "failed"
                entry["reason"] = "code_review_gates_failed"
                update_tracker(entry)
                print(json.dumps(entry, indent=2, ensure_ascii=False))
                return 2

        if args.stage == "parse-plan":
            entry["status"] = "ok"
        elif args.stage == "sanity":
            entry["results"].append(stage_sanity(args))
            entry["status"] = status_from_stage_result(entry["results"][-1])
        elif args.stage == "stage2":
            entry["results"].append(stage_stage2(args))
            entry["status"] = status_from_stage_result(entry["results"][-1])
        elif args.stage == "gpu":
            entry["results"].append(stage_gpu(args))
            entry["status"] = status_from_stage_result(entry["results"][-1])
    except subprocess.TimeoutExpired as exc:
        entry["status"] = "failed"
        entry["reason"] = f"timeout: {exc}"
    except Exception as exc:
        entry["status"] = "failed"
        entry["reason"] = repr(exc)

    update_tracker(entry)
    print(json.dumps(entry, indent=2, ensure_ascii=False))
    if entry["status"] == "ok":
        return 0
    if entry["status"] == "skipped":
        return 3
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
