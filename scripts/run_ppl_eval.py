#!/usr/bin/env python3
"""Real WikiText-2 perplexity evaluation for full KV vs HeteroKV."""

import argparse
import json
import math
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def set_cap(cap_gib: float, gpu_index: int) -> float:
    torch.cuda.set_device(0)
    total = torch.cuda.get_device_properties(0).total_memory
    fraction = min(1.0, max(0.01, cap_gib * 1024**3 / total))
    torch.cuda.set_per_process_memory_fraction(fraction, device=0)
    return fraction


def gpu_compute_apps(gpu_index: int) -> Dict[int, int]:
    cmd = [
        "nvidia-smi",
        "--query-compute-apps=pid,used_memory",
        "--format=csv,noheader,nounits",
        "-i",
        str(gpu_index),
    ]
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return {}
    apps: Dict[int, int] = {}
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[0].isdigit():
            try:
                app_pid = int(parts[0])
                apps[app_pid] = apps.get(app_pid, 0) + int(parts[1])
            except ValueError:
                pass
    return apps


def gpu_total_used_mb(gpu_index: int) -> int:
    cmd = [
        "nvidia-smi",
        "--query-gpu=memory.used",
        "--format=csv,noheader,nounits",
        "-i",
        str(gpu_index),
    ]
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
        return int(out.strip().splitlines()[0].strip())
    except Exception:
        return 0


def gpu_total_mb(gpu_index: int) -> int:
    cmd = [
        "nvidia-smi",
        "--query-gpu=memory.total",
        "--format=csv,noheader,nounits",
        "-i",
        str(gpu_index),
    ]
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
        return int(out.strip().splitlines()[0].strip())
    except Exception:
        return 0


def select_gpu_by_free_memory(required_gib: float, reserve_gib: float) -> int:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,memory.total,memory.used",
        "--format=csv,noheader,nounits",
    ]
    out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
    candidates = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        idx, total, used = int(parts[0]), int(parts[1]), int(parts[2])
        free_after = total - used - int(required_gib * 1024) - int(reserve_gib * 1024)
        candidates.append((free_after, idx, total, used))
    candidates = [item for item in candidates if item[0] >= 0]
    if not candidates:
        raise RuntimeError(
            f"no GPU has enough free memory for required={required_gib}GiB "
            f"with reserve={reserve_gib}GiB"
        )
    candidates.sort(reverse=True)
    return int(candidates[0][1])


def start_monitor(
    gpu_index: int,
    max_vram_gib: float,
    interval: float,
    allow_other_processes_if_memory_fits: bool = False,
    gpu_total_memory_limit_gib: Optional[float] = None,
) -> Dict[str, object]:
    state: Dict[str, object] = {"peak_mb": 0, "peak_gpu_used_mb": 0, "samples": [], "stop": False}
    pid = os.getpid()

    def loop() -> None:
        limit = int(max_vram_gib * 1024)
        total_limit = None
        if gpu_total_memory_limit_gib is not None and gpu_total_memory_limit_gib > 0:
            total_limit = int(gpu_total_memory_limit_gib * 1024)
        while not state["stop"]:
            apps = gpu_compute_apps(gpu_index)
            mb = apps.get(pid, 0)
            gpu_used = gpu_total_used_mb(gpu_index)
            other_apps = {other_pid: used for other_pid, used in apps.items() if other_pid != pid}
            state["peak_mb"] = max(int(state["peak_mb"]), mb)
            state["peak_gpu_used_mb"] = max(int(state["peak_gpu_used_mb"]), gpu_used)
            state["samples"].append(
                {
                    "elapsed_sec": time.time() - start,
                    "process_memory_mb": mb,
                    "gpu_memory_used_mb": gpu_used,
                    "other_processes": other_apps,
                }
            )
            if other_apps and not allow_other_processes_if_memory_fits:
                print(f"[PPL][SAFETY] other GPU processes detected: {other_apps}", flush=True)
                os._exit(8)
            if total_limit is not None and gpu_used > total_limit:
                print(f"[PPL][FUSE] total GPU memory {gpu_used} MB > {total_limit} MB", flush=True)
                os._exit(10)
            if mb > limit:
                print(f"[PPL][FUSE] process memory {mb} MB > {limit} MB", flush=True)
                os._exit(9)
            time.sleep(interval)

    start = time.time()
    thread = threading.Thread(target=loop, daemon=True)
    thread.start()
    state["thread"] = thread
    return state


def load_wikitext_tokens(tokenizer, max_tokens: int) -> torch.Tensor:
    dataset = load_dataset(
        "wikitext",
        "wikitext-2-raw-v1",
        split="test",
        download_mode="reuse_dataset_if_exists",
    )
    text = "\n\n".join(row["text"] for row in dataset if row["text"].strip())
    ids = tokenizer(text, return_tensors="pt").input_ids
    if ids.shape[-1] < max_tokens + 1:
        raise RuntimeError(f"WikiText tokenized length {ids.shape[-1]} < requested {max_tokens + 1}")
    return ids[:, : max_tokens + 1]


@torch.no_grad()
def eval_full(model, ids: torch.Tensor, loss_start_token: int = 0) -> Dict[str, float]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    context_cpu = ids[:, :-1]
    targets_cpu = ids[:, 1:]
    loss_start_token = max(0, min(int(loss_start_token), context_cpu.shape[-1] - 1))
    start = time.time()

    if loss_start_token > 0:
        prefix = context_cpu[:, :loss_start_token].to("cuda:0")
        out = model(input_ids=prefix, use_cache=True, logits_to_keep=1)
        past = out.past_key_values
        del prefix, out
        total_nll = 0.0
        total_tokens = 0
        for token_pos in range(loss_start_token, context_cpu.shape[-1]):
            input_token = context_cpu[:, token_pos : token_pos + 1].to("cuda:0")
            target_token = targets_cpu[:, token_pos : token_pos + 1].to("cuda:0")
            out = model(
                input_ids=input_token,
                past_key_values=past,
                use_cache=True,
                logits_to_keep=1,
            )
            past = out.past_key_values
            loss_sum = F.cross_entropy(
                out.logits[:, -1, :].float(),
                target_token.reshape(-1),
                reduction="sum",
            )
            total_nll += float(loss_sum.item())
            total_tokens += int(target_token.numel())
            del input_token, target_token, out, loss_sum
        elapsed = time.time() - start
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3
        max_reserved = torch.cuda.max_memory_reserved() / 1024**3
        del past
        torch.cuda.empty_cache()
        return {
            "nll": total_nll,
            "tokens": total_tokens,
            "ppl": float(math.exp(total_nll / total_tokens)),
            "elapsed_sec": elapsed,
            "max_allocated_gib": max_allocated,
            "max_reserved_gib": max_reserved,
            "eval_style": "decode_suffix_full_kv",
        }

    context = context_cpu.to("cuda:0")
    targets = targets_cpu.to("cuda:0")
    logits = model(input_ids=context, use_cache=False).logits
    logits_for_loss = logits[:, loss_start_token:, :]
    targets_for_loss = targets[:, loss_start_token:]
    loss_sum = F.cross_entropy(
        logits_for_loss.reshape(-1, logits_for_loss.shape[-1]).float(),
        targets_for_loss.reshape(-1),
        reduction="sum",
    )
    tokens = int(targets_for_loss.numel())
    elapsed = time.time() - start
    return {
        "nll": float(loss_sum.item()),
        "tokens": tokens,
        "ppl": float(math.exp(loss_sum.item() / tokens)),
        "elapsed_sec": elapsed,
        "max_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "max_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
        "eval_style": "single_forward_full_kv",
    }


def build_heterokv_cache(model, args):
    from src.core.engine_wrapper import build_fused_cache

    return build_fused_cache(
        num_layers=getattr(model.config, "num_hidden_layers", None),
        device="cuda:0",
        sink_tokens=args.sink_tokens,
        keep_tail=args.keep_tail,
        chunk_size=args.chunk_size,
        enable_quant=True,
        enable_prefetch=False,
        enable_triton=False,
        self_healing=args.heterokv_self_healing,
        adaptive_self_healing=False,
        enable_method_d=args.enable_method_d,
        method_d_gate_margin=args.method_d_gate_margin,
        method_d_token_window=args.method_d_token_window,
        method_d_layer_min=args.method_d_layer_min,
        method_d_layer_max=args.method_d_layer_max,
        method_d_top_k=args.method_d_top_k,
        method_d_retrieval_bias=args.method_d_retrieval_bias,
        method_d_score_reduce=args.method_d_score_reduce,
        method_d_top_r=args.method_d_top_r,
        method_d_query_history_tokens=args.method_d_query_history_tokens,
        method_d_consensus_boost=args.method_d_consensus_boost,
        method_d_min_position=args.method_d_min_position,
        method_d_tail_guard_tokens=args.method_d_tail_guard_tokens,
        method_d_focus_radius=args.method_d_focus_radius,
        method_d_source_token_boost=args.method_d_source_token_boost,
        method_d_source_query_tokens=args.method_d_source_query_tokens,
        method_d_focus_bias=args.method_d_focus_bias,
        method_d_nonfocus_penalty=args.method_d_nonfocus_penalty,
        method_d_source_fusion_alpha=args.method_d_source_fusion_alpha,
        method_d_source_fusion_focus_only=args.method_d_source_fusion_focus_only,
        method_d_retrieve_focus_only=args.method_d_retrieve_focus_only,
        method_d_retrieve_focus_context_tokens=args.method_d_retrieve_focus_context_tokens,
        method_d_source_gate_bypass_threshold=args.method_d_source_gate_bypass_threshold,
        method_d_reuse_gate_bypass=args.method_d_reuse_gate_bypass,
        method_d_reuse_kv_cache=args.method_d_reuse_kv_cache,
        method_d_triton_scoring=args.method_d_triton_scoring,
        method_d_triton_scoring_batch_chunks=args.method_d_triton_scoring_batch_chunks,
    )


@torch.no_grad()
def eval_heterokv_chunked(model, ids: torch.Tensor, args) -> Dict[str, float]:
    from src.core.fused_attention_patch import patch_qwen2_attention_for_heterokv

    patch_qwen2_attention_for_heterokv(model)
    cache = build_heterokv_cache(model, args)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    context = ids[:, :-1].to("cuda:0")
    targets = ids[:, 1:].to("cuda:0")
    total_nll = 0.0
    total_tokens = 0
    start = time.time()
    for chunk_start in range(0, context.shape[-1], args.chunk_size):
        chunk_end = min(context.shape[-1], chunk_start + args.chunk_size)
        chunk_ids = context[:, chunk_start:chunk_end]
        chunk_targets = targets[:, chunk_start:chunk_end]
        position_ids = torch.arange(
            chunk_start, chunk_end, dtype=torch.long, device="cuda:0"
        ).unsqueeze(0)
        cache_position = torch.arange(chunk_start, chunk_end, dtype=torch.long, device="cuda:0")
        attention_mask = torch.ones((1, chunk_end), dtype=torch.long, device="cuda:0")
        if args.method_d_source_token_boost > 0 and hasattr(cache, "set_source_token_ids"):
            cache.set_source_token_ids(ids[0, :chunk_end].detach().cpu())
        logits = model(
            input_ids=chunk_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_position=cache_position,
            past_key_values=cache,
            use_cache=True,
        ).logits
        loss_sum = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(),
            chunk_targets.reshape(-1),
            reduction="sum",
        )
        total_nll += float(loss_sum.item())
        total_tokens += int(chunk_targets.numel())
        del chunk_ids, chunk_targets, position_ids, cache_position, attention_mask, logits, loss_sum
    elapsed = time.time() - start
    return {
        "nll": total_nll,
        "tokens": total_tokens,
        "ppl": float(math.exp(total_nll / total_tokens)),
        "elapsed_sec": elapsed,
        "max_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "max_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
        "method_d_event_count": len(cache.get_method_d_events()) if hasattr(cache, "get_method_d_events") else 0,
    }


@torch.no_grad()
def eval_heterokv_decode_suffix(model, ids: torch.Tensor, args) -> Dict[str, float]:
    from src.core.fused_attention_patch import patch_qwen2_attention_for_heterokv

    patch_qwen2_attention_for_heterokv(model)
    cache = build_heterokv_cache(model, args)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    context = ids[:, :-1]
    targets = ids[:, 1:]
    context_len = context.shape[-1]
    prefix_tokens = max(1, min(int(args.eval_prefix_tokens), context_len - 1))
    total_nll = 0.0
    total_tokens = 0
    start = time.time()

    for chunk_start in range(0, prefix_tokens, args.chunk_size):
        chunk_end = min(prefix_tokens, chunk_start + args.chunk_size)
        chunk_ids = context[:, chunk_start:chunk_end].to("cuda:0")
        position_ids = torch.arange(chunk_start, chunk_end, dtype=torch.long, device="cuda:0").unsqueeze(0)
        cache_position = torch.arange(chunk_start, chunk_end, dtype=torch.long, device="cuda:0")
        attention_mask = torch.ones((1, chunk_end), dtype=torch.long, device="cuda:0")
        if args.method_d_source_token_boost > 0 and hasattr(cache, "set_source_token_ids"):
            cache.set_source_token_ids(ids[0, :chunk_end].detach().cpu())
        _ = model(
            input_ids=chunk_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_position=cache_position,
            past_key_values=cache,
            use_cache=True,
        )
        del chunk_ids, position_ids, cache_position, attention_mask

    for token_pos in range(prefix_tokens, context_len):
        input_token = context[:, token_pos : token_pos + 1].to("cuda:0")
        target_token = targets[:, token_pos : token_pos + 1].to("cuda:0")
        position_ids = torch.tensor([[token_pos]], dtype=torch.long, device="cuda:0")
        cache_position = torch.tensor([token_pos], dtype=torch.long, device="cuda:0")
        attention_mask = torch.ones((1, token_pos + 1), dtype=torch.long, device="cuda:0")
        if args.method_d_source_token_boost > 0 and hasattr(cache, "set_source_token_ids"):
            cache.set_source_token_ids(ids[0, : token_pos + 1].detach().cpu())
        logits = model(
            input_ids=input_token,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_position=cache_position,
            past_key_values=cache,
            use_cache=True,
        ).logits
        loss_sum = F.cross_entropy(
            logits[:, -1, :].float(),
            target_token.reshape(-1),
            reduction="sum",
        )
        total_nll += float(loss_sum.item())
        total_tokens += int(target_token.numel())
        del input_token, target_token, position_ids, cache_position, attention_mask, logits, loss_sum

    elapsed = time.time() - start
    events = cache.get_method_d_events() if hasattr(cache, "get_method_d_events") else []
    summary = cache.memory_summary() if hasattr(cache, "memory_summary") else {}
    return {
        "nll": total_nll,
        "tokens": total_tokens,
        "ppl": float(math.exp(total_nll / total_tokens)),
        "elapsed_sec": elapsed,
        "max_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "max_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
        "eval_prefix_tokens": prefix_tokens,
        "method_d_event_count": len(events),
        "method_d_events_tail": events[-20:],
        "memory_summary": summary,
    }


@torch.no_grad()
def eval_heterokv(model, ids: torch.Tensor, args) -> Dict[str, float]:
    if args.heterokv_eval_style == "decode_suffix":
        return eval_heterokv_decode_suffix(model, ids, args)
    return eval_heterokv_chunked(model, ids, args)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-index", type=int, default=1)
    parser.add_argument("--auto-select-gpu", action="store_true")
    parser.add_argument("--cap-gib", type=float, default=22.0)
    parser.add_argument("--max-vram-gib", type=float, default=30.0)
    parser.add_argument("--gpu-reserve-gib", type=float, default=4.0)
    parser.add_argument("--allow-other-processes-if-memory-fits", action="store_true")
    parser.add_argument("--monitor-interval-sec", type=float, default=5.0)
    parser.add_argument(
        "--attn-implementation",
        choices=["eager", "sdpa", "flash_attention_2"],
        default="sdpa",
        help=(
            "Attention backend for full-KV/PPL probes. SDPA is the default "
            "because eager attention can materialize oversized temporaries and "
            "turn a PPL comparison into a test-configuration OOM."
        ),
    )
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--modes", nargs="+", choices=["full", "heterokv"], default=["full", "heterokv"])
    parser.add_argument("--sink-tokens", type=int, default=64)
    parser.add_argument("--keep-tail", type=int, default=2048)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--heterokv-eval-style", choices=["chunked", "decode_suffix"], default="chunked")
    parser.add_argument("--eval-prefix-tokens", type=int, default=2048)
    parser.add_argument("--heterokv-self-healing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--enable-method-d", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--method-d-gate-margin", type=float, default=1.10)
    parser.add_argument("--method-d-token-window", type=int, default=0)
    parser.add_argument("--method-d-layer-min", type=int, default=0)
    parser.add_argument("--method-d-layer-max", type=int, default=None)
    parser.add_argument("--method-d-top-k", type=int, default=None)
    parser.add_argument("--method-d-retrieval-bias", type=float, default=0.0)
    parser.add_argument("--method-d-score-reduce", default="max")
    parser.add_argument("--method-d-top-r", type=int, default=8)
    parser.add_argument("--method-d-query-history-tokens", type=int, default=1)
    parser.add_argument("--method-d-consensus-boost", type=float, default=0.0)
    parser.add_argument("--method-d-min-position", type=int, default=0)
    parser.add_argument("--method-d-tail-guard-tokens", type=int, default=0)
    parser.add_argument("--method-d-focus-radius", type=int, default=0)
    parser.add_argument("--method-d-source-token-boost", type=float, default=0.0)
    parser.add_argument("--method-d-source-query-tokens", type=int, default=64)
    parser.add_argument("--method-d-focus-bias", type=float, default=0.0)
    parser.add_argument("--method-d-nonfocus-penalty", type=float, default=0.0)
    parser.add_argument("--method-d-source-fusion-alpha", type=float, default=0.0)
    parser.add_argument("--method-d-source-fusion-focus-only", action="store_true")
    parser.add_argument("--method-d-retrieve-focus-only", action="store_true")
    parser.add_argument("--method-d-retrieve-focus-context-tokens", type=int, default=0)
    parser.add_argument("--method-d-source-gate-bypass-threshold", type=float, default=0.0)
    parser.add_argument("--method-d-reuse-gate-bypass", action="store_true")
    parser.add_argument("--method-d-reuse-kv-cache", action="store_true")
    parser.add_argument("--method-d-triton-scoring", action="store_true")
    parser.add_argument("--method-d-triton-scoring-batch-chunks", type=int, default=8)
    parser.add_argument("--output", default="experiments/ppl_eval.json")
    args = parser.parse_args()

    if args.auto_select_gpu:
        args.gpu_index = select_gpu_by_free_memory(
            required_gib=max(float(args.cap_gib), float(args.max_vram_gib)),
            reserve_gib=float(args.gpu_reserve_gib),
        )
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)
        print(f"[PPL][SAFETY] auto-selected GPU{args.gpu_index}", flush=True)

    total_mb = gpu_total_mb(args.gpu_index)
    used_mb = gpu_total_used_mb(args.gpu_index)
    required_mb = int(max(float(args.cap_gib), float(args.max_vram_gib)) * 1024)
    reserve_mb = int(float(args.gpu_reserve_gib) * 1024)
    if total_mb and used_mb + required_mb + reserve_mb > total_mb:
        raise RuntimeError(
            "insufficient remaining GPU memory: "
            f"GPU{args.gpu_index} used={used_mb}MB total={total_mb}MB "
            f"required={required_mb}MB reserve={reserve_mb}MB"
        )

    cap_fraction = set_cap(args.cap_gib, args.gpu_index)
    monitor = start_monitor(
        args.gpu_index,
        args.max_vram_gib,
        args.monitor_interval_sec,
        allow_other_processes_if_memory_fits=args.allow_other_processes_if_memory_fits,
        gpu_total_memory_limit_gib=max(
            0.0,
            (total_mb / 1024.0 - float(args.gpu_reserve_gib)) if total_mb else 0.0,
        ),
    )
    model_path = PROJECT_ROOT / "models" / "Qwen2.5-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        trust_remote_code=True,
        attn_implementation=args.attn_implementation,
    )
    model.eval()
    ids = load_wikitext_tokens(tokenizer, args.max_tokens)
    results = {
        "model": str(model_path),
        "dataset": "wikitext-2-raw-v1/test",
        "max_tokens": args.max_tokens,
        "loss_start_token": args.eval_prefix_tokens if args.heterokv_eval_style == "decode_suffix" else 0,
        "heterokv_eval_style": args.heterokv_eval_style,
        "attn_implementation": args.attn_implementation,
        "cap_gib": args.cap_gib,
        "cap_fraction": cap_fraction,
        "gpu_index": args.gpu_index,
        "gpu_memory_total_mb": total_mb,
        "gpu_memory_used_before_mb": used_mb,
        "gpu_reserve_gib": args.gpu_reserve_gib,
        "allow_other_processes_if_memory_fits": args.allow_other_processes_if_memory_fits,
        "method_d_config": {
            "heterokv_self_healing": args.heterokv_self_healing,
            "enable_method_d": args.enable_method_d,
            "score_reduce": args.method_d_score_reduce,
            "top_r": args.method_d_top_r,
            "query_history_tokens": args.method_d_query_history_tokens,
            "top_k": args.method_d_top_k,
            "token_window": args.method_d_token_window,
            "consensus_boost": args.method_d_consensus_boost,
            "min_position": args.method_d_min_position,
            "focus_radius": args.method_d_focus_radius,
            "source_token_boost": args.method_d_source_token_boost,
            "source_query_tokens": args.method_d_source_query_tokens,
            "focus_bias": args.method_d_focus_bias,
            "nonfocus_penalty": args.method_d_nonfocus_penalty,
            "source_fusion_alpha": args.method_d_source_fusion_alpha,
            "source_fusion_focus_only": args.method_d_source_fusion_focus_only,
            "retrieve_focus_only": args.method_d_retrieve_focus_only,
            "retrieve_focus_context_tokens": args.method_d_retrieve_focus_context_tokens,
            "source_gate_bypass_threshold": args.method_d_source_gate_bypass_threshold,
            "reuse_gate_bypass": args.method_d_reuse_gate_bypass,
            "reuse_kv_cache": args.method_d_reuse_kv_cache,
            "triton_scoring": args.method_d_triton_scoring,
            "triton_scoring_batch_chunks": args.method_d_triton_scoring_batch_chunks,
        },
        "modes": {},
    }
    if "full" in args.modes:
        results["modes"]["full"] = eval_full(model, ids, results["loss_start_token"])
    if "heterokv" in args.modes:
        results["modes"]["heterokv"] = eval_heterokv(model, ids, args)
    if "full" in results["modes"] and "heterokv" in results["modes"]:
        full = results["modes"]["full"]["ppl"]
        hetero = results["modes"]["heterokv"]["ppl"]
        results["relative_ppl_delta"] = (hetero - full) / full
    monitor["stop"] = True
    time.sleep(min(0.2, args.monitor_interval_sec))
    results["monitor"] = {
        "peak_process_memory_mb": int(monitor["peak_mb"]),
        "peak_gpu_memory_used_mb": int(monitor["peak_gpu_used_mb"]),
        "samples_tail": monitor["samples"][-20:],
    }
    out = PROJECT_ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
