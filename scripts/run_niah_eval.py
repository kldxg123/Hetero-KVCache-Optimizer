#!/usr/bin/env python3
"""Safety-gated Needle-in-a-Haystack evaluation for HeteroKV."""

import argparse
import json
import os
import random
import re
import string
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_stage2_smoke import assert_safe, set_cap, set_seed


def random_code(rng: random.Random) -> str:
    return "".join(rng.choice(string.digits) for _ in range(6))


def build_niah_prompt(tokenizer, target_tokens: int, depth: float, code: str) -> tuple[str, int, int, int]:
    filler_unit = (
        "Archive line: routine benchmark text for a long-context cache test. "
        "This line contains no target code and has no bearing on the answer. "
    )
    needle = (
        f"\n[NEEDLE]\nThe target code is {code}.\n[/NEEDLE]\n"
        f"Verification copy: target_code={code}.\n"
    )
    question = "\nQuestion: What is the target code? Reply with only the exact 6 digits.\n"

    def apply_chat(user_content: str) -> str:
        messages = [
            {
                "role": "system",
                "content": "You answer extraction questions. Return only the exact requested digits.",
            },
            {"role": "user", "content": user_content},
        ]
        if getattr(tokenizer, "chat_template", None):
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return user_content

    filler_ids = tokenizer(filler_unit, add_special_tokens=False).input_ids
    base_prompt = apply_chat(needle + question)
    base_len = len(tokenizer(base_prompt, add_special_tokens=False).input_ids)
    filler_budget = max(128, target_tokens - base_len)

    prompt = base_prompt
    actual_len = base_len
    for _ in range(4):
        depth_clamped = max(0.0, min(1.0, depth))
        before_budget = int(filler_budget * depth_clamped)
        after_budget = max(0, filler_budget - before_budget)
        before = (filler_ids * ((before_budget // len(filler_ids)) + 1))[:before_budget]
        after = (filler_ids * ((after_budget // len(filler_ids)) + 1))[:after_budget]
        user_content = (
            tokenizer.decode(before, skip_special_tokens=True)
            + needle
            + tokenizer.decode(after, skip_special_tokens=True)
            + question
        )
        prompt = apply_chat(user_content)
        actual_len = len(tokenizer(prompt, add_special_tokens=False).input_ids)
        delta = target_tokens - actual_len
        if abs(delta) <= 32 or delta <= 0:
            break
        filler_budget += delta

    enc = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    offsets = getattr(enc, "offset_mapping", None)
    code_start = prompt.find(code)
    if offsets is not None and code_start >= 0:
        code_end = code_start + len(code)
        token_start = 0
        token_end = len(enc.input_ids)
        for idx, (start, end) in enumerate(offsets):
            if end > code_start:
                token_start = idx
                break
        for idx, (start, end) in enumerate(offsets[token_start:], start=token_start):
            if start >= code_end:
                token_end = idx
                break
        return prompt, len(enc.input_ids), token_start, token_end

    return prompt, actual_len, -1, -1


def answer_matches(code: str, generated: str) -> tuple[bool, bool]:
    exact = code in generated
    digit_normalized = code in re.sub(r"\D", "", generated)
    return exact or digit_normalized, exact


def build_cache(args, model, mode: str, needle_range=None):
    if mode == "full_kv_baseline":
        return None
    from src.core.engine_wrapper import build_fused_cache

    retrieval_modes = {"heterokv_dotproduct", "heterokv_oracle_retrieval"}
    cache = build_fused_cache(
        num_layers=getattr(model.config, "num_hidden_layers", None),
        device="cuda:0",
        sink_tokens=args.sink_tokens,
        keep_tail=args.prefill_keep_tail or args.keep_tail,
        chunk_size=args.chunk_size,
        enable_quant=True,
        enable_prefetch=False,
        enable_triton=False,
        self_healing=(mode in retrieval_modes),
        adaptive_self_healing=False,
        enable_method_d=(mode in retrieval_modes),
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
        method_d_require_source_overlap=args.method_d_require_source_overlap,
        method_d_allow_source_before_min_position=args.method_d_allow_source_before_min_position,
        method_d_focus_bias=args.method_d_focus_bias,
        method_d_nonfocus_penalty=args.method_d_nonfocus_penalty,
        method_d_source_fusion_alpha=args.method_d_source_fusion_alpha,
        method_d_source_fusion_low_alpha=args.method_d_source_fusion_low_alpha,
        method_d_source_fusion_source_threshold=args.method_d_source_fusion_source_threshold,
        method_d_source_fusion_focus_only=args.method_d_source_fusion_focus_only,
        method_d_source_cue_focus=args.method_d_source_cue_focus,
        method_d_source_cue_answer_tokens=args.method_d_source_cue_answer_tokens,
        method_d_retrieve_focus_only=args.method_d_retrieve_focus_only,
        method_d_retrieve_focus_context_tokens=args.method_d_retrieve_focus_context_tokens,
        method_d_reuse_ttl_tokens=args.method_d_reuse_ttl_tokens,
        method_d_reuse_source_threshold=args.method_d_reuse_source_threshold,
        method_d_source_gate_bypass_threshold=args.method_d_source_gate_bypass_threshold,
        method_d_reuse_gate_bypass=args.method_d_reuse_gate_bypass,
        method_d_reuse_kv_cache=args.method_d_reuse_kv_cache,
        method_d_triton_scoring=args.method_d_triton_scoring,
        method_d_triton_scoring_batch_chunks=args.method_d_triton_scoring_batch_chunks,
        diagnostic_bf16_dram=args.diagnostic_bf16_dram,
    )
    if mode == "heterokv_oracle_retrieval" and hasattr(cache, "set_method_d_oracle_range"):
        cache.set_method_d_oracle_range(needle_range)
    return cache


def run_one_case(tokenizer, model, prompt, cache, args):
    import torch

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    actual_tokens = int(inputs.input_ids.shape[-1])
    decode_path = "hf_generate"

    if cache is None and args.fullkv_manual_decode:
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        generated_ids = []
        decode_path = "fullkv_prefill_decode"
        with torch.inference_mode():
            prefill_start = time.time()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
            )
            prefill_elapsed = time.time() - prefill_start
            past = outputs.past_key_values
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated_ids.append(next_token)
            current = next_token
            del outputs

            decode_start = time.time()
            decode_forwards = 0
            for _ in range(max(0, int(args.max_new_tokens) - 1)):
                attention_mask = torch.cat(
                    [
                        attention_mask,
                        torch.ones(
                            (attention_mask.shape[0], 1),
                            dtype=attention_mask.dtype,
                            device=attention_mask.device,
                        ),
                    ],
                    dim=-1,
                )
                outputs = model(
                    input_ids=current,
                    attention_mask=attention_mask,
                    past_key_values=past,
                    use_cache=True,
                )
                past = outputs.past_key_values
                next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated_ids.append(next_token)
                current = next_token
                decode_forwards += 1
                del outputs
                if (
                    tokenizer.eos_token_id is not None
                    and int(next_token.item()) == tokenizer.eos_token_id
                ):
                    break
            decode_elapsed = time.time() - decode_start

        elapsed = time.time() - start
        generated_tensor = torch.cat(generated_ids, dim=-1)
        generated = tokenizer.decode(generated_tensor[0], skip_special_tokens=True).strip()
        perf = {
            "total_sec": elapsed,
            "prefill_sec": prefill_elapsed,
            "decode_sec": decode_elapsed,
            "prefill_tokens": int(input_ids.shape[-1]),
            "decode_steps": int(decode_forwards),
            "generated_tokens": int(generated_tensor.shape[-1]),
            "decode_ms_per_step": (decode_elapsed / max(1, int(decode_forwards))) * 1000.0,
        }
        del inputs, input_ids, attention_mask, current, generated_tensor, past
        return generated, elapsed, actual_tokens, decode_path, perf

    if cache is None or not args.heterokv_chunked_decode:
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
        generated = tokenizer.decode(
            output[0, inputs.input_ids.shape[-1] :],
            skip_special_tokens=True,
        ).strip()
        perf = {
            "total_sec": elapsed,
            "prefill_sec": None,
            "decode_sec": elapsed,
            "decode_steps": int(args.max_new_tokens),
            "decode_ms_per_step": (elapsed / max(1, int(args.max_new_tokens))) * 1000.0,
        }
        del inputs, output
        return generated, elapsed, actual_tokens, decode_path, perf

    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    if hasattr(cache, "set_source_token_ids"):
        cache.set_source_token_ids(input_ids[0].detach().cpu())
    if args.method_d_source_cue_focus and hasattr(cache, "set_source_cue_token_ids"):
        cue_texts = [
            "The target code is ",
            " target_code=",
            "target_code=",
        ]
        cues = [
            tokenizer(text, add_special_tokens=False).input_ids
            for text in cue_texts
        ]
        cache.set_source_cue_token_ids(
            cues,
            answer_tokens=args.method_d_source_cue_answer_tokens,
        )
    seq_len = input_ids.shape[-1]
    decode_suffix = max(1, min(int(args.heterokv_decode_suffix_tokens), seq_len))
    decode_path = f"chunked_prefill_decode_suffix_{decode_suffix}"
    prefill_end = max(seq_len - decode_suffix, 0)
    device = input_ids.device

    with torch.inference_mode():
        prefill_start = time.time()
        for chunk_start in range(0, prefill_end, args.chunk_size):
            chunk_end = min(prefill_end, chunk_start + args.chunk_size)
            chunk_ids = input_ids[:, chunk_start:chunk_end]
            position_ids = torch.arange(
                chunk_start, chunk_end, dtype=torch.long, device=device
            ).unsqueeze(0)
            cache_position = torch.arange(
                chunk_start, chunk_end, dtype=torch.long, device=device
            )
            chunk_mask = None if args.heterokv_no_attention_mask else attention_mask[:, :chunk_end]
            outputs = model(
                input_ids=chunk_ids,
                attention_mask=chunk_mask,
                position_ids=position_ids,
                cache_position=cache_position,
                past_key_values=cache,
                use_cache=True,
            )
            del outputs, chunk_ids, position_ids, cache_position, chunk_mask
        prefill_elapsed = time.time() - prefill_start

        if args.decode_keep_tail is not None and hasattr(cache, "force_shrink_hbm_budget"):
            cache.force_shrink_hbm_budget(args.decode_keep_tail)
            decode_path += f"_shrink_to_{args.decode_keep_tail}"

        current = input_ids[:, prefill_end:prefill_end + 1]
        generated_ids = []
        total_decode_steps = decode_suffix + args.max_new_tokens
        decode_start = time.time()
        for step in range(total_decode_steps):
            pos = prefill_end + step
            position_ids = torch.tensor([[pos]], dtype=torch.long, device=device)
            cache_position = torch.tensor([pos], dtype=torch.long, device=device)
            step_mask = None
            if not args.heterokv_no_attention_mask:
                step_mask = torch.ones(
                    (input_ids.shape[0], pos + 1), dtype=attention_mask.dtype, device=device
                )
            outputs = model(
                input_ids=current,
                attention_mask=step_mask,
                position_ids=position_ids,
                cache_position=cache_position,
                past_key_values=cache,
                use_cache=True,
            )
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            if step < decode_suffix - 1:
                current = input_ids[:, prefill_end + step + 1:prefill_end + step + 2]
            else:
                generated_ids.append(next_token)
                current = next_token
            del outputs, position_ids, cache_position, step_mask
            if (
                step >= decode_suffix - 1
                and tokenizer.eos_token_id is not None
                and int(next_token.item()) == tokenizer.eos_token_id
            ):
                break
        decode_elapsed = time.time() - decode_start

    elapsed = time.time() - start
    generated_tensor = torch.cat(generated_ids, dim=-1) if generated_ids else current[:, :0]
    generated = tokenizer.decode(generated_tensor[0], skip_special_tokens=True).strip()
    perf = {
        "total_sec": elapsed,
        "prefill_sec": prefill_elapsed,
        "decode_sec": decode_elapsed,
        "prefill_tokens": int(prefill_end),
        "decode_steps": int(step + 1 if "step" in locals() else 0),
        "decode_suffix_tokens": int(decode_suffix),
        "generated_tokens": int(generated_tensor.shape[-1]),
        "decode_ms_per_step": (decode_elapsed / max(1, int(step + 1 if "step" in locals() else 0))) * 1000.0,
    }
    del inputs, input_ids, attention_mask, current, generated_tensor
    return generated, elapsed, actual_tokens, decode_path, perf


def run_niah(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from src.core.fused_attention_patch import patch_qwen2_attention_for_heterokv

    rng = random.Random(args.seed)
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
        attn_implementation=args.attn_implementation,
    )
    model.eval()
    patched = 0

    rows = []
    by_mode = {
        mode: {"correct": 0, "total": 0, "accuracy": 0.0}
        for mode in args.modes
    }
    for target_tokens in args.lengths:
        for depth in args.depths:
            for trial in range(args.trials):
                code = random_code(rng)
                prompt, _, needle_start, needle_end = build_niah_prompt(
                    tokenizer, target_tokens, depth, code
                )
                for mode in args.modes:
                    if mode != "full_kv_baseline" and patched == 0:
                        patched = patch_qwen2_attention_for_heterokv(model)
                    needle_range = (needle_start, needle_end)
                    cache = build_cache(args, model, mode, needle_range=needle_range)
                    row = {
                        "mode": mode,
                        "target_tokens": target_tokens,
                        "depth": depth,
                        "trial": trial,
                        "code": code,
                        "needle_token_range": [needle_start, needle_end],
                    }
                    try:
                        generated, elapsed, actual_tokens, decode_path, perf = run_one_case(
                            tokenizer, model, prompt, cache, args
                        )
                        ok, exact_ok = answer_matches(code, generated)
                        row.update(
                            {
                                "actual_input_tokens": actual_tokens,
                                "generated": generated[:200],
                                "correct": ok,
                                "exact_correct": exact_ok,
                                "decode_path": decode_path,
                                "elapsed_sec": elapsed,
                                "latency_breakdown": perf,
                                "max_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
                                "max_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
                            }
                        )
                        if cache is not None:
                            row["memory_summary"] = cache.memory_summary()
                            if hasattr(cache, "get_method_d_events"):
                                events = cache.get_method_d_events()
                                row["method_d_event_count"] = len(events)
                                row["method_d_events_tail"] = events[-512:]
                                row["method_d_gate_allowed_count"] = sum(
                                    1 for event in events if event.get("gate_allowed")
                                )
                                row["method_d_selected_chunk_count"] = sum(
                                    len(event.get("selected_chunks", [])) for event in events
                                )
                            if hasattr(cache, "get_attention_probe_events"):
                                probes = cache.get_attention_probe_events()
                                row["attention_probe_event_count"] = len(probes)
                                row["attention_probe_events_tail"] = probes[-512:]
                        by_mode[mode]["correct"] += int(ok)
                        by_mode[mode]["total"] += 1
                    except torch.cuda.OutOfMemoryError as exc:
                        row.update(
                            {
                                "actual_input_tokens": None,
                                "generated": "",
                                "correct": False,
                                "oom": True,
                                "error": str(exc),
                                "max_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
                                "max_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
                            }
                        )
                        by_mode[mode]["total"] += 1
                    except Exception as exc:
                        row.update(
                            {
                                "actual_input_tokens": None,
                                "generated": "",
                                "correct": False,
                                "oom": False,
                                "error": str(exc),
                                "max_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
                                "max_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
                            }
                        )
                        by_mode[mode]["total"] += 1
                    rows.append(row)
                    del cache
                    torch.cuda.empty_cache()

    for stats in by_mode.values():
        stats["accuracy"] = stats["correct"] / stats["total"] if stats["total"] else 0.0
    primary = by_mode.get(args.primary_mode, {"correct": 0, "total": 0, "accuracy": 0.0})
    return {
        "cap": cap,
        "model_path": str(model_path),
        "attn_implementation": args.attn_implementation,
        "patched_attention_modules": patched,
        "primary_mode": args.primary_mode,
        "accuracy": primary["accuracy"],
        "correct": primary["correct"],
        "total": primary["total"],
        "passed_accuracy": primary["accuracy"] >= args.min_accuracy,
        "by_mode": by_mode,
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-index", type=int, default=1)
    parser.add_argument("--cap-gib", type=float, default=22.0)
    parser.add_argument("--allow-busy", action="store_true")
    parser.add_argument(
        "--attn-implementation",
        choices=["eager", "sdpa", "flash_attention_2"],
        default="eager",
        help=(
            "Attention backend for unpatched model runs. Keep eager for the "
            "current HeteroKV patch path; use sdpa/flash_attention_2 for fair "
            "FullKV latency/OOM baseline probes when supported."
        ),
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--lengths", type=int, nargs="+", default=[4096, 8192])
    parser.add_argument("--depths", type=float, nargs="+", default=[0.25, 0.5, 0.75, 0.9])
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=[
            "full_kv_baseline",
            "heterokv_no_retrieval",
            "heterokv_dotproduct",
            "heterokv_oracle_retrieval",
        ],
        default=["heterokv_dotproduct"],
    )
    parser.add_argument("--primary-mode", default="heterokv_dotproduct")
    parser.add_argument("--min-accuracy", type=float, default=0.95)
    parser.add_argument("--sink-tokens", type=int, default=64)
    parser.add_argument("--keep-tail", type=int, default=2048)
    parser.add_argument("--prefill-keep-tail", type=int, default=None)
    parser.add_argument("--decode-keep-tail", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--method-d-gate-margin", type=float, default=1.10)
    parser.add_argument("--method-d-token-window", type=int, default=0)
    parser.add_argument("--method-d-layer-min", type=int, default=0)
    parser.add_argument("--method-d-layer-max", type=int, default=None)
    parser.add_argument("--method-d-top-k", type=int, default=None)
    parser.add_argument("--method-d-retrieval-bias", type=float, default=0.0)
    parser.add_argument(
        "--method-d-score-reduce",
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
    parser.add_argument("--method-d-top-r", type=int, default=8)
    parser.add_argument("--method-d-query-history-tokens", type=int, default=1)
    parser.add_argument("--method-d-consensus-boost", type=float, default=0.0)
    parser.add_argument("--method-d-min-position", type=int, default=0)
    parser.add_argument("--method-d-tail-guard-tokens", type=int, default=0)
    parser.add_argument("--method-d-focus-radius", type=int, default=0)
    parser.add_argument("--method-d-source-token-boost", type=float, default=0.0)
    parser.add_argument("--method-d-source-query-tokens", type=int, default=64)
    parser.add_argument("--method-d-require-source-overlap", action="store_true")
    parser.add_argument("--method-d-allow-source-before-min-position", action="store_true")
    parser.add_argument("--method-d-focus-bias", type=float, default=0.0)
    parser.add_argument("--method-d-nonfocus-penalty", type=float, default=0.0)
    parser.add_argument("--method-d-source-fusion-alpha", type=float, default=0.0)
    parser.add_argument("--method-d-source-fusion-low-alpha", type=float, default=0.0)
    parser.add_argument("--method-d-source-fusion-source-threshold", type=float, default=0.0)
    parser.add_argument("--method-d-source-fusion-focus-only", action="store_true")
    parser.add_argument("--method-d-source-cue-focus", action="store_true")
    parser.add_argument("--method-d-source-cue-answer-tokens", type=int, default=8)
    parser.add_argument("--method-d-retrieve-focus-only", action="store_true")
    parser.add_argument("--method-d-retrieve-focus-context-tokens", type=int, default=0)
    parser.add_argument("--method-d-reuse-ttl-tokens", type=int, default=0)
    parser.add_argument("--method-d-reuse-source-threshold", type=float, default=0.0)
    parser.add_argument("--method-d-source-gate-bypass-threshold", type=float, default=0.0)
    parser.add_argument("--method-d-reuse-gate-bypass", action="store_true")
    parser.add_argument("--method-d-reuse-kv-cache", action="store_true")
    parser.add_argument("--method-d-triton-scoring", action="store_true")
    parser.add_argument("--method-d-triton-scoring-batch-chunks", type=int, default=8)
    parser.add_argument("--diagnostic-bf16-dram", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument(
        "--fullkv-manual-decode",
        action="store_true",
        help="For full-KV baselines, split prefill and token-by-token decode timing.",
    )
    parser.add_argument("--heterokv-chunked-decode", action="store_true", default=True)
    parser.add_argument("--no-heterokv-chunked-decode", dest="heterokv_chunked_decode", action="store_false")
    parser.add_argument("--heterokv-decode-suffix-tokens", type=int, default=1)
    parser.add_argument("--heterokv-no-attention-mask", action="store_true")
    parser.add_argument("--output", default="experiments/niah_eval.json")
    args = parser.parse_args()
    if args.primary_mode not in args.modes:
        parser.error("--primary-mode must be included in --modes")

    result = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "gpu_index": args.gpu_index,
        "cap_gib": args.cap_gib,
        "seed": args.seed,
        "lengths_requested": args.lengths,
        "depths_requested": args.depths,
        "trials": args.trials,
        "modes": args.modes,
        "primary_mode": args.primary_mode,
    }
    try:
        set_seed(args.seed)
        result["safety"] = assert_safe(args.gpu_index, args.allow_busy)
        result["niah"] = run_niah(args)
        result["status"] = "ok" if result["niah"]["passed_accuracy"] else "quality_failed"
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
    if result["status"] == "quality_failed":
        return 4
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
