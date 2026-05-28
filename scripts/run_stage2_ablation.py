#!/usr/bin/env python3
"""Stage-2 quality ablation for full KV vs short KV vs dot-product retrieval."""

import argparse
import json
import os
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_stage2_smoke import assert_safe, build_prompt, set_cap, set_seed


def run_generate(tokenizer, model, prompt, cache, max_new_tokens):
    import torch

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
    start = time.time()
    kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "use_cache": True,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if cache is not None:
        kwargs["past_key_values"] = cache
    with torch.inference_mode():
        output = model.generate(**inputs, **kwargs)
    elapsed = time.time() - start
    generated = tokenizer.decode(output[0, inputs.input_ids.shape[-1] :], skip_special_tokens=True)
    row = {
        "actual_input_tokens": int(inputs.input_ids.shape[-1]),
        "generated": generated.strip()[:200],
        "ok": "HETEROKV_SMOKE_OK" in generated,
        "elapsed_sec": elapsed,
        "max_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "max_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
    }
    del inputs, output
    torch.cuda.empty_cache()
    return row


def safe_run_generate(tokenizer, model, prompt, cache, max_new_tokens):
    import torch

    try:
        return run_generate(tokenizer, model, prompt, cache, max_new_tokens)
    except torch.cuda.OutOfMemoryError as exc:
        row = {
            "actual_input_tokens": None,
            "generated": "",
            "ok": False,
            "oom": True,
            "error": str(exc),
            "elapsed_sec": None,
            "max_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
            "max_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
        }
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
        return row
    except Exception as exc:
        row = {
            "actual_input_tokens": None,
            "generated": "",
            "ok": False,
            "oom": False,
            "error": str(exc),
            "elapsed_sec": None,
            "max_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
            "max_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
        }
        torch.cuda.empty_cache()
        return row


def build_cache(args, model, self_healing: bool, method_d: bool):
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
        self_healing=self_healing,
        adaptive_self_healing=False,
        enable_method_d=method_d,
    )


def run_ablation(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

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

    rows = []
    for target in args.lengths:
        prompt = build_prompt(target)
        full = safe_run_generate(tokenizer, model, prompt, None, args.max_new_tokens)
        full.update({"target_tokens": target, "mode": "full_kv_baseline"})
        rows.append(full)

    patched = patch_qwen2_attention_for_heterokv(model)
    for target in args.lengths:
        prompt = build_prompt(target)
        for mode, self_healing, method_d in [
            ("heterokv_no_retrieval", False, False),
            ("heterokv_dotproduct", True, True),
        ]:
            cache = build_cache(args, model, self_healing=self_healing, method_d=method_d)
            row = safe_run_generate(tokenizer, model, prompt, cache, args.max_new_tokens)
            row.update(
                {
                    "target_tokens": target,
                    "mode": mode,
                    "memory_summary": cache.memory_summary(),
                }
            )
            rows.append(row)
            del cache
            torch.cuda.empty_cache()

    summary = {}
    for row in rows:
        summary.setdefault(str(row["target_tokens"]), {})[row["mode"]] = {
            "ok": row.get("ok"),
            "oom": row.get("oom", False),
            "error": row.get("error"),
        }

    return {
        "cap": cap,
        "model_path": str(model_path),
        "patched_attention_modules": patched,
        "rows": rows,
        "summary": summary,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-index", type=int, default=1)
    parser.add_argument("--cap-gib", type=float, default=22.0)
    parser.add_argument("--allow-busy", action="store_true")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--lengths", type=int, nargs="+", default=[4096, 8192])
    parser.add_argument("--sink-tokens", type=int, default=64)
    parser.add_argument("--keep-tail", type=int, default=2048)
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--output", default="experiments/stage2_ablation.json")
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
        result["ablation"] = run_ablation(args)
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
