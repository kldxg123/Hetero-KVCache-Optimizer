#!/usr/bin/env python3
"""
test_quantization_degradation.py
================================
Rigorous test for 4-bit KV quantization accuracy degradation under FORCED eviction.

Fixes from v1:
  - Correct NIAH construction: prefix_filler + needle + suffix_filler (needle guaranteed in input)
  - Token-level assembly ensures needle is never truncated
  - 24GB memory cap simulates edge device

Configs tested (all with sink=64):
  - baseline:  Native HF DynamicCache (all KV in BF16)
  - hk_tail4k: keep_tail=4096 (no eviction at 4K — control group)
  - hk_tail2k: keep_tail=2048 (moderate eviction)
  - hk_tail1k: keep_tail=1024 (heavy eviction)
  - hk_tail512: keep_tail=512 (aggressive eviction)
  - hk_tail256: keep_tail=256 (extreme eviction)
"""

import os, sys, gc, time, json, random, warnings
import torch
import numpy as np

warnings.filterwarnings('ignore')
random.seed(42)
np.random.seed(42)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.core.engine_wrapper import build_fused_cache

DEVICE = "cuda:0"
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "models", "Qwen2.5-7B-Instruct")
MEMORY_FRACTION = 24.0 / 80.0

CONFIGS = [
    {"name": "baseline",   "keep_tail": None,  "enable_quant": False, "self_healing": False, "desc": "Native HF (BF16, no eviction)"},
    {"name": "hk_tail4k",  "keep_tail": 4096,  "enable_quant": True,  "self_healing": True,  "desc": "tail=4096 (no eviction at 4K)"},
    {"name": "hk_tail2k",  "keep_tail": 2048,  "enable_quant": True,  "self_healing": True,  "desc": "tail=2048 (moderate eviction)"},
    {"name": "hk_tail1k",  "keep_tail": 1024,  "enable_quant": True,  "self_healing": True,  "desc": "tail=1024 (heavy eviction)"},
    {"name": "hk_tail512", "keep_tail": 512,   "enable_quant": True,  "self_healing": True,  "desc": "tail=512 (aggressive eviction)"},
    {"name": "hk_tail256", "keep_tail": 256,   "enable_quant": True,  "self_healing": True,  "desc": "tail=256 (extreme eviction)"},
    {"name": "hk1k_NOheal","keep_tail": 1024,  "enable_quant": True,  "self_healing": False, "desc": "tail=1024 NO self-healing"},
    {"name": "hk256_NOheal","keep_tail": 256,  "enable_quant": True,  "self_healing": False, "desc": "tail=256 NO self-healing"},
]

FILLER = [
    "The sustainability report highlights a 15% reduction in carbon emissions across manufacturing facilities.",
    "Classical musicians in Vienna performed a sold-out concert featuring works by Mozart and Brahms.",
    "Marine biologists cataloged 47 new species of deep-sea fish during the Pacific expedition.",
    "The stock market index rose by 230 points following positive employment data from the labor department.",
    "Urban planners approved the construction of a 12-acre public park featuring native plant gardens.",
    "Cryptographic researchers demonstrated a novel lattice-based encryption scheme resistant to quantum attacks.",
    "The documentary about ancient Egyptian hieroglyphics won the best historical film award at Cannes.",
    "Agricultural engineers developed a drought-resistant wheat variety suitable for semi-arid climates.",
    "The city council debated zoning changes that would allow mixed-use development near transit stations.",
    "Neuroscientists identified a neural circuit responsible for risk-averse behavior in primates.",
    "The space agency confirmed the discovery of organic molecules on the surface of Europa.",
    "Professional chess players competed in the rapid tournament using a new Swiss-pairing system.",
    "Pharmaceutical companies announced Phase 3 clinical trial results for a novel antiviral medication.",
    "The geological survey mapped previously unknown fault lines beneath the metropolitan area.",
    "Digital artists showcased generative AI artwork at the contemporary museum of visual arts.",
]

NEEDLES = [
    ("The unique identifier for this session is UNICORN-42-FALCON.", "unicorn-42-falcon", "unicorn", "falcon"),
    ("The access code provided by the administrator is MERCURY-9-VENUS.", "mercury-9-venus", "mercury", "venus"),
    ("The project verification key is CRYSTAL-3-OPAL.", "crystal-3-opal", "crystal", "opal"),
]


def build_niah_input(tokenizer, target_tokens: int, needle: str, needle_keywords: tuple,
                     depth: float) -> dict:
    """Build NIAH input with needle guaranteed in the tokenized output."""
    sys_part = "<|im_start|>system\nAnswer questions based on the context.<|im_end|>\n<|im_start|>user\n"
    q_part = "\n\nWhat is the unique identifier or code mentioned in the text above? Reply with it exactly.<|im_end|>\n<|im_start|>assistant\n"

    sys_ids = tokenizer.encode(sys_part)
    q_ids = tokenizer.encode(q_part)
    needle_ids = tokenizer.encode(needle)

    filler_budget = target_tokens - len(sys_ids) - len(q_ids) - len(needle_ids)
    if filler_budget < 100:
        filler_budget = 100

    prefix_budget = max(10, int(filler_budget * depth))
    suffix_budget = filler_budget - prefix_budget

    # Build prefix filler to exact token budget
    random.seed(42)
    prefix_ids = []
    idx = 0
    while len(prefix_ids) < prefix_budget:
        sent_ids = tokenizer.encode(FILLER[idx % len(FILLER)])
        prefix_ids.extend(sent_ids)
        idx += 1
    prefix_ids = prefix_ids[:prefix_budget]

    # Build suffix filler
    suffix_ids = []
    while len(suffix_ids) < suffix_budget:
        sent_ids = tokenizer.encode(FILLER[idx % len(FILLER)])
        suffix_ids.extend(sent_ids)
        idx += 1
    suffix_ids = suffix_ids[:suffix_budget]

    # Assemble: sys + prefix + needle + suffix + question
    all_ids = sys_ids + prefix_ids + needle_ids + suffix_ids + q_ids

    # Verify needle presence
    has_needle = False
    needle_pos = -1.0
    for i in range(max(0, len(all_ids) - len(needle_ids) + 1)):
        if all_ids[i:i + len(needle_ids)] == needle_ids:
            has_needle = True
            needle_pos = i / len(all_ids)
            break

    return {
        "input_ids": torch.tensor([all_ids], dtype=torch.long),
        "attention_mask": torch.ones(1, len(all_ids), dtype=torch.long),
        "length": len(all_ids),
        "has_needle": has_needle,
        "needle_pos": needle_pos,
    }


@torch.inference_mode()
def run_single(model, tokenizer, config: dict, input_ids: torch.Tensor,
               attention_mask: torch.Tensor, max_gen: int = 16) -> dict:
    """Run a single inference with given config."""
    input_ids = input_ids.to(DEVICE)
    attention_mask = attention_mask.to(DEVICE)
    input_len = input_ids.shape[1]
    num_layers = len(model.model.layers)

    if config["keep_tail"] is not None:
        cache = build_fused_cache(
            num_layers=num_layers, sink_tokens=64,
            keep_tail=config["keep_tail"], device=DEVICE,
            enable_quant=config["enable_quant"], group_size=128,
            enable_prefetch=True,
            self_healing=config.get("self_healing", False),
        )
    else:
        cache = None

    torch.cuda.reset_peak_memory_stats(DEVICE)
    torch.cuda.synchronize(DEVICE)
    t0 = time.time()

    oom = False
    gen_text = ""
    peak_mem = 0.0
    try:
        outputs = model.generate(
            input_ids=input_ids, attention_mask=attention_mask,
            max_new_tokens=max_gen, num_beams=1, do_sample=False,
            use_cache=True, past_key_values=cache,
            pad_token_id=tokenizer.eos_token_id,
        )
        torch.cuda.synchronize(DEVICE)
        elapsed = time.time() - t0
        peak_mem = torch.cuda.max_memory_allocated(DEVICE) / 1024**3
        gen_text = tokenizer.decode(outputs[0, input_len:], skip_special_tokens=True)
    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "CUDA" in str(e):
            oom = True
            elapsed = time.time() - t0
        else:
            raise
    finally:
        del cache
        gc.collect()
        torch.cuda.empty_cache()

    return {
        "config": config["name"],
        "input_tokens": input_len,
        "generated_text": gen_text,
        "peak_mem_gb": round(peak_mem, 3),
        "time_s": round(elapsed, 3),
        "oom": oom,
    }


def main():
    print("=" * 80)
    print(" Hetero-KV Quantization Degradation Test (v2)")
    print(f" Simulating 24GB edge device | {4}x A100 80GB")
    print("=" * 80)

    torch.cuda.set_per_process_memory_fraction(MEMORY_FRACTION, 0)
    print(f"[Memory] Cap: {MEMORY_FRACTION*80:.0f}GB on GPU 0")

    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16,
        device_map={"": DEVICE}, trust_remote_code=True,
    ).eval()
    num_layers = len(model.model.layers)
    print(f"[Model] {num_layers} layers, weights={torch.cuda.memory_allocated(DEVICE)/1024**3:.2f}GB")

    all_results = []

    # ── NIAH Test ──────────────────────────────────────────────────────
    test_lengths = [4096, 8192]
    depths = [0.25, 0.50, 0.75]

    print(f"\n{'='*80}")
    print(" NIAH Retrieval Test")
    print(f"{'='*80}")

    for target in test_lengths:
        print(f"\n{'─'*70}")
        print(f" Context: {target} tokens")
        print(f"{'─'*70}")
        for depth in depths:
            needle_info = NEEDLES[int(depth * 10) % len(NEEDLES)]
            needle_text, needle_full, kw1, kw2 = needle_info

            niah = build_niah_input(tokenizer, target, needle_text, (kw1, kw2), depth)
            if not niah["has_needle"]:
                print(f"  depth={depth:.0%} SKIP (needle not in input)")
                continue

            print(f"\n  depth={depth:.0%} needle@{niah['needle_pos']:.0%} "
                  f"actual_len={niah['length']}")

            for config in CONFIGS:
                print(f"    [{config['name']:12s}] ", end="", flush=True)
                r = run_single(model, tokenizer, config,
                              niah["input_ids"], niah["attention_mask"])

                hit = False
                if not r["oom"]:
                    gen_lower = r["generated_text"].lower()
                    hit = kw1 in gen_lower or kw2 in gen_lower or needle_full in gen_lower

                result = {
                    "test": "NIAH", "target": target, "depth": depth,
                    "needle_pos": round(niah["needle_pos"], 2),
                    **{k: v for k, v in r.items() if k != "generated_text"},
                    "hit": hit,
                }
                all_results.append(result)

                if r["oom"]:
                    print("OOM")
                elif hit:
                    print(f"HIT  mem={r['peak_mem_gb']:.2f}GB t={r['time_s']:.1f}s "
                          f"gen='{r['generated_text'][:60]}'")
                else:
                    print(f"MISS mem={r['peak_mem_gb']:.2f}GB t={r['time_s']:.1f}s "
                          f"gen='{r['generated_text'][:60]}'")

    # ── Summary ────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(" RESULTS SUMMARY")
    print(f"{'='*80}")

    print(f"\n{'Config':14s} | {'4K @25%':>8s} {'4K @50%':>8s} {'4K @75%':>8s} | "
          f"{'8K @25%':>8s} {'8K @50%':>8s} {'8K @75%':>8s} | {'OOM':>3s}")
    print("─" * 95)

    for config in CONFIGS:
        row = f"{config['name']:14s} |"
        ooms = 0
        for target in test_lengths:
            for depth in depths:
                matches = [r for r in all_results
                           if r["config"] == config["name"]
                           and r["target"] == target and r["depth"] == depth]
                if matches:
                    m = matches[0]
                    if m["oom"]:
                        row += f" {'OOM':>8s}"
                        ooms += 1
                    elif m["hit"]:
                        row += f" {'HIT':>8s}"
                    else:
                        row += f" {'MISS':>8s}"
                else:
                    row += f" {'---':>8s}"
            row += " |"
        row += f" {ooms:>3d}"
        print(row)

    # ── Memory comparison ──────────────────────────────────────────────
    print(f"\n{'Config':14s} | {'4K Peak':>8s} {'8K Peak':>8s} | {'Evicted':>8s}")
    print("─" * 50)
    for config in CONFIGS:
        row = f"{config['name']:14s} |"
        for target in test_lengths:
            matches = [r for r in all_results
                       if r["config"] == config["name"]
                       and r["target"] == target and not r["oom"]]
            if matches:
                avg = np.mean([r["peak_mem_gb"] for r in matches])
                row += f" {avg:>7.2f}"
            else:
                row += f" {'OOM':>7s}"

        # Calculate eviction ratio at 4K
        if config["keep_tail"] is not None:
            max_hbm = 64 + config["keep_tail"]
            evict_pct = max(0, (4096 - max_hbm) / 4096 * 100)
            row += f" | {evict_pct:>6.0f}%"
        else:
            row += " |    0%"
        print(row)

    # ── Degradation analysis ───────────────────────────────────────────
    print(f"\n{'='*80}")
    print(" DEGRADATION ANALYSIS")
    print(f"{'='*80}")

    baseline_4k = [r for r in all_results if r["config"] == "baseline"
                   and r["target"] == 4096 and not r["oom"]]
    b_hits = sum(1 for r in baseline_4k if r["hit"])

    hk4k_4k = [r for r in all_results if r["config"] == "hk_tail4k"
               and r["target"] == 4096 and not r["oom"]]
    h4k_hits = sum(1 for r in hk4k_4k if r["hit"])

    print(f"\nControl (baseline vs hk_tail4k at 4K, should be identical):")
    print(f"  Baseline:  {b_hits}/{len(baseline_4k)} hits")
    print(f"  hk_tail4k: {h4k_hits}/{len(hk4k_4k)} hits")
    if b_hits == h4k_hits:
        print(f"  PASS: Control confirms no eviction at keep_tail=4096 with 4K input")
    else:
        print(f"  WARNING: Control shows difference!")

    print(f"\nRetrieval degradation at 4K tokens:")
    for config in CONFIGS:
        matches = [r for r in all_results if r["config"] == config["name"]
                   and r["target"] == 4096 and not r["oom"]]
        if matches:
            hits = sum(1 for r in matches if r["hit"])
            rate = hits / len(matches) * 100
            delta = hits - b_hits
            evict = ""
            if config["keep_tail"] is not None:
                evict_pct = max(0, (4096 - 64 - config["keep_tail"]) / 4096 * 100)
                evict = f"evict={evict_pct:.0f}%"
            print(f"  {config['name']:12s}: {hits}/{len(matches)} ({rate:.0f}%) "
                  f"delta={delta:+d} {evict}")

    print(f"\nRetrieval degradation at 8K tokens:")
    baseline_8k = [r for r in all_results if r["config"] == "baseline"
                   and r["target"] == 8192 and not r["oom"]]
    b8_hits = sum(1 for r in baseline_8k if r["hit"])
    for config in CONFIGS:
        matches = [r for r in all_results if r["config"] == config["name"]
                   and r["target"] == 8192 and not r["oom"]]
        if matches:
            hits = sum(1 for r in matches if r["hit"])
            rate = hits / len(matches) * 100
            delta = hits - b8_hits
            evict = ""
            if config["keep_tail"] is not None:
                evict_pct = max(0, (8192 - 64 - config["keep_tail"]) / 8192 * 100)
                evict = f"evict={evict_pct:.0f}%"
            print(f"  {config['name']:12s}: {hits}/{len(matches)} ({rate:.0f}%) "
                  f"delta={delta:+d} {evict}")

    # ── Save ───────────────────────────────────────────────────────────
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "experiments", "quantization_degradation_v3.json")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {save_path}")

    # ── Self-healing comparison ────────────────────────────────────────
    print(f"\n{'='*80}")
    print(" SELF-HEALING COMPARISON")
    print(f"{'='*80}")

    heal_configs = [c for c in CONFIGS if c.get("self_healing")]
    noheal_configs = [c for c in CONFIGS if not c.get("self_healing") and c["keep_tail"] is not None]

    print(f"\n{'Config pair':30s} | {'4K heal':>8s} {'4K no-heal':>11s} | {'8K heal':>8s} {'8K no-heal':>11s}")
    print("─" * 80)

    for hc in heal_configs:
        # Find matching no-heal config by keep_tail
        nc = None
        for n in noheal_configs:
            if n["keep_tail"] == hc["keep_tail"]:
                nc = n
                break
        if nc is None:
            continue

        label = f"tail={hc['keep_tail']}"
        row = f"{label:30s} |"
        for target in [4096, 8192]:
            h_matches = [r for r in all_results if r["config"] == hc["name"]
                         and r["target"] == target and not r["oom"]]
            n_matches = [r for r in all_results if r["config"] == nc["name"]
                         and r["target"] == target and not r["oom"]]
            h_hits = sum(1 for r in h_matches if r["hit"])
            n_hits = sum(1 for r in n_matches if r["hit"])
            row += f" {h_hits}/{len(h_matches):>3d}"
            row += f"   {n_hits}/{len(n_matches):>3d}   |"
        print(row)

    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
