#!/usr/bin/env python3
"""
quick_self_healing_comparison.py
==================================
Quick comparison: Baseline vs Self-healing ON vs Self-healing OFF

Tests only the most critical scenarios:
- NIAH: 4K/8K × 3 depths (25%, 50%, 75%) × 3 configs
- Memory: 4K/8K/16K × 3 configs
"""

import os, sys, gc, time, json
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.core.engine_wrapper import build_fused_cache

DEVICE = "cuda:0"
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "models", "Qwen2.5-7B-Instruct")
MEMORY_FRACTION = 24.0 / 80.0

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


def build_niah_input(tokenizer, target_tokens: int, needle: str, depth: float):
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

    random.seed(42)
    prefix_ids = []
    idx = 0
    while len(prefix_ids) < prefix_budget:
        sent_ids = tokenizer.encode(FILLER[idx % len(FILLER)])
        prefix_ids.extend(sent_ids)
        idx += 1
    prefix_ids = prefix_ids[:prefix_budget]

    suffix_ids = []
    while len(suffix_ids) < suffix_budget:
        sent_ids = tokenizer.encode(FILLER[idx % len(FILLER)])
        suffix_ids.extend(sent_ids)
        idx += 1
    suffix_ids = suffix_ids[:suffix_budget]

    all_ids = sys_ids + prefix_ids + needle_ids + suffix_ids + q_ids

    return {
        "input_ids": torch.tensor([all_ids], dtype=torch.long),
        "attention_mask": torch.ones(1, len(all_ids), dtype=torch.long),
        "length": len(all_ids),
    }


@torch.inference_mode()
def run_test(model, tokenizer, config_name: str, keep_tail: int, self_healing: bool,
             input_ids: torch.Tensor, attention_mask: torch.Tensor):
    input_ids = input_ids.to(DEVICE)
    attention_mask = attention_mask.to(DEVICE)
    input_len = input_ids.shape[1]
    num_layers = len(model.model.layers)

    # Build cache
    cache = build_fused_cache(
        num_layers=num_layers, sink_tokens=64,
        keep_tail=keep_tail, device=DEVICE,
        enable_quant=True, group_size=128,
        enable_prefetch=True,
        self_healing=self_healing,
    )

    torch.cuda.reset_peak_memory_stats(DEVICE)
    torch.cuda.synchronize(DEVICE)
    t0 = time.time()

    try:
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=16,
            num_beams=1,
            do_sample=False,
            use_cache=True,
            past_key_values=cache,
            pad_token_id=tokenizer.eos_token_id,
        )
        torch.cuda.synchronize(DEVICE)
        elapsed = time.time() - t0
        peak_mem = torch.cuda.max_memory_allocated(DEVICE) / 1024**3
        gen_text = tokenizer.decode(outputs[0, input_len:], skip_special_tokens=True)

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            return {"oom": True, "peak_mem_gb": 0}
        raise

    del cache
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "text": gen_text,
        "peak_mem_gb": round(peak_mem, 3),
        "time_s": round(elapsed, 3),
        "oom": False,
    }


def main():
    print("=" * 70)
    print(" Quick Self-Healing Comparison")
    print(" Baseline vs Heal ON vs Heal OFF | 24GB Memory Cap")
    print("=" * 70)

    torch.cuda.set_per_process_memory_fraction(MEMORY_FRACTION, 0)

    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16,
        device_map={"": DEVICE}, trust_remote_code=True,
    ).eval()

    results = []

    # Test configurations
    configs = [
        ("baseline", None, False),
        ("hk_healON", 1024, True),
        ("hk_healOFF", 1024, False),
    ]

    # ========================================================================
    # NIAH Test
    # ========================================================================
    print(f"\n{'='*70}")
    print(" NIAH Retrieval Test")
    print(f"{'='*70}")

    print(f"\n{'Config':12s} | {'4K@25%':>8s} {'4K@50%':>8s} {'4K@75%':>8s} | "
          f"{'8K@25%':>8s} {'8K@50%':>8s} {'8K@75%':>8s}")
    print("-" * 70)

    for config_name, keep_tail, self_healing in configs:
        row = f"{config_name:12s} |"

        for target in [4096, 8192]:
            for depth_idx, depth in enumerate([0.25, 0.50, 0.75]):
                needle_info = NEEDLES[depth_idx]
                needle_text, needle_full, kw1, kw2 = needle_info

                niah = build_niah_input(tokenizer, target, needle_text, depth)

                if config_name == "baseline":
                    result = run_test(model, tokenizer, config_name, 4096, False,
                                     niah["input_ids"], niah["attention_mask"])
                else:
                    result = run_test(model, tokenizer, config_name, keep_tail, self_healing,
                                     niah["input_ids"], niah["attention_mask"])

                if result["oom"]:
                    row += f" {'OOM':>8s}"
                else:
                    gen_lower = result["text"].lower()
                    hit = kw1 in gen_lower or kw2 in gen_lower or needle_full in gen_lower
                    row += f" {'HIT' if hit else 'MISS':>8s}"
                    results.append({
                        "config": config_name,
                        "context": target,
                        "depth": depth,
                        "hit": hit,
                        "peak_mem_gb": result["peak_mem_gb"],
                        "time_s": result["time_s"],
                    })

        print(row)

    # ========================================================================
    # Memory & Latency Test
    # ========================================================================
    print(f"\n{'='*70}")
    print(" Memory & Latency Test")
    print(f"{'='*70}")

    print(f"\n{'Context':8s} {'Config':12s} | {'Peak Mem':>10s} {'Time':>8s}")
    print("-" * 45)

    for length in [4096, 8192, 16384]:
        for config_name, keep_tail, self_healing in configs:
            prompt = "The quick brown fox jumps over the lazy dog. " * (length // 10 + 1)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                               max_length=length).to(DEVICE)

            if config_name == "baseline":
                result = run_test(model, tokenizer, config_name, 4096, False,
                                 inputs["input_ids"], inputs["attention_mask"])
            else:
                result = run_test(model, tokenizer, config_name, keep_tail, self_healing,
                                 inputs["input_ids"], inputs["attention_mask"])

            if result["oom"]:
                print(f"{length:8d} {config_name:12s} | {'OOM':>10s}")
            else:
                print(f"{length:8d} {config_name:12s} | {result['peak_mem_gb']:>10.3f} {result['time_s']:>8.3f}")

    # ========================================================================
    # Save Results
    # ========================================================================
    save_path = "experiments/quick_self_healing_comparison.json"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {save_path}")

    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
