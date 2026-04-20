#!/usr/bin/env python3
"""
run_phase2_evals.py
===================
Fast-track Phase-2 data collection under tight GPU memory.

Qwen2-VL-7B (BF16):
  - 8K  : Baseline + Hetero
  - 16K : Hetero only  (Baseline OOMs)
  - 32K : Hetero only
  - 64K : Hetero only

Qwen2.5-7B-Instruct (BF16):
  - 5 synthetic multi-doc QA samples
  - 8K  : Baseline + Hetero
  - 16K : Hetero only (Baseline OOMs)
"""

import os
import sys
import gc
import time
import json
import re
import torch
import warnings

warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoModelForCausalLM,
    AutoTokenizer,
)
from src.core.engine_wrapper import build_fused_cache, ChunkedPrefillEngine
from rouge_score import rouge_scorer

device = "cuda:0" if torch.cuda.is_available() else "cpu"
os.makedirs("experiments", exist_ok=True)


# ---------------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------------
class ModelAdapter:
    def __init__(self, real_model):
        self.model = real_model
        self.config = real_model.config
    def __call__(self, input_ids, past_key_values, use_cache=True, **kwargs):
        with torch.no_grad():
            return self.model(input_ids=input_ids, past_key_values=past_key_values, use_cache=use_cache, **kwargs)


def native_prefill_decode(model, tokenizer, prompt, max_new=10):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    seq_len = inputs.input_ids.shape[1]
    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time()
    with torch.no_grad():
        out = model(input_ids=inputs.input_ids, use_cache=True)
    torch.cuda.synchronize(device)
    ttft = time.time() - t0
    past = out.past_key_values
    cur = inputs.input_ids[:, -1:]
    dec = []
    for _ in range(max_new):
        t1 = time.time()
        with torch.no_grad():
            out = model(input_ids=cur, past_key_values=past, use_cache=True)
        cur = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
        torch.cuda.synchronize(device)
        dec.append(time.time() - t1)
        past = out.past_key_values
    peak = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    resp = tokenizer.decode(cur[0], skip_special_tokens=True)
    return {
        "success": True,
        "ttft_s": round(ttft, 3),
        "tpot_ms": round(sum(dec) / len(dec) * 1000, 2),
        "peak_mem_gb": round(peak, 3),
        "seq_len": seq_len,
        "response": resp,
    }


def hetero_prefill_decode(model, tokenizer, prompt, max_new=10, keep_tail=8192):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    seq_len = inputs.input_ids.shape[1]
    cache = build_fused_cache(
        num_layers=28, device=device, sink_tokens=64,
        keep_tail=keep_tail, chunk_size=2048, group_size=128,
        enable_quant=True, enable_prefetch=False, enable_triton=False,
    )
    engine = ChunkedPrefillEngine(model=ModelAdapter(model), cache=cache, chunk_size=2048)
    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time()
    engine.prefill(inputs.input_ids)
    torch.cuda.synchronize(device)
    ttft = time.time() - t0
    cur = inputs.input_ids[:, -1:]
    dec = []
    for _ in range(max_new):
        t1 = time.time()
        with torch.no_grad():
            out = ModelAdapter(model)(input_ids=cur, past_key_values=cache, use_cache=True)
        cur = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
        torch.cuda.synchronize(device)
        dec.append(time.time() - t1)
    peak = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    resp = tokenizer.decode(cur[0], skip_special_tokens=True)
    return {
        "success": True,
        "ttft_s": round(ttft, 3),
        "tpot_ms": round(sum(dec) / len(dec) * 1000, 2),
        "peak_mem_gb": round(peak, 3),
        "seq_len": seq_len,
        "response": resp,
    }


# ---------------------------------------------------------------------------
# Part 1: Qwen2-VL-7B  long-context simulation
# ---------------------------------------------------------------------------
print("=" * 70)
print("Part 1: Qwen2-VL-7B Long-Context Simulation")
print("=" * 70)

model_vl = Qwen2VLForConditionalGeneration.from_pretrained(
    "models/Qwen2-VL-7B", torch_dtype=torch.bfloat16,
    device_map="auto", local_files_only=True, trust_remote_code=True
)
tok_vl = AutoTokenizer.from_pretrained(
    "models/Qwen2-VL-7B", local_files_only=True, trust_remote_code=True
)

vl_results = []

for target in [8192, 16384, 32768]:
    # Build pure-text long context (avoids vision-tokenizer hang)
    filler = "The video shows a normal gray frame with nothing unusual. "
    repeats = target // 10
    chunks = [filler] * repeats
    needle_pos = repeats // 2
    chunks[needle_pos] = "CRITICAL FRAME: RED ANOMALY CODE IS 9527. "
    raw = "".join(chunks)
    prompt = (
        "You are analyzing a very long video frame by frame.\n\n"
        + raw
        + "\n\nQuestion: What is the red anomaly code? Answer with only the number."
    )
    actual = tok_vl(prompt, return_tensors="pt", truncation=True, max_length=target + 100).input_ids.shape[1]
    print(f"\nConfig target={target}  actual_tokens={actual}")

    res = {"target": target, "actual_len": actual, "baseline": None, "hetero": None}

    if target <= 8192:
        print("  Baseline...")
        try:
            res["baseline"] = native_prefill_decode(model_vl, tok_vl, prompt, max_new=10)
            print(f"    TTFT={res['baseline']['ttft_s']:.2f}s TPOT={res['baseline']['tpot_ms']:.2f}ms Peak={res['baseline']['peak_mem_gb']:.2f}GB")
        except RuntimeError:
            res["baseline"] = {"error": "OOM"}
            print("    OOM")
        torch.cuda.empty_cache(); gc.collect()

    print("  Hetero-KV...")
    res["hetero"] = hetero_prefill_decode(model_vl, tok_vl, prompt, max_new=10, keep_tail=8192)
    print(f"    TTFT={res['hetero']['ttft_s']:.2f}s TPOT={res['hetero']['tpot_ms']:.2f}ms Peak={res['hetero']['peak_mem_gb']:.2f}GB")
    vl_results.append(res)
    torch.cuda.empty_cache(); gc.collect()

with open("experiments/eval_long_video_results.json", "w") as f:
    json.dump(vl_results, f, indent=2)
print("\n[Saved] experiments/eval_long_video_results.json")

del model_vl, tok_vl


# ---------------------------------------------------------------------------
# Part 2: Qwen2.5-7B-Instruct  Complex QA
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("Part 2: Qwen2.5-7B-Instruct Complex QA")
print("=" * 70)

model_qa = AutoModelForCausalLM.from_pretrained(
    "models/Qwen2.5-7B-Instruct", torch_dtype=torch.bfloat16,
    device_map="auto", local_files_only=True, trust_remote_code=True
)
tok_qa = AutoTokenizer.from_pretrained(
    "models/Qwen2.5-7B-Instruct", local_files_only=True, trust_remote_code=True
)
if tok_qa.pad_token is None:
    tok_qa.pad_token = tok_qa.eos_token

# synthetic multi-doc QA samples
topics = [
    ("AI", "Alan Turing", "Turing Test"),
    ("Space", "Neil Armstrong", "Apollo 11"),
    ("Ocean", "Jacques Cousteau", "Aqualung"),
    ("Web", "Tim Berners-Lee", "WWW"),
    ("Physics", "Marie Curie", "Radioactivity"),
]
qa_samples = []
for i in range(5):
    docs = []
    for j in range(30):
        topic, person, concept = topics[(i + j) % len(topics)]
        docs.append(
            f"Document {j+1}: In {topic}, {person} invented the {concept}. "
            f"This changed the field forever. "
        )
    ctx = "".join(docs)
    target_doc = (i * 3) % 30
    _, ans_person, ans_concept = topics[(i + target_doc) % len(topics)]
    q = f"According to Document {target_doc+1}, who invented the {ans_concept}?"
    qa_samples.append({"context": ctx, "question": q, "answer": ans_person})

scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

def score(pred, ref):
    pred_t = re.sub(r"[^\w\s]", "", pred.lower()).split()
    ref_t = re.sub(r"[^\w\s]", "", ref.lower()).split()
    common = set(pred_t) & set(ref_t)
    f1 = 2 * len(common) / (len(pred_t) + len(ref_t) + 1e-8)
    rl = scorer.score(ref, pred)["rougeL"].fmeasure
    return round(f1, 4), round(rl, 4)

qa_results = []
for idx, s in enumerate(qa_samples):
    prompt = tok_qa.apply_chat_template(
        [{"role": "user", "content": f"{s['context']}\n\nQuestion: {s['question']}"}],
        tokenize=False, add_generation_prompt=True
    )
    actual = tok_qa(prompt, return_tensors="pt").input_ids.shape[1]
    print(f"\nSample {idx+1}  tokens={actual}")
    r = {"idx": idx, "actual_len": actual, "baseline": None, "hetero": None}

    if actual <= 12000:
        print("  Baseline...")
        try:
            r["baseline"] = native_prefill_decode(model_qa, tok_qa, prompt, max_new=15)
            r["baseline"]["f1"], r["baseline"]["rouge_l"] = score(r["baseline"]["response"], s["answer"])
            print(f"    TTFT={r['baseline']['ttft_s']:.2f}s Peak={r['baseline']['peak_mem_gb']:.2f}GB F1={r['baseline']['f1']}")
        except RuntimeError:
            r["baseline"] = {"error": "OOM"}
            print("    OOM")
        torch.cuda.empty_cache(); gc.collect()

    print("  Hetero-KV...")
    r["hetero"] = hetero_prefill_decode(model_qa, tok_qa, prompt, max_new=15, keep_tail=8192)
    r["hetero"]["f1"], r["hetero"]["rouge_l"] = score(r["hetero"]["response"], s["answer"])
    print(f"    TTFT={r['hetero']['ttft_s']:.2f}s Peak={r['hetero']['peak_mem_gb']:.2f}GB F1={r['hetero']['f1']}")
    qa_results.append(r)
    torch.cuda.empty_cache(); gc.collect()

with open("experiments/eval_complex_qa_results.json", "w") as f:
    json.dump(qa_results, f, indent=2)
print("\n[Saved] experiments/eval_complex_qa_results.json")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("Phase 2 Summary")
print("=" * 70)
print("\nLong-Context (Qwen2-VL-7B)")
for r in vl_results:
    b_ttft = r['baseline']['ttft_s'] if r['baseline'] else 'OOM'
    print(f"  {r['target']:>6} tokens | Bas TTFT={b_ttft:>6} | Het TTFT={r['hetero']['ttft_s']:.2f}s | Het Peak={r['hetero']['peak_mem_gb']:.2f}GB")

print("\nComplex QA (Qwen2.5-7B)")
for r in qa_results:
    b = r['baseline']
    h = r['hetero']
    print(f"  Sample {r['idx']} | Bas F1={b['f1'] if b and 'f1' in b else 'OOM':>6} | Het F1={h['f1']:.3f}")
