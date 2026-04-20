#!/usr/bin/env python3
"""
eval_complex_qa.py
==================
Phase 2b: Complex Long-Text QA & Summarization Evaluation.

Loads Qwen2.5-7B-Instruct and benchmarks it on a LongBench-compatible
multi-document QA / summarization pipeline, comparing:
  - Baseline Native HF cache
  - Hetero-KV cache (kernelized)

Metrics: ROUGE-L, Token-F1, TTFT, TPOT, Peak VRAM.
"""

import os
import sys
import gc
import time
import json
import re
import torch
import warnings
from typing import Dict, List, Tuple

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

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.cache_utils import DynamicCache
from src.core.engine_wrapper import build_fused_cache, ChunkedPrefillEngine

# ROUGE
from rouge_score import rouge_scorer


# ---------------------------------------------------------------------------
# Dataset loader (with graceful fallback to synthetic LongBench-like data)
# ---------------------------------------------------------------------------

def _make_synthetic_dataset(num_samples: int = 10, target_tokens: int = 8000):
    """
    Generate high-fidelity synthetic multi-document QA samples.
    Each sample consists of ~target_tokens tokens worth of unrelated 'documents'
    followed by a question about one specific document.
    """
    topics = [
        ("Artificial Intelligence", "Alan Turing", "The Turing Test"),
        ("Space Exploration", "Neil Armstrong", "Apollo 11"),
        ("Marine Biology", "Jacques Cousteau", "The Aqualung"),
        ("Computer Science", "Tim Berners-Lee", "The World Wide Web"),
        ("Physics", "Marie Curie", "Radioactivity"),
        ("Mathematics", "Euclid", "Elements"),
        ("Medicine", "Alexander Fleming", "Penicillin"),
        ("Geography", "Roald Amundsen", "South Pole"),
        ("Literature", "William Shakespeare", "Hamlet"),
        ("Chemistry", "Dmitri Mendeleev", "Periodic Table"),
    ]

    samples = []
    for i in range(num_samples):
        docs = []
        # cycle through topics to create a long context
        for j in range(20):
            topic, person, concept = topics[(i + j) % len(topics)]
            doc = (
                f"Document {j+1}: {topic}. "
                f"In the field of {topic.lower()}, one of the most influential figures was {person}. "
                f"Their groundbreaking work on {concept} changed the discipline forever. "
                f"Historians agree that without the contribution of {person}, "
                f"modern {topic.lower()} would not have progressed as rapidly. "
                f"The legacy of {concept} continues to inspire researchers today. "
            )
            docs.append(doc)
        context = "\n".join(docs)
        target_doc_idx = i % 20
        _, target_person, target_concept = topics[(i + target_doc_idx) % len(topics)]
        question = (
            f"According to Document {target_doc_idx+1}, who made the groundbreaking work on {target_concept}?"
        )
        answer = target_person
        samples.append({
            "context": context,
            "input": question,
            "answers": [answer],
            "length": len(context.split()) + len(question.split()),  # rough word count
        })
    return samples


def load_longbench_qa(split_size: int = 10, subset_name: str = "hotpotqa"):
    """Attempt to load LongBench from HuggingFace; fallback to synthetic on failure."""
    try:
        from datasets import load_dataset
        ds = load_dataset("THUDM/LongBench", subset_name, split="test", streaming=True)
        samples = []
        for i, item in enumerate(ds):
            if i >= split_size:
                break
            samples.append({
                "context": item.get("context", ""),
                "input": item.get("input", ""),
                "answers": item.get("answers", []),
                "length": len(item.get("context", "").split()),
            })
        print(f"[Dataset] Loaded {len(samples)} samples from LongBench/{subset_name}")
        return samples, f"LongBench/{subset_name}"
    except Exception as e:
        print(f"[Dataset] LongBench download failed ({e}). Falling back to synthetic data.")
        samples = _make_synthetic_dataset(num_samples=split_size)
        return samples, "synthetic_longqa"


# ---------------------------------------------------------------------------
# Scoring utilities
# ---------------------------------------------------------------------------

def normalize_answer(s: str) -> str:
    """Lowercase, remove articles/punctuation/extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        return re.sub(r"[^\w\s]", '', text)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def token_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    common = set(pred_tokens) & set(gt_tokens)
    num_same = len(common)
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return (2 * precision * recall) / (precision + recall)


def score_qa(prediction: str, references: List[str]) -> Tuple[float, float]:
    """Return (max_token_f1, max_rouge_l)."""
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    max_f1 = 0.0
    max_rl = 0.0
    for ref in references:
        f1 = token_f1(prediction, ref)
        rl = scorer.score(ref, prediction)["rougeL"].fmeasure
        max_f1 = max(max_f1, f1)
        max_rl = max(max_rl, rl)
    return max_f1, max_rl


# ---------------------------------------------------------------------------
# Model & inference helpers
# ---------------------------------------------------------------------------

def load_qwen25(model_path: str = "models/Qwen2.5-7B-Instruct", device: str = "cuda:0"):
    """Load model. Use 4-bit if memory is tight (auto-detected)."""
    torch.cuda.empty_cache()
    gc.collect()

    # Heuristic: if free mem < 20GB, load 4-bit to be safe
    free_gb = torch.cuda.mem_get_info(device)[0] / (1024 ** 3)
    print(f"[Loader] GPU free memory: {free_gb:.2f} GB")

    if free_gb < 20:
        print("[Loader] Using 4-bit NF4 quantization to fit into available VRAM.")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            local_files_only=True,
            trust_remote_code=True,
        )
    else:
        print("[Loader] Loading in BF16.")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            local_files_only=True,
            trust_remote_code=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


class _ModelPrefillAdapter:
    def __init__(self, real_model):
        self.model = real_model
        self.config = real_model.config

    def __call__(self, input_ids, past_key_values, use_cache=True, **kwargs):
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs
            )
        return outputs


def run_generation_native(model, tokenizer, prompt_text: str, device: str, max_new_tokens: int = 40):
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats(device)

    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    seq_len = inputs.input_ids.shape[1]

    # Chunked prefill for native baseline to avoid OOM on long sequences
    chunk_size = 2048
    past_kv = DynamicCache()

    t0 = time.time()
    with torch.no_grad():
        for start in range(0, seq_len, chunk_size):
            end = min(start + chunk_size, seq_len)
            chunk_ids = inputs.input_ids[:, start:end]
            chunk_pos = torch.arange(start, end, dtype=torch.long, device=device).unsqueeze(0)
            chunk_cache_pos = torch.arange(0, end - start, dtype=torch.long, device=device)
            out = model(
                input_ids=chunk_ids,
                past_key_values=past_kv,
                use_cache=True,
                position_ids=chunk_pos,
                cache_position=chunk_cache_pos,
            )
    torch.cuda.synchronize(device)
    ttft = time.time() - t0

    # Use the full DynamicCache returned by the model
    past_kv = out.past_key_values
    current_input = inputs.input_ids[:, -1:]
    decode_times = []
    generated = [current_input]

    for step in range(max_new_tokens):
        t1 = time.time()
        pos_id = torch.tensor([[seq_len + step]], dtype=torch.long, device=device)
        cache_pos = torch.tensor([past_kv.get_seq_length()], dtype=torch.long, device=device)
        with torch.no_grad():
            out = model(
                input_ids=current_input,
                past_key_values=past_kv,
                use_cache=True,
                position_ids=pos_id,
                cache_position=cache_pos,
            )
        next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
        torch.cuda.synchronize(device)
        decode_times.append(time.time() - t1)
        past_kv = out.past_key_values
        generated.append(next_token)
        current_input = next_token

    tpot = sum(decode_times) / len(decode_times)
    peak = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    steady = torch.cuda.memory_allocated(device) / (1024 ** 3)

    full_ids = torch.cat(generated, dim=-1)
    # generated[0] is the last input token, [1:] are new tokens
    new_token_ids = full_ids[:, 1:] if full_ids.shape[-1] > 1 else full_ids
    response = tokenizer.decode(new_token_ids[0], skip_special_tokens=True)

    del past_kv
    return {
        "success": True,
        "ttft_s": ttft,
        "tpot_ms": tpot * 1000,
        "peak_mem_gb": peak,
        "steady_mem_gb": steady,
        "seq_len": seq_len,
        "response": response,
    }


def run_generation_hetero(model, tokenizer, prompt_text: str, device: str, max_new_tokens: int = 40, keep_tail: int = 8192):
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats(device)

    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    input_ids = inputs.input_ids
    seq_len = input_ids.shape[1]

    cache = build_fused_cache(
        device=device,
        sink_tokens=64,
        keep_tail=keep_tail,
        chunk_size=2048,
        group_size=128,
        enable_quant=True,
        enable_prefetch=False,
        enable_triton=False,
    )

    adapter = _ModelPrefillAdapter(model)
    engine = ChunkedPrefillEngine(model=adapter, cache=cache, chunk_size=2048)

    # chunked prefill
    t0 = time.time()
    engine.prefill(input_ids)
    torch.cuda.synchronize(device)
    ttft = time.time() - t0

    # decode
    current_input = input_ids[:, -1:]
    decode_times = []
    generated = [current_input]

    for step in range(max_new_tokens):
        t1 = time.time()
        # Explicit position_ids for correct RoPE, cache_position for mask dims
        pos_id = torch.tensor([[seq_len + step]], dtype=torch.long, device=device)
        cache_pos = torch.tensor([cache.get_seq_length()], dtype=torch.long, device=device)
        with torch.no_grad():
            out = adapter(
                input_ids=current_input,
                past_key_values=cache,
                use_cache=True,
                position_ids=pos_id,
                cache_position=cache_pos,
            )
        next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
        torch.cuda.synchronize(device)
        decode_times.append(time.time() - t1)
        generated.append(next_token)
        current_input = next_token

    tpot = sum(decode_times) / len(decode_times)
    peak = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    steady = torch.cuda.memory_allocated(device) / (1024 ** 3)

    full_ids = torch.cat(generated, dim=-1)
    # generated[0] is the last input token, [1:] are new tokens
    new_token_ids = full_ids[:, 1:] if full_ids.shape[-1] > 1 else full_ids
    response = tokenizer.decode(new_token_ids[0], skip_special_tokens=True)

    return {
        "success": True,
        "ttft_s": ttft,
        "tpot_ms": tpot * 1000,
        "peak_mem_gb": peak,
        "steady_mem_gb": steady,
        "seq_len": seq_len,
        "response": response,
    }


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def build_prompt(tokenizer, context: str, question: str) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Answer the question based ONLY on the provided documents. Be concise."},
        {"role": "user", "content": f"Documents:\n{context}\n\nQuestion: {question}"},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("=" * 80)
    print(" Phase 2b: Complex Long-Text QA Evaluation (Qwen2.5-7B-Instruct)")
    print("=" * 80)

    print("Loading dataset...")
    samples, dataset_name = load_longbench_qa(split_size=10)
    print(f"Dataset source: {dataset_name} | Samples: {len(samples)}")

    print(f"Loading model on {device}...")
    model, tokenizer = load_qwen25(device=device)

    results = {
        "dataset": dataset_name,
        "model": "Qwen2.5-7B-Instruct",
        "samples": [],
    }

    baseline_f1s, baseline_rls = [], []
    hetero_f1s, hetero_rls = [], []

    for idx, sample in enumerate(samples):
        print(f"\n--- Sample {idx+1}/{len(samples)} ---")
        prompt = build_prompt(tokenizer, sample["context"], sample["input"])
        refs = sample["answers"]

        sample_res = {"idx": idx, "question": sample["input"], "refs": refs}

        # Baseline
        print("  [Baseline] Generating...")
        try:
            baseline_out = run_generation_native(model, tokenizer, prompt, device, max_new_tokens=40)
            if baseline_out["success"]:
                pred = baseline_out["response"]
                f1, rl = score_qa(pred, refs)
                baseline_f1s.append(f1)
                baseline_rls.append(rl)
                sample_res["baseline"] = {
                    **baseline_out,
                    "f1": round(f1, 4),
                    "rouge_l": round(rl, 4),
                }
                print(f"    -> F1={f1:.3f} ROUGE-L={rl:.3f} "
                      f"TTFT={baseline_out['ttft_s']:.3f}s "
                      f"TPOT={baseline_out['tpot_ms']:.2f}ms "
                      f"Peak={baseline_out['peak_mem_gb']:.2f}GB")
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                sample_res["baseline"] = {"error": "OOM"}
                print("    -> OOM (native)")
                torch.cuda.empty_cache()
                gc.collect()
            else:
                raise

        torch.cuda.empty_cache()
        gc.collect()

        # Hetero-KV
        print("  [Hetero-KV] Generating...")
        try:
            hetero_out = run_generation_hetero(model, tokenizer, prompt, device, max_new_tokens=40, keep_tail=8192)
            if hetero_out["success"]:
                pred = hetero_out["response"]
                f1, rl = score_qa(pred, refs)
                hetero_f1s.append(f1)
                hetero_rls.append(rl)
                sample_res["hetero"] = {
                    **hetero_out,
                    "f1": round(f1, 4),
                    "rouge_l": round(rl, 4),
                }
                print(f"    -> F1={f1:.3f} ROUGE-L={rl:.3f} "
                      f"TTFT={hetero_out['ttft_s']:.3f}s "
                      f"TPOT={hetero_out['tpot_ms']:.2f}ms "
                      f"Peak={hetero_out['peak_mem_gb']:.2f}GB")
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                sample_res["hetero"] = {"error": "OOM"}
                print("    -> OOM (hetero)")
                torch.cuda.empty_cache()
                gc.collect()
            else:
                raise

        results["samples"].append(sample_res)
        torch.cuda.empty_cache()
        gc.collect()

    # Aggregate
    def avg(lst):
        return sum(lst) / len(lst) if lst else 0.0

    summary = {
        "baseline": {
            "avg_f1": round(avg(baseline_f1s), 4),
            "avg_rouge_l": round(avg(baseline_rls), 4),
        },
        "hetero": {
            "avg_f1": round(avg(hetero_f1s), 4),
            "avg_rouge_l": round(avg(hetero_rls), 4),
        },
    }
    results["summary"] = summary

    print("\n" + "=" * 80)
    print(" Aggregate Results")
    print("=" * 80)
    print(f"  Baseline  | F1={summary['baseline']['avg_f1']:.4f}  ROUGE-L={summary['baseline']['avg_rouge_l']:.4f}")
    print(f"  Hetero-KV | F1={summary['hetero']['avg_f1']:.4f}  ROUGE-L={summary['hetero']['avg_rouge_l']:.4f}")
    print(f"  Gap       | ΔF1={summary['hetero']['avg_f1']-summary['baseline']['avg_f1']:+.4f} "
          f"ΔROUGE-L={summary['hetero']['avg_rouge_l']-summary['baseline']['avg_rouge_l']:+.4f}")

    os.makedirs("experiments", exist_ok=True)
    out_path = "experiments/eval_complex_qa_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[Saved] {out_path}")


if __name__ == "__main__":
    main()
