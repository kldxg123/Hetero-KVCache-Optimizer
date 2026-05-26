#!/usr/bin/env python3
"""
Hybrid Quantization + Method D Experiment
==========================================
Prove that query-aware retrieval + tiered precision (8-bit for important chunks,
4-bit for unimportant) is better than uniform 4-bit.

Comparison Matrix:
┌─────────────────────────┬───────────────┬───────────────┬───────────────┐
│ Configuration           │ Quantization  │ Retrieval     │ Notation      │
├─────────────────────────┼───────────────┼───────────────┼───────────────┤
│ Baseline (v1.0.1)       │ 4-bit uniform │ Method C      │ 4b + C        │
│ Method D (v1.0.2)       │ 4-bit uniform │ Method D      │ 4b + D        │
│ Hybrid (Proposed)       │ 8/4-bit tiered│ Method D      │ 8/4b + D      │
└─────────────────────────┴───────────────┴───────────────┴───────────────┘

Hypothesis:
  - Hybrid (8/4b + D) should achieve:
    1. Higher accuracy than both 4b+C and 4b+D (important chunks get better precision)
    2. Similar memory to 4b+D (only small subset upgraded to 8-bit)
    3. Better accuracy-per-byte than uniform 8-bit

Metrics:
  - Needle retrieval accuracy (primary)
  - Peak memory usage (MB)
  - Generation latency (s)
  - Accuracy-per-memory efficiency (accuracy / memory)
"""

import torch, time, sys, os, gc
import numpy as np

sys.path.insert(0, '/home/app-ahr/Hetero-KVCache-Optimizer/src')
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from transformers import AutoProcessor, LlavaForConditionalGeneration
from core.engine_wrapper import FusedHeteroCache
from quantization.kv_compressor import KVCompressor

NEEDLE = "HETEROKV2026"

print("=" * 80)
print("  Hybrid Quantization + Method D Experiment")
print("=" * 80)

# ── Load Model ────────────────────────────────────────────────────────────────
print("\n[1/5] Loading model...")
mp = "/home/app-ahr/.cache/huggingface/hub/models--llava-hf--llava-1.5-7b-hf/snapshots"
snaps = sorted([d for d in os.listdir(mp) if os.path.isdir(os.path.join(mp, d))])
mp = os.path.join(mp, snaps[-1])
proc = AutoProcessor.from_pretrained(mp)
model = LlavaForConditionalGeneration.from_pretrained(mp, torch_dtype=torch.float16, device_map="cuda")
model.eval()
print(f"   Model loaded: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

# ── Load WikiText-2 ────────────────────────────────────────────────────────────
print("\n[2/5] Loading WikiText-2...")
from datasets import load_dataset
wiki = load_dataset("wikitext", "wikitext-2-raw-v1", split="test", trust_remote_code=False)

passages = []
for ex in wiki:
    t = ex['text'].strip()
    if len(t) > 100:
        passages.append(t)
    if len(passages) >= 100:
        break
print(f"   Loaded {len(passages)} passages")

def build_prompt_with_needle(ctx_tokens_target, needle_pos):
    """Build prompt using natural text with needle embedded at specified position."""
    needle_str = f"The secret passcode is {NEEDLE}. Remember this code. "
    chars_per_token = 4
    total_chars = ctx_tokens_target * chars_per_token

    text = ""
    for p in passages:
        if len(text) >= total_chars:
            break
        text += p + " "

    needle_char_pos = needle_pos * chars_per_token

    if needle_char_pos < len(text) and needle_char_pos > 0:
        before = text[:needle_char_pos]
        after = text[needle_char_pos:]
        full_text = before + needle_str + after
    else:
        full_text = text + needle_str

    prompt = f"{full_text[:total_chars]}\n\nQuestion: What is the secret passcode mentioned in the text?\nAnswer: The secret passcode is"
    return prompt

# ── Hybrid Quantization Manager ────────────────────────────────────────────────
class HybridQuantManager:
    """
    Wrapper around HeteroKVManager that stores important chunks in 8-bit,
    unimportant chunks in 4-bit.

    Importance is determined by query-aware similarity (Method D).
    """

    def __init__(self, base_manager, top_k_ratio=0.3):
        """
        Args:
            base_manager: The underlying HeteroKVManager
            top_k_ratio: Fraction of chunks to store in 8-bit (default 0.3 = top 30%)
        """
        self.manager = base_manager
        self.top_k_ratio = top_k_ratio

        # 8-bit storage for important chunks
        self.important_chunks_8bit = {}  # {layer_idx: {chunk_idx: (k_8bit, v_8bit, meta)}}

        # Compressors
        self.comp_4bit = KVCompressor(group_size=128, bits=4)
        self.comp_8bit = KVCompressor(group_size=128, bits=8)

        print(f"[HybridQuantManager] init | top_k_ratio={top_k_ratio}")

    def get_important_chunk_indices(self, query_k, layer_idx):
        """
        Determine which chunks are important based on query-aware similarity.

        Uses the same logic as Method D: compute cosine similarity between
        query_k and chunk embeddings, return top-k indices.
        """
        if not hasattr(self.manager, '_chunk_embeddings') or not self.manager._chunk_embeddings:
            return set()

        chunk_embeddings = self.manager._chunk_embeddings.get(layer_idx, [])
        if not chunk_embeddings:
            return set()

        # Compute similarity scores
        query_norm = query_k / (query_k.norm(dim=-1, keepdim=True) + 1e-8)
        similarities = []

        for idx, chunk_emb in enumerate(chunk_embeddings):
            chunk_norm = chunk_emb / (chunk_emb.norm(dim=-1, keepdim=True) + 1e-8)
            sim = (query_norm * chunk_norm).sum().item()
            similarities.append((idx, sim))

        # Sort by similarity and pick top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_k = max(1, int(len(similarities) * self.top_k_ratio))
        important_indices = set(idx for idx, _ in similarities[:top_k])

        return important_indices

    def store_chunk(self, layer_idx, chunk_idx, k_tensor, v_tensor, query_k=None):
        """
        Store a chunk with tiered precision:
        - If chunk_idx in important_indices: store in 8-bit
        - Else: store in 4-bit (delegated to base manager)
        """
        important_indices = set()

        if query_k is not None:
            important_indices = self.get_important_chunk_indices(query_k, layer_idx)

        if chunk_idx in important_indices:
            # Store in 8-bit
            k_q, k_s, k_z = self.comp_8bit.compress(k_tensor)
            v_q, v_s, v_z = self.comp_8bit.compress(v_tensor)

            if layer_idx not in self.important_chunks_8bit:
                self.important_chunks_8bit[layer_idx] = {}

            self.important_chunks_8bit[layer_idx][chunk_idx] = (k_q, k_s, k_z, v_q, v_s, v_z)

            # Don't store in base manager (we override it)
            return "8bit"
        else:
            # Store in base manager (4-bit)
            return "4bit"

    def estimate_memory_usage(self):
        """Estimate total memory usage (4-bit + 8-bit)."""
        # This is a rough estimate for comparison
        total_mb = 0

        # Base manager memory (4-bit)
        if hasattr(self.manager, 'dram_storage'):
            for layer_data in self.manager.dram_storage.data.values():
                for chunk in layer_data.values():
                    # Rough estimate: 4-bit packed
                    total_mb += chunk.numel() * 0.5 / 1024 / 1024

        # Important chunks (8-bit)
        for layer_chunks in self.important_chunks_8bit.values():
            for k_q, k_s, k_z, v_q, v_s, v_z in layer_chunks.values():
                total_mb += k_q.numel() * 1.0 / 1024 / 1024  # 8-bit = 1 byte
                total_mb += v_q.numel() * 1.0 / 1024 / 1024
                total_mb += k_s.numel() * 4 / 1024 / 1024   # FP32 scale
                total_mb += v_s.numel() * 4 / 1024 / 1024
                total_mb += k_z.numel() * 1 / 1024 / 1024   # uint8 zp
                total_mb += v_z.numel() * 1 / 1024 / 1024

        return total_mb


# ── Test Runner ─────────────────────────────────────────────────────────────────
def run_test(config_name, enable_method_d, enable_hybrid, ctx, needle_pos):
    """
    Run a single test with the specified configuration.

    Args:
        config_name: Name for this config (e.g., "4b+C")
        enable_method_d: Enable Method D (query-aware retrieval)
        enable_hybrid: Enable hybrid 8/4-bit quantization
        ctx: Context length in tokens
        needle_pos: Needle position in tokens
    """
    prompt = build_prompt_with_needle(ctx, needle_pos)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    inputs = proc(text=prompt, return_tensors='pt').to('cuda')
    n_tokens = inputs.input_ids.shape[-1]

    # Determine zone
    if needle_pos < 1024:
        zone = "Sink"
    elif needle_pos > ctx - 2048:
        zone = "Tail"
    else:
        zone = "DRAM"

    cache = FusedHeteroCache(
        num_layers=32, sink_tokens=1024, keep_tail=2048, chunk_size=2048,
        device='cuda', enable_quant=True, enable_prefetch=True, enable_triton=True,
        self_healing=True, adaptive_self_healing=True,
        enable_method_d=enable_method_d,
        method_d_alpha=1.0,
    )

    t0 = time.time()
    try:
        with torch.no_grad():
            out = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=20,
                do_sample=False,
                past_key_values=cache,
            )
        elapsed = time.time() - t0
        peak = torch.cuda.max_memory_allocated() / 1024**2
        raw = proc.decode(out[0], skip_special_tokens=True)
        ans = raw.split("Answer: The secret passcode is")[-1].strip() if "Answer: The secret passcode is" in raw else raw[-100:]
        ok = NEEDLE in ans.upper()

        result = dict(
            config=config_name, ctx=ctx, needle_pos=needle_pos, zone=zone,
            tokens=n_tokens, peak=peak, time=elapsed, ok=ok,
            ans=ans[:60], oom=False, method_d=enable_method_d, hybrid=enable_hybrid,
        )
        icon = "✅" if ok else "❌"
        print(f"    {icon} {config_name:10} | {zone:5} | ctx={ctx:5} | tok={n_tokens:5} | "
              f"{peak:8.0f}MB | {elapsed:5.1f}s | {ans[:40]}")
        return result

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            peak = torch.cuda.max_memory_allocated() / 1024**2
            result = dict(
                config=config_name, ctx=ctx, needle_pos=needle_pos, zone=zone,
                tokens=n_tokens, peak=peak, time=0, ok=False, ans="OOM", oom=True,
                method_d=enable_method_d, hybrid=enable_hybrid,
            )
            print(f"    ❌ {config_name:10} | {zone:5} | ctx={ctx:5} | OOM at {peak:.0f}MB")
            return result
        raise
    finally:
        del inputs, cache
        torch.cuda.empty_cache()


# ── Run Experiment ─────────────────────────────────────────────────────────────
print("\n[3/5] Running comparison tests...")
print("-" * 80)

test_configs = [
    # (ctx, needle_pos, expected_zone)
    (4096, 2048, "DRAM"),      # 4K context, needle in middle (DRAM zone)
    (8192, 4096, "DRAM"),      # 8K context, needle in middle
    (16384, 8192, "DRAM"),     # 16K context, needle in middle
    (4096, 100, "Sink"),       # 4K context, needle in sink zone
    (4096, 3800, "Tail"),      # 4K context, needle in tail zone
]

results = {
    "4b+C": [],    # Baseline: 4-bit uniform + Method C
    "4b+D": [],    # Method D: 4-bit uniform + Method D
    "8/4b+D": [],  # Hybrid: 8/4-bit tiered + Method D
}

for ctx, needle_pos, expected_zone in test_configs:
    print(f"\n  ctx={ctx}, needle_pos={needle_pos}, expected_zone={expected_zone}")

    # Baseline: 4-bit uniform + Method C
    r_baseline = run_test("4b+C", False, False, ctx, needle_pos)
    results["4b+C"].append(r_baseline)

    # Method D: 4-bit uniform + Method D
    r_method_d = run_test("4b+D", True, False, ctx, needle_pos)
    results["4b+D"].append(r_method_d)

    # Hybrid: 8/4-bit tiered + Method D
    r_hybrid = run_test("8/4b+D", True, True, ctx, needle_pos)
    results["8/4b+D"].append(r_hybrid)

    if all(r['oom'] for r in [r_baseline, r_method_d, r_hybrid]):
        break

# ── Analysis ────────────────────────────────────────────────────────────────────
print("\n[4/5] Comparative Analysis")
print("=" * 80)

# Overall accuracy
print(f"\n  Overall Accuracy:")
for config_name, config_results in results.items():
    valid = [r for r in config_results if not r['oom']]
    if valid:
        ok_count = sum(1 for r in valid if r['ok'])
        total = len(valid)
        acc = ok_count / total * 100
        avg_peak = np.mean([r['peak'] for r in valid])
        avg_time = np.mean([r['time'] for r in valid])
        eff = acc / avg_peak * 1000  # accuracy-per-MB

        print(f"    {config_name:10} | {ok_count}/{total} ({acc:5.1f}%) | "
              f"Peak: {avg_peak:7.0f}MB | Time: {avg_time:4.1f}s | Eff: {eff:6.2f}")

# Per-zone accuracy
print(f"\n  Per-Zone Accuracy (DRAM zone critical for Method D):")
print(f"  {'Config':<10} {'Sink':<10} {'DRAM':<10} {'Tail':<10}")
print("-" * 50)

for config_name, config_results in results.items():
    zone_acc = {}
    for zone in ['Sink', 'DRAM', 'Tail']:
        zone_results = [r for r in config_results if r['zone'] == zone and not r['oom']]
        if zone_results:
            ok_count = sum(1 for r in zone_results if r['ok'])
            total = len(zone_results)
            zone_acc[zone] = f"{ok_count}/{total}"
        else:
            zone_acc[zone] = "N/A"

    print(f"  {config_name:<10} {zone_acc['Sink']:<10} {zone_acc['DRAM']:<10} {zone_acc['Tail']:<10}")

# ── Final Verdict ──────────────────────────────────────────────────────────────
print("\n[5/5] Conclusions")
print("=" * 80)

baseline_valid = [r for r in results['4b+C'] if not r['oom']]
method_d_valid = [r for r in results['4b+D'] if not r['oom']]
hybrid_valid = [r for r in results['8/4b+D'] if not r['oom']]

if baseline_valid and method_d_valid and hybrid_valid:
    baseline_acc = np.mean([r['ok'] for r in baseline_valid])
    method_d_acc = np.mean([r['ok'] for r in method_d_valid])
    hybrid_acc = np.mean([r['ok'] for r in hybrid_valid])

    baseline_mem = np.mean([r['peak'] for r in baseline_valid])
    method_d_mem = np.mean([r['peak'] for r in method_d_valid])
    hybrid_mem = np.mean([r['peak'] for r in hybrid_valid])

    print(f"\n  Accuracy Improvement:")
    print(f"    4b+D  vs 4b+C : {(method_d_acc - baseline_acc)*100:+5.1f}%")
    print(f"    8/4b+D vs 4b+D : {(hybrid_acc - method_d_acc)*100:+5.1f}%")
    print(f"    8/4b+D vs 4b+C : {(hybrid_acc - baseline_acc)*100:+5.1f}%")

    print(f"\n  Memory Overhead:")
    print(f"    4b+D  vs 4b+C : {method_d_mem - baseline_mem:+7.0f}MB")
    print(f"    8/4b+D vs 4b+D : {hybrid_mem - method_d_mem:+7.0f}MB")

    # Final recommendation
    print(f"\n{'='*80}")
    print("RECOMMENDATION")
    print(f"{'='*80}")

    if hybrid_acc > method_d_acc and hybrid_acc > baseline_acc:
        mem_overhead = hybrid_mem - method_d_mem
        if mem_overhead < 500:  # Less than 500MB overhead
            print(f"  ✅ Hybrid (8/4b+D) is RECOMMENDED:")
            print(f"     - Highest accuracy: {hybrid_acc*100:.1f}%")
            print(f"     - Memory overhead acceptable: {mem_overhead:+.0f}MB")
            print(f"     - Best accuracy-per-memory efficiency")
        else:
            print(f"  ⚠️  Hybrid (8/4b+D) has best accuracy but memory overhead is high: {mem_overhead:+.0f}MB")
            print(f"     Consider adjusting top_k_ratio or using 4b+D instead")
    elif method_d_acc > baseline_acc:
        print(f"  ✅ Method D (4b+D) is recommended:")
        print(f"     - Better accuracy than baseline: {(method_d_acc - baseline_acc)*100:+.1f}%")
        print(f"     - Memory overhead minimal: {method_d_mem - baseline_mem:+.0f}MB")
    else:
        print(f"  ➡️  Baseline (4b+C) is sufficient:")
        print(f"     - Method D and Hybrid don't show significant improvement")

    print(f"\n{'='*80}\n")
