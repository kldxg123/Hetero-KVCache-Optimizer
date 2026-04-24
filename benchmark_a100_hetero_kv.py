#!/usr/bin/env python3
"""
A100-Enhanced Hetero-KV Benchmark with Power Monitoring
Modified benchmark with 24GB memory constraint and GPU power profiling
"""

import os
import sys
import torch
import time
import json
from pathlib import Path

# Add source directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from scripts.a100_emulator import A100Emulator

# Patch to bypass Transformers version checks (keep original)
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

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

# Import core components
from src.core.engine_wrapper import build_fused_cache, ChunkedPrefillEngine

# Dummy model simulating Qwen2-VL for generating real KV tensor pressure
class DummyQwen2VL:
    def __init__(self, hidden_size=128, num_heads=32, num_layers=1):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers

    def __call__(self, input_ids, past_key_values, use_cache=True, position_ids=None, attention_mask=None, cache_position=None):
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]

        # Simulate Transformer forward pass per layer, generating real FP16 tensors into Cache
        for layer_idx in range(self.num_layers):
            # Typical Qwen2-VL KV dimensions: [Batch, Heads, SeqLen, HeadDim]
            k = torch.randn(batch_size, self.num_heads, seq_len, self.hidden_size, dtype=torch.float16, device=input_ids.device)
            v = torch.randn(batch_size, self.num_heads, seq_len, self.hidden_size, dtype=torch.float16, device=input_ids.device)

            # Core: trigger FusedHeteroCache update logic (includes chunked interception and DRAM eviction)
            past_key_values.update(k, v, layer_idx=layer_idx)

        return None


def run_a100_benchmark(token_lengths=[45025, 65536, 128000], enable_power_monitoring=True):
    """
    Run Hetero-KV benchmark with A100 24GB memory constraint
    """
    print("\n" + "=" * 70)
    print("🚀 A100-Enhanced Hetero-KV Benchmark with Power Profiling")
    print("=" * 70)

    # Setup A100 emulator
    emulator = A100Emulator(memory_fraction=24.0/80.0)
    emulator.reset_peak_memory()

    # Create log directory
    Path("logs").mkdir(parents=True, exist_ok=True)

    results = []
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    if device == "cpu":
        print("⚠️ No GPU detected, running CPU emulation")
        enable_power_monitoring = False
    else:
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated(device) / (1024 ** 3)
        print(f"🔧 Initial memory: {initial_memory:.2f}GB")

    # Process each token length
    for target_seq_len in token_lengths:
        print(f"\n🎯 Testing with {target_seq_len:,} tokens...")

        if enable_power_monitoring:
            emulator.log_power_metrics("before")
            emulator.log_memory_metrics(f"before_{target_seq_len}")

        start_time = time.time()
        try:
            # Build cache with current configuration
            cache = build_fused_cache(
                device=device,
                sink_tokens=64,
                keep_tail=8192,
                chunk_size=2048,
                group_size=128,
                enable_quant=True,
                enable_prefetch=True,
                enable_triton=False
            )

            # Setup engine and model
            model = DummyQwen2VL(num_layers=1)
            engine = ChunkedPrefillEngine(model=model, cache=cache, chunk_size=2048)

            # Create input tensor
            input_ids = torch.randint(0, 10000, (1, target_seq_len), device=device)

            print(f"📊 Starting prefill with {target_seq_len:,} tokens...")

            # Run benchmark
            engine.prefill(input_ids)
            if device != "cpu":
                torch.cuda.synchronize()

            end_time = time.time()

            # Calculate metrics
            if device != "cpu":
                peak_mem_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
                current_mem_gb = torch.cuda.memory_allocated(device) / (1024 ** 3)

                # Check if we violated the 24GB constraint
                memory_violated = peak_mem_gb > 24.0
                efficiency = min(1.0, 24.0 / peak_mem_gb) if peak_mem_gb > 0 else 1.0
            else:
                peak_mem_gb, current_mem_gb = 0.0, 0.0
                memory_violated = False
                efficiency = 1.0

            # Log final metrics
            if enable_power_monitoring:
                emulator.log_power_metrics("after")
                emulator.log_memory_metrics(f"after_{target_seq_len}")

            # Store results
            result = {
                "tokens": target_seq_len,
                "ttft": end_time - start_time,
                "peak_memory_gb": peak_mem_gb,
                "steady_memory_gb": current_mem_gb,
                "oom_violation": memory_violated,
                "memory_efficiency": efficiency,
                "dram_chunks": len(cache.dram_table),
                "seq_length": cache.get_seq_length()
            }

            results.append(result)

            # Display results
            status = "⚠️ OOM VIOLATION" if memory_violated else "✅ SUCCESS"
            print(f"\n📈 Results for {target_seq_len:,} tokens:")
            print(f"   Status: {status}")
            print(f"   TTFT: {result['ttft']:.3f} s")
            print(f"   Peak Memory: {result['peak_memory_gb']:.2f}GB (24GB limit)")
            print(f"   Efficiency: {result['memory_efficiency']:.1%}")
            print(f"   DRAM Chunks: {result['dram_chunks']}")

        except RuntimeError as e:
            if "OutOfMemory" in str(e) or "CUDA out of memory" in str(e):
                print(f"   ❌ OOM at {target_seq_len:,} tokens")
                results.append({
                    "tokens": target_seq_len,
                    "oom": True,
                    "error": str(e)
                })
            else:
                print(f"   ❌ Unknown error: {str(e)}")
                results.append({
                    "tokens": target_seq_len,
                    "error": str(e)
                })

        # Cleanup
        if device != "cpu":
            torch.cuda.empty_cache()

    # Generate summary
    print("\n" + "=" * 70)
    print("📊 BENCHMARK SUMMARY (A100 24GB Constraint)")
    print("=" * 70)

    successful_runs = [r for r in results if "oom" not in r and "error" not in r]
    max_tokens_success = max([r["tokens"] for r in successful_runs]) if successful_runs else 0

    if successful_runs:
        avg_ttft = sum(r["ttft"] for r in successful_runs) / len(successful_runs)
        avg_peak_mem = sum(r["peak_memory_gb"] for r in successful_runs) / len(successful_runs)
        avg_efficiency = sum(r["memory_efficiency"] for r in successful_runs) / len(successful_runs)

        print(f"🎯 Maximum successful tokens: {max_tokens_success:,}")
        print(f"⏱️  Average TTFT: {avg_ttft:.3f} s")
        print(f"📊 Average Peak Memory: {avg_peak_mem:.2f}GB")
        print(f"🎯 Average Efficiency: {avg_efficiency:.1%}")

        # Power summary if monitoring was enabled
        if enable_power_monitoring and torch.cuda.is_available():
            print(f"⚡ Power monitoring completed")

    print(f"\n🔧 Configuration: A100-80GB simulated with 24GB constraint (24/80 fraction)")
    print(f"📦 Hetero-KV: 64 sink + 8192 tail + 4-bit compression")

    # Save results
    save_results(results, "a100_benchmark_results.json")

    # Cleanup
    del emulator

    return results


def save_results(results, filename):
    """Save benchmark results to JSON file"""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    output = {
        "timestamp": timestamp,
        "benchmark_type": "A100-Enhanced Hetero-KV",
        "configuration": {
            "memory_limit_gb": 24.0,
            "a100_fraction": 24.0/80.0,
            "gpu_model": "A100-80GB (simulated)"
        },
        "results": results,
        "summary": {
            "total_runs": len(results),
            "successful_runs": len([r for r in results if "oom" not in r and "error" not in r]),
            "max_successful_tokens": max([r["tokens"] for r in results if "oom" not in r and "error" not in r])
                if [r for r in results if "oom" not in r and "error" not in r] else 0
        }
    }

    results_file = Path("logs") / filename
    with open(results_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n💾 Results saved to: {results_file}")


def main():
    """Main benchmark entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="A100-Enhanced Hetero-KV Benchmark")
    parser.add_argument("--tokens", nargs="+", type=int,
                       default=[45025, 65536, 128000],
                       help="Token lengths to test")
    parser.add_argument("--no-power", action="store_true",
                       help="Disable power monitoring")

    args = parser.parse_args()

    # Run benchmark
    run_a100_benchmark(
        token_lengths=args.tokens,
        enable_power_monitoring=not args.no_power
    )


if __name__ == "__main__":
    main()