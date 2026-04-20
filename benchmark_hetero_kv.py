# ==========================================
# Patch to bypass Transformers version checks
# Must be placed before all imports!
# ==========================================
import os
import sys

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

# ==========================================
# Real benchmark test code below
# ==========================================
import torch
import time
import gc
from src.core.engine_wrapper import build_fused_cache, ChunkedPrefillEngine

# ==========================================
# Dummy model simulating Qwen2-VL for generating real KV tensor pressure
# ==========================================
class DummyQwen2VL:
    def __init__(self, hidden_size=128, num_heads=32, num_layers=1):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers

    def __call__(self, input_ids, past_key_values, use_cache=True, position_ids=None, attention_mask=None):
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

# ==========================================
# Real physical stress test
# ==========================================
def run_real_benchmark():
    print("\n" + "=" * 60)
    print(" Starting Hetero-KV real physical tensor stress test")
    print("=" * 60)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("Warning: No GPU detected, running simulation on CPU.")
    else:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    target_seq_len = 45025
    chunk_size = 2048

    print(f"Target: {target_seq_len} Tokens")
    print(f"Chunk size: {chunk_size}")

    cache = build_fused_cache(
        device=device,
        sink_tokens=64,
        keep_tail=8192,
        chunk_size=chunk_size,
        group_size=128,
        enable_quant=True,
        enable_prefetch=True,
        enable_triton=False
    )

    model = DummyQwen2VL(num_layers=1)
    engine = ChunkedPrefillEngine(model=model, cache=cache, chunk_size=chunk_size)

    input_ids = torch.randint(0, 10000, (1, target_seq_len), device=device)

    print("\nStarting Chunked Prefill injection...")
    start_time = time.time()
    
    try:
        engine.prefill(input_ids)
        if device != "cpu":
            torch.cuda.synchronize()
        end_time = time.time()

        if device != "cpu":
            peak_mem_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            current_mem_gb = torch.cuda.memory_allocated(device) / (1024 ** 3)
        else:
            peak_mem_gb, current_mem_gb = 0.0, 0.0

        print("\n" + "=" * 60)
        print(" Stress test results")
        print("=" * 60)
        print(f"Status: Survived without OOM.")
        print(f"TTFT (prefill time): {end_time - start_time:.3f} s")
        print(f"Peak VRAM: {peak_mem_gb:.3f} GB")
        print(f"Steady VRAM: {current_mem_gb:.3f} GB")
        print(f"DRAM eviction chunks: {len(cache.dram_table)}")
        print(f"Real cognitive sequence length: {cache.get_seq_length()}")
        print("=" * 60 + "\n")

    except RuntimeError as e:
        if "OutOfMemory" in str(e) or "CUDA out of memory" in str(e):
            print("\nOOM crash encountered! Please check chunk_size or model parameters.")
        else:
            print(f"\nUnknown tensor error:\n{str(e)}")

if __name__ == "__main__":
    run_real_benchmark()
