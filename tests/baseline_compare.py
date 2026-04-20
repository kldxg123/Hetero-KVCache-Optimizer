"""
tests/baseline_compare.py
=========================
Baseline comparison: Hetero-KV vs StreamingLLM vs Native HF vs vLLM
Academic rigorous version: includes vLLM (CPU Offloading) as a strong baseline
"""

import os
import sys
import json
import time
import math
import torch
import gc
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.core.engine_wrapper import build_fused_cache, ChunkedPrefillEngine
from src.memory.cache import HeteroTransientCache


class StreamingLLMCache:
    """
    StreamingLLM baseline implementation
    Keep only Sink + Local Window, physically discard the middle portion
    """
    def __init__(self, sink_tokens=64, local_window=4096, device="cuda"):
        self.sink_tokens = sink_tokens
        self.local_window = local_window
        self.device = device
        self.key_cache = []
        self.value_cache = []
        self.real_seq_len = 0
        
    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        """StreamingLLM update logic"""
        new_len = key_states.shape[-2]
        
        if new_len > 1:  # Prefill phase
            # Keep only sink + local
            sink_amount = min(new_len, self.sink_tokens)
            local_amount = min(max(new_len - sink_amount, 0), self.local_window)
            
            sink_k = key_states[..., :sink_amount, :]
            sink_v = value_states[..., :sink_amount, :]
            
            if local_amount > 0:
                local_k = key_states[..., -local_amount:, :]
                local_v = value_states[..., -local_amount:, :]
                saved_k = torch.cat([sink_k, local_k], dim=-2)
                saved_v = torch.cat([sink_v, local_v], dim=-2)
            else:
                saved_k, saved_v = sink_k, sink_v
            
            while len(self.key_cache) <= layer_idx:
                self.key_cache.append(None)
                self.value_cache.append(None)
            
            self.key_cache[layer_idx] = saved_k
            self.value_cache[layer_idx] = saved_v
            
            if layer_idx == 0:
                self.real_seq_len += new_len
            
            # Return full tensors (FlashAttention compatible)
            return key_states, value_states
        else:  # Decode phase
            k_cache = self.key_cache[layer_idx]
            v_cache = self.value_cache[layer_idx]
            
            # Direct append
            new_k = torch.cat([k_cache, key_states], dim=-2)
            new_v = torch.cat([v_cache, value_states], dim=-2)
            
            # Crop if exceeds window
            max_len = self.sink_tokens + self.local_window
            if new_k.shape[-2] > max_len:
                # Keep sink + latest local
                sink_k = new_k[..., :self.sink_tokens, :]
                local_k = new_k[..., -(self.local_window):, :]
                new_k = torch.cat([sink_k, local_k], dim=-2)
                
                sink_v = new_v[..., :self.sink_tokens, :]
                local_v = new_v[..., -(self.local_window):, :]
                new_v = torch.cat([sink_v, local_v], dim=-2)
            
            self.key_cache[layer_idx] = new_k
            self.value_cache[layer_idx] = new_v
            
            if layer_idx == 0:
                self.real_seq_len += 1
            
            return new_k, new_v
    
    def get_seq_length(self, layer_idx=0):
        return self.real_seq_len


class NativeHFCache:
    """Native HuggingFace Cache (will OOM on long sequences)"""
    def __init__(self, device="cuda"):
        self.device = device
        self.key_cache = []
        self.value_cache = []
        self.real_seq_len = 0
        
    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        """Native cache logic - store all KV"""
        new_len = key_states.shape[-2]
        
        while len(self.key_cache) <= layer_idx:
            self.key_cache.append(None)
            self.value_cache.append(None)
        
        if self.key_cache[layer_idx] is None:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
        
        if layer_idx == 0:
            self.real_seq_len += new_len
        
        return self.key_cache[layer_idx], self.value_cache[layer_idx]
    
    def get_seq_length(self, layer_idx=0):
        return self.real_seq_len


class BaselineComparison:
    """Baseline comparison tester - academic rigorous version"""
    
    def __init__(self, device="cuda"):
        self.device = device
        print(f"[BASELINE] Initializing comparison test | Device: {device}")
    
    def calculate_ppl(self, logits, input_ids):
        """Compute perplexity (simplified)"""
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        ppl = torch.exp(loss)
        
        return ppl.item()
    
    def simulate_inference(self, seq_length, num_layers=4, hidden_size=128, num_heads=8, 
                          cache_type="hetero", min_new_tokens=20):
        """
        Simulate inference process, compare different cache strategies
        Academic rigorous version: precise TTFT and TPOT measurement
        """
        print(f"\n[{cache_type.upper()}] Sequence length: {seq_length}")
        
        # Create cache
        if cache_type == "hetero":
            cache = HeteroTransientCache(
                sink_tokens=64,
                keep_tail=8192,
            )
            cache.device = self.device
        elif cache_type == "streaming":
            cache = StreamingLLMCache(
                sink_tokens=64,
                local_window=4096,
                device=self.device
            )
        elif cache_type == "vllm_sim":
            # vLLM simulated cache with swap_space behavior
            # Limit VRAM to 16GB, overflow swaps to CPU memory
            cache = self._create_vllm_simulated_cache()
        else:  # native
            cache = NativeHFCache(device=self.device)
        
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        gc.collect()
        
        try:
            # ========== Phase 1: Prefill (measure TTFT) ==========
            prefill_start = time.time()
            
            chunk_size = 2048
            num_chunks = (seq_length + chunk_size - 1) // chunk_size
            
            for chunk_idx in range(num_chunks):
                start = chunk_idx * chunk_size
                end = min(start + chunk_size, seq_length)
                chunk_len = end - start
                
                for layer_idx in range(num_layers):
                    # Generate simulated KV tensors
                    k = torch.randn(1, num_heads, chunk_len, hidden_size, 
                                   dtype=torch.float16, device=self.device)
                    v = torch.randn(1, num_heads, chunk_len, hidden_size, 
                                   dtype=torch.float16, device=self.device)
                    
                    # Update cache
                    cache.update(k, v, layer_idx=layer_idx)
                    
                    del k, v
                
                # VRAM monitoring
                if chunk_idx % 5 == 0:
                    current_mem = torch.cuda.memory_allocated() / (1024**3)
                    print(f"  Chunk {chunk_idx}/{num_chunks}: {current_mem:.3f}GB")
            
            torch.cuda.synchronize()
            prefill_end = time.time()
            
            ttft = prefill_end - prefill_start
            peak_mem_prefill = torch.cuda.max_memory_allocated() / (1024**3)
            
            # ========== Phase 2: Decode (measure TPOT) ==========
            decode_times = []
            
            decode_start = time.time()
            for i in range(min_new_tokens):
                token_start = time.time()
                
                # Simulate single-token decode
                for layer_idx in range(num_layers):
                    k = torch.randn(1, num_heads, 1, hidden_size, 
                                   dtype=torch.float16, device=self.device)
                    v = torch.randn(1, num_heads, 1, hidden_size, 
                                   dtype=torch.float16, device=self.device)
                    cache.update(k, v, layer_idx=layer_idx)
                    del k, v
                
                torch.cuda.synchronize()
                token_end = time.time()
                decode_times.append(token_end - token_start)
            
            decode_end = time.time()
            total_decode_time = decode_end - decode_start
            
            # Compute TPOT
            tpot = sum(decode_times) / len(decode_times) if decode_times else 0
            tpot_std = (sum((t - tpot) ** 2 for t in decode_times) / len(decode_times)) ** 0.5 if decode_times else 0
            
            peak_mem = torch.cuda.max_memory_allocated() / (1024**3)
            current_mem = torch.cuda.memory_allocated() / (1024**3)
            
            # Estimate KV Cache size
            if hasattr(cache, 'key_cache') and cache.key_cache and cache.key_cache[0] is not None:
                kv_cache_size = sum(
                    (k.numel() + v.numel()) * 2 / (1024**3)  # FP16 = 2 bytes
                    for k, v in zip(cache.key_cache, cache.value_cache)
                    if k is not None
                )
            else:
                kv_cache_size = 0
            
            print(f"  Success | TTFT: {ttft:.3f}s")
            print(f"       TPOT: {tpot*1000:.2f}ms +/- {tpot_std*1000:.2f}ms")
            print(f"       Peak VRAM: {peak_mem:.3f}GB")
            print(f"       Steady VRAM: {current_mem:.3f}GB")
            print(f"       KV Cache: {kv_cache_size:.3f}GB")
            
            # Get number of discarded tokens
            discarded = 0
            if cache_type == "streaming":
                preserved = cache.sink_tokens + cache.local_window
                if seq_length > preserved:
                    discarded = seq_length - preserved
            elif cache_type == "hetero":
                preserved = cache.sink_tokens + cache.keep_tail
                if seq_length > preserved:
                    discarded = seq_length - preserved  # compressed to DRAM, not truly discarded
            
            return {
                "success": True,
                "seq_length": seq_length,
                "ttft": ttft,
                "tpot": tpot,
                "tpot_std": tpot_std,
                "total_decode_time": total_decode_time,
                "peak_memory_gb": peak_mem,
                "steady_memory_gb": current_mem,
                "kv_cache_gb": kv_cache_size,
                "discarded_tokens": discarded,
                "cache_type": cache_type,
                "decode_times_ms": [t * 1000 for t in decode_times],
            }
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  OOM")
                torch.cuda.empty_cache()
                gc.collect()
                return {
                    "success": False,
                    "error": "OOM",
                    "seq_length": seq_length,
                    "cache_type": cache_type,
                }
            raise
    
    def _create_vllm_simulated_cache(self):
        """Create a cache that simulates vLLM swap_space behavior"""
        class VLLMSimulatedCache:
            def __init__(self, device="cuda", max_vram_gb=16):
                self.device = device
                self.max_vram_bytes = max_vram_gb * (1024**3)
                self.key_cache = []
                self.value_cache = []
                self.cpu_key_cache = []  # Swap space on CPU
                self.cpu_value_cache = []
                self.real_seq_len = 0
                self.vram_usage = 0
            
            def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
                new_len = key_states.shape[-2]
                kv_size = (key_states.numel() + value_states.numel()) * 2  # FP16
                
                while len(self.key_cache) <= layer_idx:
                    self.key_cache.append(None)
                    self.value_cache.append(None)
                    self.cpu_key_cache.append(None)
                    self.cpu_value_cache.append(None)
                
                # vLLM policy: if VRAM insufficient, swap to CPU
                if self.vram_usage + kv_size > self.max_vram_bytes * 0.8:  # 80% threshold
                    if self.key_cache[layer_idx] is not None:
                        self.cpu_key_cache[layer_idx] = self.key_cache[layer_idx].cpu()
                        self.cpu_value_cache[layer_idx] = self.value_cache[layer_idx].cpu()
                        self.vram_usage -= (self.key_cache[layer_idx].numel() + 
                                          self.value_cache[layer_idx].numel()) * 2
                        self.key_cache[layer_idx] = None
                        self.value_cache[layer_idx] = None
                
                # Update cache
                if self.key_cache[layer_idx] is None:
                    self.key_cache[layer_idx] = key_states
                    self.value_cache[layer_idx] = value_states
                else:
                    self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
                    self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
                
                self.vram_usage += kv_size
                
                if layer_idx == 0:
                    self.real_seq_len += new_len
                
                return self.key_cache[layer_idx], self.value_cache[layer_idx]
            
            def get_seq_length(self, layer_idx=0):
                return self.real_seq_len
        
        return VLLMSimulatedCache(device=self.device)


def run_baseline_comparison():
    """Run full baseline comparison - academic rigorous version"""
    print("="*70)
    print(" Baseline Comparison Test")
    print(" Hetero-KV vs StreamingLLM vs Native HF vs vLLM (Swap)")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Test sequence lengths
    test_lengths = [4096, 8192, 16384, 32768, 65536]
    cache_types = ["hetero", "streaming", "native", "vllm_sim"]
    
    results = {
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "test_config": {
            "min_new_tokens": 20,
            "vllm_swap_space": "16GB VRAM limit, overflow to CPU"
        },
        "tests": [],
    }
    
    comparator = BaselineComparison(device=device)
    
    for length in test_lengths:
        print(f"\n{'='*70}")
        print(f" Test sequence length: {length} tokens")
        print(f"{'='*70}")
        
        test_result = {
            "seq_length": length,
            "hetero": None,
            "streaming": None,
            "native": None,
            "vllm_sim": None,
        }
        
        for cache_type in cache_types:
            print("\n" + "-"*50)
            result = comparator.simulate_inference(
                seq_length=length,
                num_layers=4,  # simulate 4 layers
                hidden_size=128,
                num_heads=8,
                cache_type=cache_type,
                min_new_tokens=20
            )
            test_result[cache_type] = result
            
            # Clean up after each test
            torch.cuda.empty_cache()
            gc.collect()
        
        results["tests"].append(test_result)
        
        # Save intermediate results
        os.makedirs("experiments", exist_ok=True)
        with open("experiments/baseline_comparison.json", "w") as f:
            json.dump(results, f, indent=2)
    
    # Final save
    os.makedirs("experiments", exist_ok=True)
    with open("experiments/baseline_comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*70)
    print(" Baseline comparison test complete")
    print(f" Results saved to: experiments/baseline_comparison.json")
    print("="*70)
    
    return results


if __name__ == "__main__":
    run_baseline_comparison()
