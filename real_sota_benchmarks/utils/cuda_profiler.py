"""
cuda_profiler.py
================
Core: rigorous TTFT and TPOT timers, plus precise Peak VRAM / Steady VRAM capture.
All time measurements are wrapped in torch.cuda.synchronize();
All VRAM measurements use torch.cuda.max_memory_allocated(), strictly separating
model weights from dynamic KV Cache.
"""

import torch
import time
from typing import Dict, Any, Optional, Callable
from contextlib import contextmanager


@contextmanager
def cuda_timer(name: str, results_dict: Optional[Dict[str, float]] = None):
    """CUDA synchronized context timer."""
    torch.cuda.synchronize()
    start = time.perf_counter()
    try:
        yield
    finally:
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        if results_dict is not None:
            results_dict[name] = elapsed


def reset_memory_stats(device: str = "cuda"):
    """Clear cache and reset peak memory statistics."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)


def get_memory_stats(device: str = "cuda") -> Dict[str, float]:
    """Get current memory statistics (GB)."""
    return {
        "allocated_gb": torch.cuda.memory_allocated(device) / (1024**3),
        "reserved_gb": torch.cuda.memory_reserved(device) / (1024**3),
        "max_allocated_gb": torch.cuda.max_memory_allocated(device) / (1024**3),
    }


def measure_model_weights_memory(model, device: str = "cuda") -> float:
    """
    Measure model weights VRAM (GB).
    For quantized models, accumulates element_size across all parameters.
    """
    total_bytes = 0
    for p in model.parameters():
        if p.is_quantized:
            total_bytes += p.numel() * p.element_size()
        else:
            total_bytes += p.numel() * p.element_size()
    return total_bytes / (1024**3)


def measure_past_key_values_memory(past_key_values) -> float:
    """
    Measure HuggingFace native past_key_values list VRAM (GB).
    past_key_values format: list of tuples, each layer is (key, value).
    """
    total_bytes = 0
    if past_key_values is None:
        return 0.0
    for layer_cache in past_key_values:
        if isinstance(layer_cache, tuple) and len(layer_cache) >= 2:
            k, v = layer_cache[0], layer_cache[1]
            total_bytes += k.numel() * k.element_size()
            total_bytes += v.numel() * v.element_size()
    return total_bytes / (1024**3)


def measure_dynamic_cache_memory(cache) -> float:
    """Measure transformers DynamicCache and its subclasses VRAM (GB)."""
    total_bytes = 0
    if cache is None:
        return 0.0
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        for k in cache.key_cache:
            if isinstance(k, torch.Tensor):
                total_bytes += k.numel() * k.element_size()
        for v in cache.value_cache:
            if isinstance(v, torch.Tensor):
                total_bytes += v.numel() * v.element_size()
    return total_bytes / (1024**3)


class NativeHFProfiler:
    """
    Rigorous profiler for Native HuggingFace generate/manual-decode.
    Separates Prefill (TTFT) and Decode (TPOT), and records memory breakdown.
    """

    def __init__(self, model, device: str = "cuda"):
        self.model = model
        self.device = device
        self.ttft = 0.0
        self.tpot = 0.0
        self.prefill_peak_gb = 0.0
        self.decode_peak_gb = 0.0
        self.steady_memory_gb = 0.0
        self.kv_cache_gb = 0.0
        self.model_weights_gb = measure_model_weights_memory(model, device)

    def run_prefill(self, inputs: Dict[str, torch.Tensor]):
        """Run a rigorous prefill, returning outputs (including past_key_values)."""
        reset_memory_stats(self.device)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = self.model(**inputs, use_cache=True)
        torch.cuda.synchronize()
        self.ttft = time.perf_counter() - t0
        self.prefill_peak_gb = torch.cuda.max_memory_allocated(self.device) / (1024**3)
        self.kv_cache_gb = measure_past_key_values_memory(outputs.past_key_values)
        return outputs

    def run_decode(self, initial_input_ids: torch.Tensor, past_key_values, num_tokens: int = 20):
        """
        Run num_tokens sequential token-by-token decode, computing TPOT and VRAM.
        Returns final generated token ids.
        """
        current_input = initial_input_ids[:, -1:]
        pkv = past_key_values
        decode_times = []

        reset_memory_stats(self.device)
        for _ in range(num_tokens):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                outputs = self.model(input_ids=current_input, past_key_values=pkv, use_cache=True)
            torch.cuda.synchronize()
            decode_times.append(time.perf_counter() - t0)
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
            pkv = outputs.past_key_values
            current_input = next_token

        self.tpot = sum(decode_times) / len(decode_times)
        self.decode_peak_gb = torch.cuda.max_memory_allocated(self.device) / (1024**3)
        self.steady_memory_gb = torch.cuda.memory_allocated(self.device) / (1024**3)
        return current_input

    def run_decode_with_cache_obj(self, initial_input_ids: torch.Tensor, cache_obj, num_tokens: int = 20):
        """
        For StreamingLLM and other custom Cache object decode scenarios.
        cache_obj is updated in-place; no explicit past_key_values needed.
        """
        current_input = initial_input_ids[:, -1:]
        decode_times = []

        reset_memory_stats(self.device)
        for _ in range(num_tokens):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                outputs = self.model(input_ids=current_input, past_key_values=cache_obj, use_cache=True)
            torch.cuda.synchronize()
            decode_times.append(time.perf_counter() - t0)
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
            current_input = next_token

        self.tpot = sum(decode_times) / len(decode_times)
        self.decode_peak_gb = torch.cuda.max_memory_allocated(self.device) / (1024**3)
        self.steady_memory_gb = torch.cuda.memory_allocated(self.device) / (1024**3)
        self.kv_cache_gb = measure_dynamic_cache_memory(cache_obj)
        return current_input

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_weights_gb": round(self.model_weights_gb, 3),
            "ttft_s": round(self.ttft, 4),
            "tpot_ms": round(self.tpot * 1000, 2),
            "prefill_peak_gb": round(self.prefill_peak_gb, 3),
            "decode_peak_gb": round(self.decode_peak_gb, 3),
            "steady_memory_gb": round(self.steady_memory_gb, 3),
            "kv_cache_gb": round(self.kv_cache_gb, 3),
        }


class GenerateProfiler:
    """
    Time/VRAM profiler wrapper for model.generate().
    Measures TTFT with max_new_tokens=1, then total time with max_new_tokens=N to derive TPOT.
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.ttft = 0.0
        self.tpot = 0.0
        self.peak_gb = 0.0
        self.steady_gb = 0.0

    def measure_generate(self, generate_fn: Callable, num_decode_tokens: int = 20, **gen_kwargs):
        """
        generate_fn must accept max_new_tokens and gen_kwargs and return output.
        TTFT and Decode time are separated by two generate calls.
        """
        # TTFT: 1 token
        reset_memory_stats(self.device)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        generate_fn(max_new_tokens=1, **gen_kwargs)
        torch.cuda.synchronize()
        self.ttft = time.perf_counter() - t0

        # Total: N tokens
        reset_memory_stats(self.device)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        outputs = generate_fn(max_new_tokens=num_decode_tokens, **gen_kwargs)
        torch.cuda.synchronize()
        total_time = time.perf_counter() - t0

        self.tpot = (total_time - self.ttft) / max(1, num_decode_tokens - 1)
        self.peak_gb = torch.cuda.max_memory_allocated(self.device) / (1024**3)
        self.steady_gb = torch.cuda.memory_allocated(self.device) / (1024**3)
        return outputs

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ttft_s": round(self.ttft, 4),
            "tpot_ms": round(self.tpot * 1000, 2),
            "peak_gb": round(self.peak_gb, 3),
            "steady_gb": round(self.steady_gb, 3),
        }
