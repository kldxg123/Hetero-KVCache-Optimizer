"""
tests/baselines/vllm_wrapper.py
================================
vLLM baseline wrapper for rigorous comparison.

Runs inference with PagedAttention under a configurable VRAM budget.
If vLLM is not installed, the wrapper degrades gracefully.
"""

import gc
import time
import torch
from typing import Dict, Any, List, Optional


class VLLMBaseline:
    """vLLM baseline with memory-bounded execution."""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        gpu_memory_utilization: float = 0.90,
        max_model_len: int = 32768,
        dtype: str = "bfloat16",
    ):
        self.model_path = model_path
        self.device = device
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.dtype = dtype
        self._llm = None
        self._tokenizer = None
        self._available = False
        self._init_engine()

    def _init_engine(self) -> bool:
        try:
            from vllm import LLM, SamplingParams
            from transformers import AutoTokenizer
        except ImportError:
            print("[vLLM Baseline] vllm package not installed. Skipping.")
            return False

        print(f"[vLLM Baseline] Loading {self.model_path} ...")
        print(f"[vLLM Baseline] gpu_memory_utilization={self.gpu_memory_utilization}")

        try:
            self._llm = LLM(
                model=self.model_path,
                tokenizer=self.model_path,
                tensor_parallel_size=1,
                gpu_memory_utilization=self.gpu_memory_utilization,
                max_model_len=self.max_model_len,
                dtype=self.dtype,
                trust_remote_code=True,
            )
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, trust_remote_code=True
            )
            self._available = True
            print("[vLLM Baseline] Engine ready.")
            return True
        except Exception as e:
            print(f"[vLLM Baseline] Failed to initialize engine: {e}")
            return False

    @property
    def available(self) -> bool:
        return self._available

    def run_prefill_decode(
        self,
        prompt: str,
        min_new_tokens: int = 20,
    ) -> Dict[str, Any]:
        """Measure TTFT and TPOT for a single prompt."""
        if not self._available or self._llm is None:
            return {"success": False, "error": "vLLM engine unavailable"}

        from vllm import SamplingParams

        sampling_params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=min_new_tokens,
        )

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        gc.collect()

        try:
            # vLLM prefill + decode are fused in one generate() call
            t0 = time.time()
            outputs = self._llm.generate(prompt, sampling_params)
            torch.cuda.synchronize()
            t1 = time.time()

            total_time = t1 - t0
            # Approximate: vLLM doesn't expose per-token decode times directly.
            # We treat total_time as dominated by prefill for long contexts,
            # and derive an approximate TPOT from end-to-end latency.
            num_prompt_tokens = len(self._tokenizer.encode(prompt))
            num_output_tokens = len(outputs[0].outputs[0].token_ids)

            # Heuristic split: TTFT ≈ prefill-dominated portion
            # For PagedAttention, prefill is roughly linear; decode is constant per token.
            # We approximate TTFT by scaling total time by prompt fraction.
            approx_ttft = total_time * (num_prompt_tokens / (num_prompt_tokens + num_output_tokens))
            approx_tpot = (total_time - approx_ttft) / max(num_output_tokens, 1)

            peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)

            return {
                "success": True,
                "approx_ttft": approx_ttft,
                "approx_tpot": approx_tpot,
                "total_time": total_time,
                "peak_memory_gb": peak_mem,
                "prompt_tokens": num_prompt_tokens,
                "output_tokens": num_output_tokens,
            }
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                gc.collect()
                return {"success": False, "error": "OOM"}
            return {"success": False, "error": str(e)}


def run_vllm_baseline(
    model_path: str,
    prompts: List[str],
    gpu_memory_utilization: float = 0.90,
    min_new_tokens: int = 20,
) -> List[Dict[str, Any]]:
    """Convenience function to benchmark a list of prompts."""
    wrapper = VLLMBaseline(
        model_path=model_path,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    results = []
    for p in prompts:
        results.append(wrapper.run_prefill_decode(p, min_new_tokens))
    return results


if __name__ == "__main__":
    import sys
    model = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2-VL-7B-Instruct"
    prompt = "The quick brown fox jumps over the lazy dog. " * 200
    result = run_vllm_baseline(model, [prompt], gpu_memory_utilization=0.90)
    print(result[0])
