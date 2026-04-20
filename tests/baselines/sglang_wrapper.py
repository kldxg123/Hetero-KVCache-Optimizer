"""
tests/baselines/sglang_wrapper.py
==================================
SGLang baseline wrapper for rigorous comparison.

If SGLang is not installed, the wrapper degrades gracefully.
"""

import gc
import time
import torch
from typing import Dict, Any, List


class SGLangBaseline:
    """SGLang baseline with memory-bounded execution."""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        mem_fraction_static: float = 0.85,
        max_model_len: int = 32768,
        dtype: str = "bfloat16",
    ):
        self.model_path = model_path
        self.device = device
        self.mem_fraction_static = mem_fraction_static
        self.max_model_len = max_model_len
        self.dtype = dtype
        self._runtime = None
        self._tokenizer = None
        self._available = False
        self._init_engine()

    def _init_engine(self) -> bool:
        try:
            from sglang import Runtime
            from transformers import AutoTokenizer
        except ImportError:
            print("[SGLang Baseline] sglang package not installed. Skipping.")
            return False

        print(f"[SGLang Baseline] Loading {self.model_path} ...")
        try:
            self._runtime = Runtime(
                model_path=self.model_path,
                tp_size=1,
                mem_fraction_static=self.mem_fraction_static,
                max_model_len=self.max_model_len,
                dtype=self.dtype,
                trust_remote_code=True,
            )
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, trust_remote_code=True
            )
            self._available = True
            print("[SGLang Baseline] Engine ready.")
            return True
        except Exception as e:
            print(f"[SGLang Baseline] Failed to initialize engine: {e}")
            return False

    @property
    def available(self) -> bool:
        return self._available

    def run_prefill_decode(self, prompt: str, min_new_tokens: int = 20) -> Dict[str, Any]:
        """Measure end-to-end latency for a single prompt."""
        if not self._available or self._runtime is None:
            return {"success": False, "error": "SGLang engine unavailable"}

        try:
            from sglang import Runtime
        except ImportError:
            return {"success": False, "error": "sglang not installed"}

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        gc.collect()

        try:
            t0 = time.time()
            state = self._runtime.submit(
                {
                    "text": prompt,
                    "sampling_params": {
                        "max_new_tokens": min_new_tokens,
                        "temperature": 0.0,
                    },
                }
            )
            result = self._runtime.generate(state)
            torch.cuda.synchronize()
            t1 = time.time()

            total_time = t1 - t0
            num_prompt_tokens = len(self._tokenizer.encode(prompt))
            # Approximate token count from generated text
            approx_output_tokens = min_new_tokens

            approx_ttft = total_time * (
                num_prompt_tokens / (num_prompt_tokens + approx_output_tokens)
            )
            approx_tpot = (total_time - approx_ttft) / max(approx_output_tokens, 1)
            peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)

            return {
                "success": True,
                "approx_ttft": approx_ttft,
                "approx_tpot": approx_tpot,
                "total_time": total_time,
                "peak_memory_gb": peak_mem,
                "prompt_tokens": num_prompt_tokens,
                "output_tokens": approx_output_tokens,
                "generated_text": result.get("text", ""),
            }
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                gc.collect()
                return {"success": False, "error": "OOM"}
            return {"success": False, "error": str(e)}


def run_sglang_baseline(
    model_path: str,
    prompts: List[str],
    mem_fraction_static: float = 0.85,
    min_new_tokens: int = 20,
) -> List[Dict[str, Any]]:
    """Convenience function to benchmark a list of prompts."""
    wrapper = SGLangBaseline(
        model_path=model_path,
        mem_fraction_static=mem_fraction_static,
    )
    results = []
    for p in prompts:
        results.append(wrapper.run_prefill_decode(p, min_new_tokens))
    return results


if __name__ == "__main__":
    import sys
    model = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2-VL-7B-Instruct"
    prompt = "The quick brown fox jumps over the lazy dog. " * 200
    result = run_sglang_baseline(model, [prompt])
    print(result[0])
