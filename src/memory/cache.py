"""
src/memory/cache.py
===================
HeteroTransientCache: Lightweight HF-compatible cache backed by HeteroKVManager.

This is a streamlined DynamicCache adapter for scenarios that do not require
chunked prefill orchestration or Triton fusion kernels.
"""

import torch
from transformers.cache_utils import DynamicCache

from src.memory.manager import HeteroKVManager


class HeteroTransientCache(DynamicCache):
    """
    Lightweight transient-separation cache using HeteroKVManager as the
    tiered storage backend.
    """

    def __init__(
        self,
        num_layers: int = 28,
        sink_tokens: int = 64,
        keep_tail: int = 8192,
        device: str = "cuda",
    ):
        super().__init__()
        self.sink_tokens = sink_tokens
        self.keep_tail = keep_tail
        self.real_seq_len: int = 0
        self._manager = HeteroKVManager(
            num_layers=num_layers,
            sink_tokens=sink_tokens,
            hbm_budget_tokens=keep_tail,
            device=device,
            enable_quant=False,      # lightweight path: no quantization
            enable_prefetch=False,   # lightweight path: no prefetching
        )

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        new_len = key_states.shape[-2]
        mode = "prefill" if new_len > 1 else "decode"
        out_k, out_v = self._manager.update(
            layer_idx=layer_idx,
            key_states=key_states,
            value_states=value_states,
            mode=mode,
            seq_offset=self.real_seq_len,
        )
        if layer_idx == 0:
            self.real_seq_len += new_len
        return out_k, out_v

    def get_seq_length(self, layer_idx=0):
        return self.real_seq_len

    @property
    def seen_tokens(self) -> int:
        return self.real_seq_len
