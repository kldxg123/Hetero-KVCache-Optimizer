"""
src/memory/manager.py
======================
HeteroKVManager: A Tiered Storage System for KV Cache Management.

Presents KV cache as a heterogeneous memory hierarchy (HBM + DRAM) and provides
a unified abstraction for allocation, update, compression, and retrieval.
"""

import gc
import sys
import os
from typing import Any, Dict, List, Optional, Tuple

import torch

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.quantization.kv_compressor import KVCompressor
from src.memory.dram_storage import DRAMStorageManager
from src.memory.attention_competition_queue import AttentionCompetitionQueue
from src.policy.prefetcher import AsyncPrefetcher
from src.policy.heavy_hitter import HeavyHitterOracle
from src.policy.adaptive_prefetch_controller import AdaptivePrefetchController
from src.core.scheduler import PredictivePrefetchScheduler


class HeteroKVManager:
    """
    Tiered KV Cache Storage System.

    Manages a heterogeneous memory hierarchy where hot KV tokens (Sink + recent Tail)
    reside in HBM, while overflow tokens are transparently compressed and offloaded
    to DRAM. This decouples the logical sequence length from the physical memory
    footprint, enabling O(1) steady-state memory for autoregressive decoding.

    Public API:
      - allocate(layer_idx, seq_len, budget_tokens)
      - update(layer_idx, key_states, value_states, mode, seq_offset)
      - compress(layer_idx, device='DRAM')
      - swap_in(layer_idx, chunk_key)
      - get_hbm_kv(layer_idx)
      - memory_summary()
    """

    def __init__(
        self,
        num_layers: int,
        sink_tokens: int = 64,
        hbm_budget_tokens: int = 8192,
        device: str = "cuda",
        enable_quant: bool = True,
        enable_prefetch: bool = True,
        group_size: int = 128,
        bits: int = 4,
        bandwidth_limiter=None,
    ):
        self.num_layers = num_layers
        self.sink_tokens = sink_tokens
        self.hbm_budget_tokens = hbm_budget_tokens
        self.device = device
        self.enable_quant = enable_quant
        self.group_size = group_size
        self.bits = bits
        self._bandwidth_limiter = bandwidth_limiter

        max_hbm = sink_tokens + hbm_budget_tokens

        # ════════════════════════════════════════════════════════════════
        # 三区域 HBM 分区 (用户设计：Sink + Tail + HeavyHitter)
        # ════════════════════════════════════════════════════════════════
        # Zone 1: Sink - 固定大小，系统提示 tokens
        self._sink_k: List[Optional[torch.Tensor]] = [None] * num_layers
        self._sink_v: List[Optional[torch.Tensor]] = [None] * num_layers

        # Zone 2: Tail - 固定大小，最近上下文 (滑动窗口)
        self._tail_k: List[Optional[torch.Tensor]] = [None] * num_layers
        self._tail_v: List[Optional[torch.Tensor]] = [None] * num_layers

        # Zone 3: HeavyHitter - 动态大小，高注意力 tokens
        self._heavyhitter_k: List[Optional[torch.Tensor]] = [None] * num_layers
        self._heavyhitter_v: List[Optional[torch.Tensor]] = [None] * num_layers
        self._heavyhitter_scores: List[Optional[torch.Tensor]] = [None] * num_layers
        self._heavyhitter_budget = max(hbm_budget_tokens // 2, 2048)

        # Legacy compat: _key_cache = Sink + Tail + HeavyHitter (concatenated view)
        self._key_cache: List[Optional[torch.Tensor]] = [None] * num_layers
        self._value_cache: List[Optional[torch.Tensor]] = [None] * num_layers
        self._seq_offsets: List[int] = [0] * num_layers

        # 注意力竞争队列 (Tail驱逐 + 动态取回竞争 HeavyHitter HBM)
        self._competition_queue = AttentionCompetitionQueue()

        # Compression engine
        self._compressor = KVCompressor(group_size=group_size, bits=bits)

        # Heavy Hitter Oracle for attention-driven eviction
        self._oracle = HeavyHitterOracle(
            block_size=16,
            sink_tokens=sink_tokens,
            local_window=hbm_budget_tokens,
        )

        # Adaptive prefetch controller
        self._adaptive_controller = AdaptivePrefetchController()

        # Async prefetcher for DRAM -> HBM overlap
        self._prefetcher: Optional[AsyncPrefetcher] = None
        if enable_prefetch and torch.cuda.is_available():
            self._prefetcher = AsyncPrefetcher(device=torch.device(device))

        # Predictive prefetch scheduler
        self._predictive_scheduler: Optional[PredictivePrefetchScheduler] = None

        # DRAM tier-2 storage
        self._dram = DRAMStorageManager()
        self._eviction_counter = 0

        # Adaptive self-healing: track chunk metadata for dynamic window retrieval
        self._chunk_eviction_order: List[str] = []  # Track eviction order
        self._chunk_attention_scores: Dict[str, float] = {}  # chunk_key -> avg score

        # ──────────────────────────────────────────────────────────────
        # Oracle 集成：存储最近的注意力权重
        # 用途：供 AdaptivePrefetchController 计算动态窗口 w_t
        # 时序：update_attention_scores() 存入 → get_dram_chunks_quantized_adaptive() 消费
        # ──────────────────────────────────────────────────────────────
        self._last_attention_weights: Optional[torch.Tensor] = None

    @property
    def _dram_table(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """Backward-compatible access to the DRAM storage dict."""
        return self._dram.table

    # ------------------------------------------------------------------
    # Tiered Storage API
    # ------------------------------------------------------------------

    def allocate(
        self,
        layer: int,
        budget: Optional[int] = None,
    ) -> bool:
        """
        Ensure the physical HBM pool for `layer` can accommodate up to
        `budget` tokens. If budget is None, uses hbm_budget_tokens.

        Returns True if the layer is ready for writes.
        """
        if layer < 0 or layer >= self.num_layers:
            raise ValueError(f"Invalid layer {layer}, num_layers={self.num_layers}")
        if budget is not None:
            self.hbm_budget_tokens = budget
        # Allocation is lazy in this implementation; pools are created on first update.
        # Future extensions could pre-allocate fixed-size tensors here for true zero-fragmentation.
        return True

    def update(
        self,
        layer_idx: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        mode: str = "prefill",
        seq_offset: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Unified update entry for both prefill and decode modes.

        Args:
            layer_idx: Transformer layer index.
            key_states:  New key tensor to integrate.
            value_states: New value tensor to integrate.
            mode:        Either "prefill" (seq_len > 1) or "decode" (seq_len == 1).
            seq_offset:  Logical sequence offset for this update (used for RoPE alignment).

        Returns:
            (key_states, value_states) that should be presented to the attention kernel.
            In prefill mode, this returns the *full* original tensors to maintain
            FlashAttention compatibility (transient architecture).
            In decode mode, this returns the *pruned* HBM-resident tensors.
        """
        if mode == "prefill":
            return self._prefill_update(layer_idx, key_states, value_states, seq_offset)
        elif mode == "decode":
            return self._decode_update(layer_idx, key_states, value_states, seq_offset)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def compress(self, layer: int, device: str = "DRAM") -> int:
        """
        Force-compress all HBM-resident tail tokens beyond sink_tokens for the given
        layer and move them to the specified device tier.

        Returns:
            Number of tokens compressed.
        """
        if device.upper() != "DRAM":
            raise NotImplementedError("Only DRAM compression is currently supported.")
        k_cache = self._key_cache[layer]
        v_cache = self._value_cache[layer]
        if k_cache is None or k_cache.shape[-2] <= self.sink_tokens:
            return 0

        overflow = k_cache.shape[-2] - self.sink_tokens
        k_sink = k_cache[..., : self.sink_tokens, :]
        v_sink = v_cache[..., : self.sink_tokens, :]
        k_tail = k_cache[..., self.sink_tokens :, :]
        v_tail = v_cache[..., self.sink_tokens :, :]

        self._evict_to_dram(layer, k_tail, v_tail)

        self._key_cache[layer] = k_sink
        self._value_cache[layer] = v_sink
        return overflow

    def swap_in_quantized(
        self, layer_idx: int, chunk_key: str
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        Swap a compressed chunk from DRAM back into HBM *without decompressing*.
        Returns the raw quantized tensors for use with fused_dequant_attn,
        eliminating the BF16 intermediate memory spike.

        Returns dict with keys: k_data, k_scales, k_zps, v_data, v_scales, v_zps
        (all on self.device), or None if chunk_key missing.
        """
        if chunk_key not in self._dram:
            return None

        entry = self._dram.retrieve(chunk_key)
        if entry is None:
            return None

        result = {}
        for key in ("k_data", "k_scales", "k_zps", "v_data", "v_scales", "v_zps"):
            t = entry.get(key)
            if t is not None:
                result[key] = t.to(self.device, non_blocking=True)
            else:
                return None

        if self._bandwidth_limiter is not None:
            for t in result.values():
                self._bandwidth_limiter.simulate_transfer(t)
        torch.cuda.synchronize(self.device)
        return result

    def swap_in(
        self, layer_idx: int, chunk_key: str
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Swap a compressed chunk from DRAM back into HBM.

        Returns:
            (restored_k, restored_v) as BF16 tensors, or None if chunk_key missing.
        """
        if not self._dram.contains(chunk_key):
            return None

        entry = self._dram.remove(chunk_key)

        # Fast path: prefetch hit
        if self._prefetcher is not None:
            result = self._prefetcher.fetch_if_ready(chunk_key)
            if result is not None:
                prefetched_k, _ = result
                q_v = entry["v_data"].to(self.device, non_blocking=True)
                s_v = entry["v_scales"].to(self.device, non_blocking=True)
                z_v = entry["v_zps"].to(self.device, non_blocking=True)
                if self._bandwidth_limiter is not None:
                    for t in (q_v, s_v, z_v):
                        self._bandwidth_limiter.simulate_transfer(t)
                torch.cuda.synchronize(self.device)
                restored_v = self._compressor.decompress(q_v, s_v, z_v).to(torch.bfloat16)
                return prefetched_k, restored_v

        # Slow path: synchronous decompress
        q_k = entry["k_data"].to(self.device, non_blocking=True)
        s_k = entry["k_scales"].to(self.device, non_blocking=True)
        z_k = entry["k_zps"].to(self.device, non_blocking=True)
        q_v = entry["v_data"].to(self.device, non_blocking=True)
        s_v = entry["v_scales"].to(self.device, non_blocking=True)
        z_v = entry["v_zps"].to(self.device, non_blocking=True)
        if self._bandwidth_limiter is not None:
            for t in (q_k, s_k, z_k, q_v, s_v, z_v):
                self._bandwidth_limiter.simulate_transfer(t)
        torch.cuda.synchronize(self.device)
        restored_k = self._compressor.decompress(q_k, s_k, z_k).to(torch.bfloat16)
        restored_v = self._compressor.decompress(q_v, s_v, z_v).to(torch.bfloat16)
        return restored_k, restored_v

    def get_hbm_kv(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the current HBM-resident KV tensors for the given layer."""
        if self._key_cache[layer_idx] is None:
            raise RuntimeError(f"Layer {layer_idx} has not been allocated yet.")
        return self._key_cache[layer_idx], self._value_cache[layer_idx]

    def get_seq_offset(self, layer_idx: int) -> int:
        """Return the logical sequence offset maintained for RoPE alignment."""
        return self._seq_offsets[layer_idx]

    def update_attention_scores(self, attention_weights: torch.Tensor) -> None:
        """
        Phase D: Feed attention weights from the latest decode step to the
        HeavyHitterOracle for cumulative importance tracking.

        Also stores the weights for AdaptivePrefetchController to compute
        dynamic window w_t based on attention volatility σ(A_t).
        """
        self._oracle.update(attention_weights)
        # Store for adaptive controller (copy to avoid detachment issues)
        self._last_attention_weights = attention_weights.detach().clone()

    def get_oracle_scores(self) -> Optional[torch.Tensor]:
        """Return current cumulative attention scores (for debugging/testing)."""
        return self._oracle.token_scores

    def schedule_prefetch(self, chunk_key: str) -> None:
        """Submit an asynchronous DRAM -> HBM prefetch task."""
        if self._prefetcher is None or not self._dram.contains(chunk_key):
            return
        entry = self._dram.retrieve(chunk_key)
        prefetch_entry = {
            "k_data": entry["k_data"],
            "k_scales": entry["k_scales"],
            "k_zps": entry["k_zps"],
            "v_data": entry["v_data"],
            "v_scales": entry["v_scales"],
            "v_zps": entry["v_zps"],
        }
        self._prefetcher.submit_prefetch_task(chunk_key, prefetch_entry, self._compressor)

    def predictive_prefetch_step(
        self,
        current_chunk: Optional[str] = None,
        attention_weights: Optional[torch.Tensor] = None,
        cache_miss: bool = False,
    ) -> List[str]:
        """
        Proactive prefetch: predict which DRAM chunks are needed next
        and submit async H2D transfers ahead of time.

        Integrates AdaptivePrefetchController for dynamic window sizing.
        """
        if self._prefetcher is None:
            return []

        # Lazy-init the predictive scheduler
        if self._predictive_scheduler is None and self._dram.num_entries > 0:
            self._predictive_scheduler = PredictivePrefetchScheduler(
                prefetcher=self._prefetcher,
                dram_table=self._dram.table,
                compressor=self._compressor,
            )

        if self._predictive_scheduler is not None:
            # AdaptivePrefetchController: dynamically adjust lookahead window
            new_window = self._adaptive_controller.compute_window(
                attention_weights=attention_weights,
                cache_miss=cache_miss,
            )
            self._predictive_scheduler.lookahead_window = new_window

            return self._predictive_scheduler.schedule_step(
                current_chunk=current_chunk,
                attention_weights=attention_weights,
            )
        return []

    def memory_summary(self) -> Dict[str, Any]:
        """Return a snapshot of HBM and DRAM consumption."""
        hbm_tokens = 0
        for k in self._key_cache:
            if k is not None:
                hbm_tokens += k.shape[-2]

        dram_summary = self._dram.memory_summary()

        return {
            "hbm_tokens": hbm_tokens,
            "dram_entries": dram_summary["num_entries"],
            "dram_bytes": dram_summary["total_bytes"],
            "max_hbm_tokens": self.max_hbm_tokens(),
        }

    def max_hbm_tokens(self) -> int:
        """
        返回HBM中能存储的最大token数（三区域架构）

        总HBM预算 = Sink + Tail + HeavyHitter
        = sink_tokens + hbm_budget_tokens + heavyhitter_budget
        """
        return self.sink_tokens + self.hbm_budget_tokens + self._heavyhitter_budget

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prefill_update(
        self,
        layer_idx: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        seq_offset: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        三区域架构 prefill 更新逻辑（分段预填充）

        流程：
        1. 提取Sink（开头64个tokens）→ Sink HBM分区
        2. 提取Tail（结尾2048个tokens）→ Tail HBM分区
        3. 中间tokens → 压缩到DRAM
        4. 初始化HeavyHitter分区（prefill阶段为空，后续通过竞争队列填充）
        5. 返回 Sink + Tail + HeavyHitter（初始为空）

        注意：Prefill阶段没有注意力分数，HeavyHitter分区在decode阶段动态填充
        """
        new_len = key_states.shape[-2]

        # 初始化三个分区（如果需要）
        while len(self._sink_k) <= layer_idx:
            self._sink_k.append(None)
            self._sink_v.append(None)
            self._tail_k.append(None)
            self._tail_v.append(None)
            self._heavyhitter_k.append(None)
            self._heavyhitter_v.append(None)
            self._heavyhitter_scores.append(None)
            self._key_cache.append(None)
            self._value_cache.append(None)
            self._seq_offsets.append(0)

        # ════════════════════════════════════════════════════════════════
        # 增量预填充：如果Sink已有数据，说明这是后续chunk
        # 保留Sink，将新chunk追加到Tail，超量部分驱逐到DRAM
        # ════════════════════════════════════════════════════════════════
        if self._sink_k[layer_idx] is not None:
            return self._incremental_prefill_update(
                layer_idx, key_states, value_states, seq_offset
            )

        # ════════════════════════════════════════════════════════════════
        # Step 1: 提取Sink（开头固定tokens）
        # ════════════════════════════════════════════════════════════════
        sink_amt = min(new_len, self.sink_tokens)

        if sink_amt > 0:
            self._sink_k[layer_idx] = key_states[..., :sink_amt, :].clone()
            self._sink_v[layer_idx] = value_states[..., :sink_amt, :].clone()
        else:
            self._sink_k[layer_idx] = torch.empty(
                key_states.shape[0], key_states.shape[1], 0, key_states.shape[3],
                device=key_states.device, dtype=key_states.dtype
            )
            self._sink_v[layer_idx] = torch.empty(
                value_states.shape[0], value_states.shape[1], 0, value_states.shape[3],
                device=value_states.device, dtype=value_states.dtype
            )

        # ════════════════════════════════════════════════════════════════
        # Step 2: 提取Tail（结尾固定tokens，滑动窗口）
        # ════════════════════════════════════════════════════════════════
        tail_budget = self.hbm_budget_tokens - self.sink_tokens
        tail_amt = min(new_len - sink_amt, tail_budget)

        if tail_amt > 0:
            self._tail_k[layer_idx] = key_states[..., -tail_amt:, :].clone()
            self._tail_v[layer_idx] = value_states[..., -tail_amt:, :].clone()
        else:
            self._tail_k[layer_idx] = torch.empty(
                key_states.shape[0], key_states.shape[1], 0, key_states.shape[3],
                device=key_states.device, dtype=key_states.dtype
            )
            self._tail_v[layer_idx] = torch.empty(
                value_states.shape[0], value_states.shape[1], 0, value_states.shape[3],
                device=value_states.device, dtype=value_states.dtype
            )

        # ════════════════════════════════════════════════════════════════
        # Step 3: 中间tokens → 压缩到DRAM
        # ════════════════════════════════════════════════════════════════
        body_start = sink_amt
        body_end = new_len - tail_amt

        if self.enable_quant and body_end > body_start:
            # 提取中间tokens
            body_k = key_states[..., body_start:body_end, :]
            body_v = value_states[..., body_start:body_end, :]

            # 压缩并存储到DRAM
            self._evict_to_dram(layer_idx, body_k, body_v)

        # ════════════════════════════════════════════════════════════════
        # Step 4: 初始化HeavyHitter分区（初始为空）
        # ════════════════════════════════════════════════════════════════
        # Prefill阶段没有注意力分数，HeavyHitter初始为空
        # 在decode阶段通过竞争队列动态填充
        self._heavyhitter_k[layer_idx] = torch.empty(
            key_states.shape[0], key_states.shape[1], 0, key_states.shape[3],
            device=key_states.device, dtype=key_states.dtype
        )
        self._heavyhitter_v[layer_idx] = torch.empty(
            value_states.shape[0], value_states.shape[1], 0, value_states.shape[3],
            device=value_states.device, dtype=value_states.dtype
        )
        self._heavyhitter_scores[layer_idx] = torch.empty(
            0, device=key_states.device, dtype=torch.float32
        )

        # ════════════════════════════════════════════════════════════════
        # Step 5: 更新legacy cache
        # ════════════════════════════════════════════════════════════════
        self._update_legacy_cache(layer_idx)
        self._seq_offsets[layer_idx] = seq_offset + new_len

        # Prefill: return FULL original K/V so self-attention computes correctly.
        # Truncation to Sink+Tail+HH takes effect during decode.
        return key_states, value_states

    def _incremental_prefill_update(
        self,
        layer_idx: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        seq_offset: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        增量预填充：处理后续chunk。

        保留Sink（首个chunk的开头tokens），将新chunk追加到Tail。
        如果Tail超过预算，驱逐旧Tail开头tokens到DRAM。
        返回 Sink + old_tail + new_chunk 以保证注意力计算正确。
        """
        new_len = key_states.shape[-2]
        tail_budget = self.hbm_budget_tokens - self.sink_tokens

        old_tail_k = self._tail_k[layer_idx]
        old_tail_v = self._tail_v[layer_idx]

        # Combine old Tail + current chunk
        combined_k = torch.cat([old_tail_k, key_states], dim=-2)
        combined_v = torch.cat([old_tail_v, value_states], dim=-2)

        # Return Sink + combined (for correct inter-chunk attention)
        return_k = torch.cat([self._sink_k[layer_idx], combined_k], dim=-2)
        return_v = torch.cat([self._sink_v[layer_idx], combined_v], dim=-2)

        # Evict excess from the beginning of combined Tail → DRAM
        combined_len = combined_k.shape[-2]
        if combined_len > tail_budget:
            evict_count = combined_len - tail_budget
            evicted_k = combined_k[:, :, :evict_count, :]
            evicted_v = combined_v[:, :, :evict_count, :]

            if self.enable_quant and evict_count > 0:
                self._evict_to_dram(layer_idx, evicted_k, evicted_v)

            # Keep last tail_budget tokens
            self._tail_k[layer_idx] = combined_k[:, :, evict_count:, :].clone()
            self._tail_v[layer_idx] = combined_v[:, :, evict_count:, :].clone()
        else:
            self._tail_k[layer_idx] = combined_k
            self._tail_v[layer_idx] = combined_v

        # Update legacy cache and return
        self._update_legacy_cache(layer_idx)
        self._seq_offsets[layer_idx] = seq_offset + new_len

        return return_k, return_v

    def _decode_update(
        self,
        layer_idx: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        seq_offset: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        三区域架构 decode 更新逻辑

        流程：
        1. 新token → Tail
        2. 如果Tail满 → 驱逐Tail开头tokens → 加入竞争队列
        3. 处理竞争队列 → top-K → HeavyHitter HBM分区
        4. 如果HeavyHitter满 → 驱逐低分数tokens → DRAM
        5. 返回 Sink + Tail + HeavyHitter
        """
        # 初始化三个分区（如果需要）
        while len(self._sink_k) <= layer_idx:
            self._sink_k.append(None)
            self._sink_v.append(None)
            self._tail_k.append(None)
            self._tail_v.append(None)
            self._heavyhitter_k.append(None)
            self._heavyhitter_v.append(None)
            self._heavyhitter_scores.append(None)
            self._key_cache.append(None)
            self._value_cache.append(None)
            self._seq_offsets.append(0)

        # Step 1: 新token添加到Tail
        tail_budget = self.hbm_budget_tokens - self.sink_tokens

        if self._tail_k[layer_idx] is None:
            # 第一次写入：初始化Tail
            self._tail_k[layer_idx] = key_states.clone()
            self._tail_v[layer_idx] = value_states.clone()
        else:
            tail_len = self._tail_k[layer_idx].shape[-2]

            if tail_len < tail_budget:
                # Tail未满：直接添加
                self._tail_k[layer_idx] = torch.cat([self._tail_k[layer_idx], key_states], dim=-2)
                self._tail_v[layer_idx] = torch.cat([self._tail_v[layer_idx], value_states], dim=-2)
            else:
                # ═══════════════════════════════════════════════════════════
                # Tail满：驱逐Tail开头tokens → 竞争队列
                # ═══════════════════════════════════════════════════════════
                evicted_k = self._tail_k[layer_idx][:, :, :1, :]
                evicted_v = self._tail_v[layer_idx][:, :, :1, :]

                # 获取驱逐tokens的注意力分数
                if self._oracle.token_scores is not None:
                    current_len = self._get_current_seq_length()
                    evicted_score = self._oracle.token_scores[current_len - 1:current_len]
                else:
                    evicted_score = torch.tensor([1.0], device=self.device)

                # 压缩并加入竞争队列
                if self.enable_quant:
                    k_data, k_scales, k_zps = self._compressor.compress(evicted_k)
                    v_data, v_scales, v_zps = self._compressor.compress(evicted_v)
                    self._competition_queue.enqueue(
                        k=evicted_k, v=evicted_v, scores=evicted_score,
                        compressed={'k_data': k_data, 'k_scales': k_scales, 'k_zps': k_zps,
                                'v_data': v_data, 'v_scales': v_scales, 'v_zps': v_zps},
                        layer_idx=layer_idx, prefix=f"tail_evict"
                    )
                else:
                    self._competition_queue.enqueue(
                        k=evicted_k, v=evicted_v, scores=evicted_score,
                        compressed=None, layer_idx=layer_idx, prefix=f"tail_evict"
                    )

                # 滑动Tail窗口：移除开头，添加新token到末尾
                self._tail_k[layer_idx] = torch.cat([self._tail_k[layer_idx][:, :, 1:, :], key_states], dim=-2)
                self._tail_v[layer_idx] = torch.cat([self._tail_v[layer_idx][:, :, 1:, :], value_states], dim=-2)

        # Step 2: 处理竞争队列
        self._process_competition_queue(layer_idx)

        # Step 3: 更新legacy _key_cache (Sink + Tail + HeavyHitter)
        self._update_legacy_cache(layer_idx)
        self._seq_offsets[layer_idx] = seq_offset + 1

        # 返回用于attention的KV (Sink + Tail + HeavyHitter)
        return self._key_cache[layer_idx], self._value_cache[layer_idx]

    def _get_current_seq_length(self) -> int:
        """获取当前序列长度（用于oracle分数索引）"""
        total = 0
        for layer in range(self.num_layers):
            if self._sink_k[layer] is not None:
                total += self._sink_k[layer].shape[-2]
            if self._tail_k[layer] is not None:
                total += self._tail_k[layer].shape[-2]
        return total

    def _process_competition_queue(self, layer_idx: int):
        """
        处理注意力竞争队列

        逻辑：
        1. 从队列取top-K tokens
        2. 尝试加入HeavyHitter HBM分区
        3. 如果超过预算，驱逐低分数tokens
        """
        # 计算HeavyHitter当前占用
        current_hh_len = 0
        if self._heavyhitter_k[layer_idx] is not None:
            current_hh_len = self._heavyhitter_k[layer_idx].shape[-2]

        available_budget = self._heavyhitter_budget - current_hh_len

        if available_budget > 0:
            # 从竞争队列取top-K tokens
            top_k, top_v, top_scores = self._competition_queue.dequeue_top_k(available_budget)

            if top_k is not None:
                # 加入HeavyHitter分区
                if self._heavyhitter_k[layer_idx] is None:
                    self._heavyhitter_k[layer_idx] = top_k
                    self._heavyhitter_v[layer_idx] = top_v
                    self._heavyhitter_scores[layer_idx] = top_scores
                else:
                    self._heavyhitter_k[layer_idx] = torch.cat([self._heavyhitter_k[layer_idx], top_k], dim=-2)
                    self._heavyhitter_v[layer_idx] = torch.cat([self._heavyhitter_v[layer_idx], top_v], dim=-2)
                    self._heavyhitter_scores[layer_idx] = torch.cat([self._heavyhitter_scores[layer_idx], top_scores], dim=-1)

        # 如果HeavyHitter仍超过预算，驱逐低分数tokens到DRAM
        if self._heavyhitter_k[layer_idx] is not None:
            hh_len = self._heavyhitter_k[layer_idx].shape[-2]

            if hh_len > self._heavyhitter_budget:
                # 驱除多余的tokens
                num_evict = hh_len - self._heavyhitter_budget

                # 根据分数排序，驱逐最低分的tokens
                if self._heavyhitter_scores[layer_idx] is not None:
                    _, low_score_indices = torch.topk(
                        self._heavyhitter_scores[layer_idx],
                        k=num_evict,
                        largest=False
                    )
                else:
                    low_score_indices = torch.arange(num_evict, device=self.device)

                evicted_k = self._heavyhitter_k[layer_idx][:, low_score_indices, :]
                evicted_v = self._heavyhitter_v[layer_idx][:, low_score_indices, :]

                # 压缩并驱逐到DRAM
                if self.enable_quant:
                    k_data, k_scales, k_zps = self._compressor.compress(evicted_k)
                    v_data, v_scales, v_zps = self._compressor.compress(evicted_v)
                    self._dram.store(f"hh_evict_{layer_idx}_{torch.tensor([0])}", k_data, k_scales, k_zps, v_data, v_scales, v_zps)

                # 保留剩余的高分数tokens
                keep_mask = torch.ones(hh_len, dtype=torch.bool, device=self.device)
                keep_mask[low_score_indices] = False

                self._heavyhitter_k[layer_idx] = self._heavyhitter_k[layer_idx][:, keep_mask, :]
                self._heavyhitter_v[layer_idx] = self._heavyhitter_v[layer_idx][:, keep_mask, :]
                self._heavyhitter_scores[layer_idx] = self._heavyhitter_scores[layer_idx][keep_mask]

    def _update_legacy_cache(self, layer_idx: int):
        """更新legacy _key_cache 以保持兼容性"""
        # 用sink的shape作为参考，构造正确维度的空张量
        ref = self._sink_k[layer_idx]
        if ref is not None:
            s = ref.shape
            empty_k = torch.empty(s[0], s[1], 0, s[3], device=self.device, dtype=ref.dtype)
            empty_v = torch.empty(s[0], s[1], 0, s[3], device=self.device, dtype=ref.dtype)
        elif self._tail_k[layer_idx] is not None:
            s = self._tail_k[layer_idx].shape
            empty_k = torch.empty(s[0], s[1], 0, s[3], device=self.device, dtype=self._tail_k[layer_idx].dtype)
            empty_v = torch.empty(s[0], s[1], 0, s[3], device=self.device, dtype=self._tail_k[layer_idx].dtype)
        else:
            return

        sink_k = self._sink_k[layer_idx] if self._sink_k[layer_idx] is not None else empty_k
        tail_k = self._tail_k[layer_idx] if self._tail_k[layer_idx] is not None else empty_k
        hh_k = self._heavyhitter_k[layer_idx] if self._heavyhitter_k[layer_idx] is not None else empty_k

        self._key_cache[layer_idx] = torch.cat([sink_k, tail_k, hh_k], dim=-2)

        sink_v = self._sink_v[layer_idx] if self._sink_v[layer_idx] is not None else empty_v
        tail_v = self._tail_v[layer_idx] if self._tail_v[layer_idx] is not None else empty_v
        hh_v = self._heavyhitter_v[layer_idx] if self._heavyhitter_v[layer_idx] is not None else empty_v

        self._value_cache[layer_idx] = torch.cat([sink_v, tail_v, hh_v], dim=-2)

    def decompress_dram_chunks(
        self, layer_idx: int
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], int]:
        """
        Self-healing: decompress ALL DRAM chunks for a specific layer to BF16.
        Non-destructive — entries remain in DRAM for future use.
        Only decompresses chunks whose key matches the given layer_idx.
        """
        prefix = f"l{layer_idx}_"
        dram_keys = [k for k in self._dram.table.keys() if k.startswith(prefix)]
        if not dram_keys:
            return None, None, 0

        dram_k_parts, dram_v_parts = [], []
        for chunk_key in dram_keys:
            entry = self._dram.retrieve(chunk_key)
            if entry is None:
                continue

            try:
                q_k = entry["k_data"].to(self.device, non_blocking=True)
                s_k = entry["k_scales"].to(self.device, non_blocking=True)
                z_k = entry["k_zps"].to(self.device, non_blocking=True)
                q_v = entry["v_data"].to(self.device, non_blocking=True)
                s_v = entry["v_scales"].to(self.device, non_blocking=True)
                z_v = entry["v_zps"].to(self.device, non_blocking=True)

                restored_k = self._compressor.decompress(q_k, s_k, z_k, target_dtype=torch.float16)
                restored_v = self._compressor.decompress(q_v, s_v, z_v, target_dtype=torch.float16)

                dram_k_parts.append(restored_k)
                dram_v_parts.append(restored_v)
            except Exception:
                continue

        if not dram_k_parts:
            return None, None, 0

        dram_k = torch.cat(dram_k_parts, dim=-2)
        dram_v = torch.cat(dram_v_parts, dim=-2)
        token_count = dram_k.shape[-2]

        return dram_k, dram_v, token_count

    def decompress_dram_chunks_adaptive(
        self,
        layer_idx: int,
        window_size: Optional[int] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], int]:
        """
        TRUE adaptive self-healing: decompress only TOP-W DRAM chunks based on attention scores.

        This implements the dynamic window w_t described in the paper:
          w_t = w_min + (σ_t / σ_ref - 1) · α + β · miss_rate_t

        Args:
            layer_idx: Transformer layer index
            window_size: Number of chunks to retrieve. If None, uses AdaptivePrefetchController.

        Returns:
            (dram_k, dram_v, retrieved_count) or (None, None, 0) if no DRAM data

        Memory Impact:
          - Retrieves only top-w_t chunks (NOT all chunks)
          - HBM spike = O(w_t * chunk_size), NOT O(total_chunks)
          - This is the BOUNDED O(w_t) transient promised in the paper

        Recall Impact:
          - If needle token is outside top-w_t chunks: RETRIEVAL FAILURE
          - Expected NIAH recall: roughly w_t / total_chunks (NOT 100%)
        """
        prefix = f"l{layer_idx}_"
        dram_keys = [k for k in self._dram.table.keys() if k.startswith(prefix)]

        if not dram_keys:
            return None, None, 0

        # Compute adaptive window size using AdaptivePrefetchController
        if window_size is None:
            # Reuse the same controller for self-healing window sizing
            w_t = int(self._adaptive_controller.compute_window(attention_weights=self._last_attention_weights, cache_miss=False))
            window_size = max(1, min(w_t, len(dram_keys)))

        # Rank chunks by attention score (heavy hitters first)
        ranked_chunks = sorted(
            dram_keys,
            key=lambda k: self._chunk_attention_scores.get(k, 0.0),
            reverse=True  # Highest scores first
        )

        # Select only top-w_t chunks
        selected_keys = ranked_chunks[:window_size]

        print(
            f"  [Adaptive Self-Healing] layer={layer_idx} | "
            f"total_chunks={len(dram_keys)} window={window_size} | "
            f"retrieving={len(selected_keys)} chunks | "
            f"coverage={100.0*len(selected_keys)/len(dram_keys):.1f}%"
        )

        # Decompress selected chunks
        dram_k_parts, dram_v_parts = [], []
        for chunk_key in selected_keys:
            entry = self._dram.retrieve(chunk_key)
            if entry is None:
                continue

            try:
                q_k = entry["k_data"].to(self.device, non_blocking=True)
                s_k = entry["k_scales"].to(self.device, non_blocking=True)
                z_k = entry["k_zps"].to(self.device, non_blocking=True)
                q_v = entry["v_data"].to(self.device, non_blocking=True)
                s_v = entry["v_scales"].to(self.device, non_blocking=True)
                z_v = entry["v_zps"].to(self.device, non_blocking=True)

                restored_k = self._compressor.decompress(q_k, s_k, z_k, target_dtype=torch.bfloat16)
                restored_v = self._compressor.decompress(q_v, s_v, z_v, target_dtype=torch.bfloat16)

                dram_k_parts.append(restored_k)
                dram_v_parts.append(restored_v)
            except Exception as e:
                print(f"  [Warning] Failed to decompress {chunk_key}: {e}")
                continue

        if not dram_k_parts:
            return None, None, 0

        # Concatenate in original eviction order (preserve temporal sequence)
        selected_keys_sorted = sorted(selected_keys, key=lambda k: self._chunk_eviction_order.index(k))

        # Re-sort parts to match temporal order
        key_to_part_k = dict(zip(selected_keys, dram_k_parts))
        key_to_part_v = dict(zip(selected_keys, dram_v_parts))

        dram_k_parts_sorted = [key_to_part_k[k] for k in selected_keys_sorted if k in key_to_part_k]
        dram_v_parts_sorted = [key_to_part_v[k] for k in selected_keys_sorted if k in key_to_part_v]

        dram_k = torch.cat(dram_k_parts_sorted, dim=-2)
        dram_v = torch.cat(dram_v_parts_sorted, dim=-2)
        token_count = dram_k.shape[-2]

        # Calculate HBM memory spike
        spike_mb = dram_k.element_size() * dram_k.nelement() / 1024 / 1024
        spike_mb += dram_v.element_size() * dram_v.nelement() / 1024 / 1024

        print(
            f"  [Adaptive Self-Healing] Retrieved {token_count} tokens | "
            f"HBM_spike={spike_mb:.1f}MB (vs {len(dram_keys)*2048*16/1024/1024:.1f}MB full)"
        )

        return dram_k, dram_v, token_count

    def get_dram_chunks_quantized_adaptive(
        self,
        layer_idx: int,
        window_size: Optional[int] = None,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        TRUE dynamic window + Triton kernel: Return 4-bit quantized chunks (no decompression).

        This is the CORRECT implementation where dynamic window and Triton kernel
        work TOGETHER:
          1. AdaptivePrefetchController computes w_t based on σ(A_t)
          2. Select top-w_t chunks by attention score
          3. Return 4-bit quantized data (NO BF16 decompression)
          4. Triton fused kernel computes attention directly on 4-bit data

        Memory Impact:
          - HBM spike: O(w_t * chunk_size) for 4-bit data transfer
          - NO BF16 decompression spike (eliminated!)
          - This is the paper's "eliminates 512MB transient" promise

        Args:
            layer_idx: Transformer layer index
            window_size: Number of chunks to retrieve. If None, uses adaptive controller.

        Returns:
            Dict with k_data, k_scales, k_zps, v_data, v_scales, v_zps (all 4-bit)
            Or None if no DRAM data
        """
        prefix = f"l{layer_idx}_"
        dram_keys = [k for k in self._dram.table.keys() if k.startswith(prefix)]

        if not dram_keys:
            return None

        # Compute adaptive window size
        if window_size is None:
            w_t = int(self._adaptive_controller.compute_window(attention_weights=self._last_attention_weights, cache_miss=False))
            window_size = max(1, min(w_t, len(dram_keys)))

        # Rank chunks by attention score (heavy hitters first)
        ranked_chunks = sorted(
            dram_keys,
            key=lambda k: self._chunk_attention_scores.get(k, 0.0),
            reverse=True
        )

        # Select only top-w_t chunks
        selected_keys = ranked_chunks[:window_size]

        print(
            f"  [Triton-Optimized Adaptive Self-Healing] layer={layer_idx} | "
            f"total_chunks={len(dram_keys)} window={window_size} | "
            f"retrieving={len(selected_keys)} chunks (4-bit, NO decompression) | "
            f"coverage={100.0*len(selected_keys)/len(dram_keys):.1f}%"
        )

        # Collect 4-bit quantized data (NO decompression!)
        all_k_data, all_k_scales, all_k_zps = [], [], []
        all_v_data, all_v_scales, all_v_zps = [], [], []

        for chunk_key in selected_keys:
            entry = self._dram.retrieve(chunk_key)
            if entry is None:
                continue

            # Transfer to GPU but KEEP 4-bit format!
            all_k_data.append(entry["k_data"].to(self.device, non_blocking=True))
            all_k_scales.append(entry["k_scales"].to(self.device, non_blocking=True))
            all_k_zps.append(entry["k_zps"].to(self.device, non_blocking=True))
            all_v_data.append(entry["v_data"].to(self.device, non_blocking=True))
            all_v_scales.append(entry["v_scales"].to(self.device, non_blocking=True))
            all_v_zps.append(entry["v_zps"].to(self.device, non_blocking=True))

        if not all_k_data:
            return None

        # Concatenate in original eviction order
        selected_keys_sorted = sorted(selected_keys, key=lambda k: self._chunk_eviction_order.index(k))
        key_to_idx = {k: i for i, k in enumerate(dram_keys)}
        sorted_indices = [key_to_idx[k] for k in selected_keys_sorted if k in key_to_idx]

        # Handle concatenation based on tensor dimensions
        # For data tensors (3D: [batch, heads, seq] or 4D): concatenate along dim=-2 (sequence dim)
        k_data = torch.cat([all_k_data[i] for i in sorted_indices], dim=-2)
        v_data = torch.cat([all_v_data[i] for i in sorted_indices], dim=-2)

        # For scales/zp tensors: determine dimension based on shape
        if all_k_scales[0].ndim >= 2:
            # 2D+ tensors: concatenate along sequence dimension (dim=-2)
            k_scales = torch.cat([all_k_scales[i] for i in sorted_indices], dim=-2)
            k_zps = torch.cat([all_k_zps[i] for i in sorted_indices], dim=-2)
            v_scales = torch.cat([all_v_scales[i] for i in sorted_indices], dim=-2)
            v_zps = torch.cat([all_v_zps[i] for i in sorted_indices], dim=-2)
        else:
            # 1D tensors: concatenate along dim=0 (the only dimension)
            k_scales = torch.cat([all_k_scales[i] for i in sorted_indices], dim=0)
            k_zps = torch.cat([all_k_zps[i] for i in sorted_indices], dim=0)
            v_scales = torch.cat([all_v_scales[i] for i in sorted_indices], dim=0)
            v_zps = torch.cat([all_v_zps[i] for i in sorted_indices], dim=0)

        # Calculate memory savings
        tokens_4bit = k_data.shape[-2]
        bf16_spike_mb = tokens_4bit * 2 * 32 * 2 / 1024 / 1024  # Would be BF16 size
        bit4_mb = (
            k_data.element_size() * k_data.nelement() +
            k_scales.element_size() * k_scales.nelement() +
            k_zps.element_size() * k_zps.nelement() +
            v_data.element_size() * v_data.nelement() +
            v_scales.element_size() * v_scales.nelement() +
            v_zps.element_size() * v_zps.nelement()
        ) / 1024 / 1024

        print(
            f"  [Triton-Optimized] {tokens_4bit} tokens in 4-bit | "
            f"HMB={bit4_mb:.1f}MB (vs {bf16_spike_mb:.1f}MB if BF16) | "
            f"saved={bf16_spike_mb - bit4_mb:.1f}MB"
        )

        return {
            "k_data": k_data,
            "k_scales": k_scales,
            "k_zps": k_zps,
            "v_data": v_data,
            "v_scales": v_scales,
            "v_zps": v_zps,
        }

    def _get_dram_kv_quantized(
        self, layer_idx: int
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        Get DRAM KV data in 4-bit quantized format (no decompression).

        Used by Triton fused kernel to avoid BF16 intermediate allocation.

        Returns:
            Dict with keys: k_data, k_scales, k_zps, v_data, v_scales, v_zps
            Or None if no DRAM data
        """
        prefix = f"l{layer_idx}_"
        dram_keys = [k for k in self._dram.table.keys() if k.startswith(prefix)]

        if not dram_keys:
            return None

        all_k_data, all_k_scales, all_k_zps = [], [], []
        all_v_data, all_v_scales, all_v_zps = [], [], []

        for chunk_key in dram_keys:
            entry = self._dram.retrieve(chunk_key)
            if entry is None:
                continue

            all_k_data.append(entry["k_data"])
            all_k_scales.append(entry["k_scales"])
            all_k_zps.append(entry["k_zps"])
            all_v_data.append(entry["v_data"])
            all_v_scales.append(entry["v_scales"])
            all_v_zps.append(entry["v_zps"])

        if not all_k_data:
            return None

        # Concatenate in eviction order
        sorted_keys = sorted(dram_keys, key=lambda k: self._chunk_eviction_order.index(k))

        key_to_idx = {k: i for i, k in enumerate(dram_keys)}
        sorted_indices = [key_to_idx[k] for k in sorted_keys if k in key_to_idx]

        k_data = torch.cat([all_k_data[i] for i in sorted_indices], dim=-2)
        k_scales = torch.cat([all_k_scales[i] for i in sorted_indices], dim=-2)
        k_zps = torch.cat([all_k_zps[i] for i in sorted_indices], dim=-2)
        v_data = torch.cat([all_v_data[i] for i in sorted_indices], dim=-2)
        v_scales = torch.cat([all_v_scales[i] for i in sorted_indices], dim=-2)
        v_zps = torch.cat([all_v_zps[i] for i in sorted_indices], dim=-2)

        return {
            "k_data": k_data,
            "k_scales": k_scales,
            "k_zps": k_zps,
            "v_data": v_data,
            "v_scales": v_scales,
            "v_zps": v_zps,
        }

    def count_dram_tokens(self, layer_idx: int = 0) -> int:
        """Count total tokens in DRAM for a specific layer."""
        prefix = f"l{layer_idx}_"
        dram_keys = [k for k in self._dram.table.keys() if k.startswith(prefix)]
        if not dram_keys:
            return 0
        entry = self._dram.retrieve(dram_keys[0])
        if entry is None:
            return 0
        k = entry["k_data"]
        tokens_per_chunk = k.shape[-2] if k.dim() >= 2 else 1
        return len(dram_keys) * tokens_per_chunk

    def _evict_to_dram(
        self,
        layer_idx: int,
        k_chunk: torch.Tensor,
        v_chunk: torch.Tensor,
    ) -> None:
        """
        Compress a KV chunk and move it to DRAM via DRAMStorageManager (pinned CPU memory).

        Enhanced: tracks chunk metadata for adaptive self-healing (attention scores, eviction order).
        """
        chunk_key = f"l{layer_idx}_e{self._eviction_counter}"

        q_k, k_scales, k_zps = self._compressor.compress(k_chunk)
        q_v, v_scales, v_zps = self._compressor.compress(v_chunk)

        entry = {
            "k_data": q_k,
            "k_scales": k_scales,
            "k_zps": k_zps,
            "v_data": q_v,
            "v_scales": v_scales,
            "v_zps": v_zps,
        }

        # DRAMStorageManager handles .cpu().pin_memory() internally
        self._dram.store_entry(chunk_key, entry)

        # Track metadata for adaptive self-healing
        self._chunk_eviction_order.append(chunk_key)

        # Compute and store chunk attention score (average of oracle scores for tokens in this chunk)
        if self._oracle.token_scores is not None:
            start_token_idx = self._eviction_counter * k_chunk.shape[-2]
            end_token_idx = start_token_idx + k_chunk.shape[-2]
            chunk_scores = self._oracle.token_scores[start_token_idx:end_token_idx]
            chunk_avg_score = float(chunk_scores.mean().item()) if len(chunk_scores) > 0 else 0.0
            self._chunk_attention_scores[chunk_key] = chunk_avg_score
        else:
            # Fallback: use FIFO order as score (older chunks = lower score)
            self._chunk_attention_scores[chunk_key] = 1.0 / (self._eviction_counter + 1)

        if self._bandwidth_limiter is not None:
            stored = self._dram.retrieve(chunk_key)
            if stored is not None:
                for t in stored.values():
                    self._bandwidth_limiter.simulate_transfer(t)

        if layer_idx == 0:
            tokens = k_chunk.shape[-2]
            self._eviction_counter += 1
            print(
                f"  [Evict->DRAM] layer=0 chunk={chunk_key} "
                f"tokens={tokens} score={self._chunk_attention_scores[chunk_key]:.4f} "
                f"DRAM_entries={self._dram.num_entries}"
            )


if __name__ == "__main__":
    manager = HeteroKVManager(num_layers=4, sink_tokens=64, hbm_budget_tokens=256)
    for layer in range(4):
        manager.allocate(layer, budget=512)

    # Prefill simulation
    k = torch.randn(1, 8, 512, 128, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(1, 8, 512, 128, dtype=torch.bfloat16, device="cuda")
    out_k, out_v = manager.update(0, k, v, mode="prefill")
    print("Prefill output shape:", out_k.shape, out_v.shape)
    print("HBM resident shape:", manager.get_hbm_kv(0)[0].shape)
    print("Memory summary:", manager.memory_summary())

    # Decode simulation
    k1 = torch.randn(1, 8, 1, 128, dtype=torch.bfloat16, device="cuda")
    v1 = torch.randn(1, 8, 1, 128, dtype=torch.bfloat16, device="cuda")
    out_k, out_v = manager.update(0, k1, v1, mode="decode")
    print("Decode output shape:", out_k.shape)
