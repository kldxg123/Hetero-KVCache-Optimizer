"""
src/memory/manager_three_zone_fixed.py
======================================
正确的三区域HeteroKVManager实现

按照用户设计理念：
1. Sink: 64 tokens (系统提示，固定HBM)
2. Tail: 2048 tokens (最近上下文，固定HBM)
3. HeavyHitter: 动态HBM分区 (高注意力tokens)

关键设计：
- Tail驱逐的tokens → HeavyHitter竞争队列 (非直接DRAM)
- 动态窗口取回 → 寄存器解压计算注意力 → 加入竞争队列
- 竞争队列中的tokens → 基于注意力分数 → HBM或DRAM
- HeavyHitter分区满 → 驱逐低分数tokens → DRAM

内存预算：Sink(64) + Tail(2048) + HeavyHitter(budget) = O(1) HBM占用
"""

import torch
from typing import Dict, List, Optional, Tuple
import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.quantization.kv_compressor import KVCompressor
from src.memory.dram_storage import DRAMStorageManager
from src.policy.heavy_hitter import HeavyHitterOracle


class AttentionScoreQueue:
    """
    注意力分数竞争队列

    管理"待定"HBM空间的tokens：
    - Tail驱逐的tokens (带注意力分数)
    - 动态窗口取回的tokens (寄存器计算后带分数)

    基于注意力分数决定：
    - 高分数 → 进入HeavyHitter HBM分区
    - 低分数 → 驱逐到DRAM
    """

    def __init__(self):
        self.queue: Dict[str, Dict] = {}  # {token_id: {k, v, score, layer}}

    def enqueue(self, token_ids: List[str], k_chunks: List[torch.Tensor],
               v_chunks: List[torch.Tensor], scores: List[torch.Tensor], layer: int):
        """添加tokens到竞争队列"""
        for i, token_id in enumerate(token_ids):
            self.queue[token_id] = {
                'k': k_chunks[i],
                'v': v_chunks[i],
                'score': scores[i],
                'layer': layer,
            }

    def dequeue_top_k(self, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """取出top-k高分数的tokens"""
        if not self.queue:
            return None, None

        # 按分数排序
        sorted_tokens = sorted(self.queue.items(), key=lambda x: x[1]['score'], reverse=True)

        # 取top-k
        top_k = sorted_tokens[:k]
        if not top_k:
            return None, None

        # 提取K和V
        k_list = [item[1]['k'] for item in top_k]
        v_list = [item[1]['v'] for item in top_k]

        k_cat = torch.cat(k_list, dim=-2)
        v_cat = torch.cat(v_list, dim=-2)

        # 移除已处理的tokens
        for token_id, _ in top_k:
            del self.queue[token_id]

        return k_cat, v_cat

    def get_low_score_tokens(self, threshold: float) -> List[str]:
        """获取低于阈值的tokens（用于驱逐）"""
        return [tid for tid, item in self.queue.items() if item['score'] < threshold]

    def evict_to_dram(self, token_ids: List[str], compressor, dram_storage):
        """将低分数tokens压缩并驱逐到DRAM"""
        for token_id in token_ids:
            if token_id in self.queue:
                item = self.queue[token_id]

                # 压缩
                k_4bit, v_4bit = compressor.compress(
                    item['k'].unsqueeze(-2),
                    item['v'].unsqueeze(-2)
                )

                # 存储到DRAM
                dram_storage.store(token_id, k_4bit, v_4bit)

                # 从队列移除
                del self.queue[token_id]


class HeteroKVManagerThreeZoneFixed:
    """
    三区域HeteroKV管理器 - 正确版本

    HBM分区 (O(1)内存)：
    - Sink: 64 tokens (固定)
    - Tail: 2048 tokens (固定)
    - HeavyHitter: budget tokens (动态，高注意力)

    DRAM分区 (溢出存储)：
    - 4-bit压缩chunks

    数据流：
    1. 新tokens → Tail
    2. Tail满 → 驱除tokens → AttentionScoreQueue
    3. 动态窗口取回 → 寄存器解压计算 → AttentionScoreQueue
    4. AttentionScoreQueue → top-K → HeavyHitter HBM
    5. AttentionScoreQueue → low-K → DRAM
    6. HeavyHitter满 → 驱逐低分数 → DRAM
    """

    def __init__(
        self,
        num_layers: int,
        sink_tokens: int = 64,
        tail_tokens: int = 2048,
        heavyhitter_budget: int = 4096,
        device: str = "cuda",
        enable_quant: bool = True,
        group_size: int = 128,
        bits: int = 4,
    ):
        self.num_layers = num_layers
        self.sink_tokens = sink_tokens
        self.tail_tokens = tail_tokens
        self.heavyhitter_budget = heavyhitter_budget
        self.device = device
        self.enable_quant = enable_quant
        self.group_size = group_size
        self.bits = bits

        # 三个HBM分区
        self._sink_kv: List[Optional[Dict]] = [None] * num_layers
        self._tail_kv: List[Optional[Dict]] = [None] * num_layers
        self._heavyhitter_kv: List[Optional[Dict]] = [None] * num_layers

        # 压缩器
        self._compressor = KVCompressor(group_size=group_size, bits=bits)

        # 注意力竞争队列
        self._attention_queue = AttentionScoreQueue()

        # HeavyHitterOracle (驱逐决策)
        self._oracle = HeavyHitterOracle(
            block_size=16,
            sink_tokens=sink_tokens,
            local_window=tail_tokens,
        )

        # DRAM存储
        self._dram = DRAMStorageManager()

        # 序列追踪
        self._seq_offsets: List[int] = [0] * num_layers

        print(f"[HeteroKVThreeZone] Sink={sink_tokens} Tail={tail_tokens} "
              f"HeavyHitter={heavyhitter_budget} | Total={sink_tokens+tail_tokens+heavyhitter_budget}")

    def update(
        self,
        layer_idx: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        mode: str = "decode",
        seq_offset: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """更新KV cache并返回用于attention的KV"""

        if mode == "prefill":
            return self._prefill_update(layer_idx, key_states, value_states, seq_offset)
        elif mode == "decode":
            return self._decode_update(layer_idx, key_states, value_states, seq_offset)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def _prefill_update(
        self,
        layer_idx: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        seq_offset: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prefill阶段处理

        逻辑：
        1. 前64个tokens → Sink
        2. 最后2048个tokens → Tail
        3. 中间tokens → 压缩到DRAM，加入竞争队列
        """
        new_len = key_states.shape[-2]

        # 初始化分区
        if self._sink_kv[layer_idx] is None:
            self._sink_kv[layer_idx] = {'k': torch.empty_like(key_states[:, :0]), 'v': torch.empty_like(value_states[:, :0])}
            self._tail_kv[layer_idx] = {'k': torch.empty_like(key_states[:, :0]), 'v': torch.empty_like(value_states[:, :0])}

        # 分配tokens到三个分区
        if new_len <= self.sink_tokens:
            # 全部去Sink
            self._sink_kv[layer_idx]['k'] = key_states
            self._sink_kv[layer_idx]['v'] = value_states
        elif new_len <= self.sink_tokens + self.tail_tokens:
            # Sink + Tail
            sink_k = key_states[..., :self.sink_tokens, :]
            sink_v = value_states[..., :self.sink_tokens, :]
            tail_k = key_states[..., self.sink_tokens:, :]
            tail_v = value_states[..., self.sink_tokens:, :]

            self._sink_kv[layer_idx]['k'] = sink_k
            self._sink_kv[layer_idx]['v'] = sink_v
            self._tail_kv[layer_idx]['k'] = tail_k
            self._tail_kv[layer_idx]['v'] = tail_v
        else:
            # Sink + Tail + Body (body压缩到DRAM)
            sink_k = key_states[..., :self.sink_tokens, :]
            sink_v = value_states[..., :self.sink_tokens, :]

            tail_k = key_states[..., -self.tail_tokens:, :]
            tail_v = value_states[..., -self.tail_tokens:, :]

            body_k = key_states[..., self.sink_tokens:-self.tail_tokens, :]
            body_v = value_states[..., self.sink_tokens:-self.tail_tokens, :]

            # 压缩body到DRAM
            if self.enable_quant and body_k.shape[-2] > 0:
                k_4bit, v_4bit = self._compressor.compress(body_k, body_v)
                self._dram.store(f"prefill_body_{layer_idx}", k_4bit, v_4bit)

                # 将body加入竞争队列（初始分数为平均分配）
                num_body_tokens = body_k.shape[-2]
                scores = torch.ones(num_body_tokens, dtype=torch.float32, device=self.device) / num_body_tokens

                # Token IDs
                token_ids = [f"prefill_body_{layer_idx}_{i}" for i in range(num_body_tokens)]

                # 添加到竞争队列（使用4-bit数据）
                self._attention_queue.enqueue(
                    token_ids,
                    [k_4bit['data']],  # 注意：这里使用4-bit数据
                    [v_4bit['data']],
                    [scores],
                    layer_idx
                )

            # 更新Sink和Tail
            self._sink_kv[layer_idx]['k'] = sink_k
            self._sink_kv[layer_idx]['v'] = sink_v
            self._tail_kv[layer_idx]['k'] = tail_k
            self._tail_kv[layer_idx]['v'] = tail_v

        # 管理竞争队列
        self._process_attention_queue(layer_idx)

        # 返回完整序列（FlashAttention兼容）
        return self._get_full_sequence(layer_idx)

    def _decode_update(
        self,
        layer_idx: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        seq_offset: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode阶段处理

        逻辑：
        1. 新token → Tail末尾
        2. 如果Tail满 → 驱除Tail开头 → 加入竞争队列
        3. 处理竞争队列 → top-K → HeavyHitter
        4. 返回 Sink + Tail + HeavyHitter
        """
        assert key_states.shape[-2] == 1, "Decode mode expects single token"

        # 初始化
        if self._tail_kv[layer_idx] is None:
            self._tail_kv[layer_idx] = {'k': key_states, 'v': value_states}
        else:
            tail_len = self._tail_kv[layer_idx]['k'].shape[-2]

            if tail_len >= self.tail_tokens:
                # Tail满：驱逐一个token
                evicted_k = self._tail_kv[layer_idx]['k'][:, :1, :]
                evicted_v = self._tail_kv[layer_idx]['v'][:, :1, :]

                # 获取驱逐token的注意力分数
                evicted_score = self._get_token_score(layer_idx, 1)

                # 压缩并加入竞争队列
                if self.enable_quant:
                    k_4bit, v_4bit = self._compressor.compress(evicted_k, evicted_v)

                    # 添加到竞争队列（带4-bit数据和分数）
                    token_id = f"decode_evict_{layer_idx}_{seq_offset}"
                    self._attention_queue.enqueue(
                        [token_id],
                        [k_4bit['data']],
                        [v_4bit['data']],
                        [evicted_score],
                        layer_idx
                    )

                # 滑动Tail窗口
                self._tail_kv[layer_idx]['k'] = torch.cat([
                    self._tail_kv[layer_idx]['k'][:, 1:, :],
                    key_states
                ], dim=-2)
                self._tail_kv[layer_idx]['v'] = torch.cat([
                    self._tail_kv[layer_idx]['v'][:, 1:, :],
                    value_states
                ], dim=-2)
            else:
                # Tail未满，直接添加
                self._tail_kv[layer_idx]['k'] = torch.cat([self._tail_kv[layer_idx]['k'], key_states], dim=-2)
                self._tail_kv[layer_idx]['v'] = torch.cat([self._tail_kv[layer_idx]['v'], value_states], dim=-2)

        # 处理竞争队列（将高分数tokens移入HeavyHitter）
        self._process_attention_queue(layer_idx)

        # 返回用于attention的KV
        return self._get_attention_kv(layer_idx)

    def _get_token_score(self, layer_idx: int, num_tokens: int) -> torch.Tensor:
        """获取tokens的注意力分数"""
        if self._oracle.token_scores is None:
            return torch.ones(num_tokens, dtype=torch.float32, device=self.device) / num_tokens

        current_len = self._oracle.token_scores.shape[0]
        return self._oracle.token_scores[-num_tokens:]

    def _process_attention_queue(self, layer_idx: int):
        """
        处理注意力竞争队列

        逻辑：
        1. 从队列取top-K tokens
        2. 如果HeavyHitter未满，直接加入
        3. 如果HeavyHitter已满，替换低分数tokens
        4. 剩余低分数tokens驱逐到DRAM
        """
        # 计算可用预算
        current_hh_len = 0
        if self._heavyhitter_kv[layer_idx] is not None:
            current_hh_len = self._heavyhitter_kv[layer_idx]['k'].shape[-2]

        available_budget = self.heavyhitter_budget - current_hh_len

        if available_budget > 0:
            # 从竞争队列取top-K
            top_k, top_v = self._attention_queue.dequeue_top_k(available_budget)

            if top_k is not None:
                # 添加到HeavyHitter分区
                if self._heavyhitter_kv[layer_idx] is None:
                    self._heavyhitter_kv[layer_idx] = {'k': top_k, 'v': top_v}
                else:
                    self._heavyhitter_kv[layer_idx]['k'] = torch.cat([
                        self._heavyhitter_kv[layer_idx]['k'], top_k
                    ], dim=-2)
                    self._heavyhitter_kv[layer_idx]['v'] = torch.cat([
                        self._heavyhitter_kv[layer_idx]['v'], top_v
                    ], dim=-2)

        # 如果HeavyHitter仍超过预算，驱逐低分数tokens到DRAM
        if self._heavyhitter_kv[layer_idx] is not None:
            hh_len = self._heavyhitter_kv[layer_idx]['k'].shape[-2]

            if hh_len > self.heavyhitter_budget:
                # 驱除多余的tokens
                num_evict = hh_len - self.heavyhitter_budget
                evicted_k = self._heavyhitter_kv[layer_idx]['k'][:, :num_evict, :]
                evicted_v = self._heavyhitter_kv[layer_idx]['v'][:, :num_evict, :]

                # 压缩到DRAM
                if self.enable_quant:
                    k_4bit, v_4bit = self._compressor.compress(evicted_k, evicted_v)
                    self._dram.store(f"hh_evict_{layer_idx}_{torch.tensor([0])}", k_4bit, v_4bit)

                # 保留剩余tokens
                self._heavyhitter_kv[layer_idx]['k'] = self._heavyhitter_kv[layer_idx]['k'][:, num_evict:, :]
                self._heavyhitter_kv[layer_idx]['v'] = self._heavyhitter_kv[layer_idx]['v'][:, num_evict:, :]

    def _get_full_sequence(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取完整序列（prefill用）"""
        sink = self._sink_kv[layer_idx]
        tail = self._tail_kv[layer_idx] if self._tail_kv[layer_idx] is not None else {'k': torch.empty_like(sink['k'][:, :0]), 'v': torch.empty_like(sink['v'][:, :0])}
        hh = self._heavyhitter_kv[layer_idx] if self._heavyhitter_kv[layer_idx] is not None else {'k': torch.empty_like(sink['k'][:, :0]), 'v': torch.empty_like(sink['v'][:, :0])}

        k_full = torch.cat([sink['k'], tail['k'], hh['k']], dim=-2)
        v_full = torch.cat([sink['v'], tail['v'], hh['v']], dim=-2)

        return k_full, v_full

    def _get_attention_kv(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取用于attention的KV（Sink + Tail + HeavyHitter）"""
        sink = self._sink_kv[layer_idx]
        tail = self._tail_kv[layer_idx] if self._tail_kv[layer_idx] is not None else {'k': torch.empty_like(sink['k'][:, :0]), 'v': torch.empty_like(sink['v'][:, :0])}
        hh = self._heavyhitter_kv[layer_idx] if self._heavyhitter_kv[layer_idx] is not None else {'k': torch.empty_like(sink['k'][:, :0]), 'v': torch.empty_like(sink['v'][:, :0])}

        # 拼接三个分区
        k_final = torch.cat([sink['k'], tail['k'], hh['k']], dim=-2)
        v_final = torch.cat([sink['v'], tail['v'], hh['v']], dim=-2)

        return k_final, v_final

    def update_attention_scores(self, attention_weights: torch.Tensor):
        """更新oracle的注意力分数"""
        self._oracle.update(attention_weights)

    def get_dram_chunks_for_register_compute(
        self, layer_idx: int, num_chunks: int
    ) -> Optional[Dict]:
        """
        获取DRAM chunks用于寄存器计算

        关键：返回4-bit数据，不加载到HBM
        Triton kernel将直接从4-bit数据计算注意力
        """
        # 这里应该从DRAM获取4-bit数据
        # 但为了寄存器计算，我们不拼接，直接返回chunks列表
        # Triton kernel会逐chunk处理
        pass

    def get_seq_length(self) -> int:
        """获取序列长度"""
        total = 0
        for layer in range(self.num_layers):
            if self._sink_kv[layer] is not None:
                total += self._sink_kv[layer]['k'].shape[-2]
            if self._tail_kv[layer] is not None:
                total += self._tail_kv[layer]['k'].shape[-2]
            if self._heavyhitter_kv[layer] is not None:
                total += self._heavyhitter_kv[layer]['k'].shape[-2]
        return total

    def memory_summary(self) -> Dict:
        """内存使用摘要"""
        sink_tokens = sum(kv['k'].shape[-2] for kv in self._sink_kv if kv is not None)
        tail_tokens = sum(kv['k'].shape[-2] for kv in self._tail_kv if kv is not None)
        hh_tokens = sum(kv['k'].shape[-2] for kv in self._heavyhitter_kv if kv is not None)

        return {
            'sink_tokens': sink_tokens,
            'tail_tokens': tail_tokens,
            'heavyhitter_tokens': hh_tokens,
            'total_hbm_tokens': sink_tokens + tail_tokens + hh_tokens,
            'target_budget': self.sink_tokens + self.tail_tokens + self.heavyhitter_budget,
        }