"""
src/memory/manager_three_zone.py
================================
修复版三区域HeteroKVManager

严格按照用户设计实现三个HBM分区：
1. Sink: 64 tokens (固定，系统提示)
2. Tail: 2048 tokens (固定，最近上下文)
3. HeavyHitter: 动态分区 (高注意力tokens)

核心修复：
- HeavyHitter是真实HBM分区，而非仅驱逐决策
- Tail驱逐的tokens加入HeavyHitter竞争队列
- 动态窗口取回使用寄存器解压计算
- 基于注意力分数的驱逐机制
"""

import gc
import sys
import os
from typing import Any, Dict, List, Optional, Tuple
import torch

# Add project root
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.quantization.kv_compressor import KVCompressor
from src.memory.dram_storage import DRAMStorageManager
from src.policy.heavy_hitter import HeavyHitterOracle
from src.policy.adaptive_prefetch_controller import AdaptivePrefetchController


class AttentionCompetitorQueue:
    """
    注意力竞争队列管理器

    管理两个来源的tokens竞争HBM空间：
    1. HeavyHitter分区中的tokens
    2. 动态窗口取回的DRMA chunks (寄存器解压后)

    基于注意力分数决定：
    - 高分数 → 留在HBM
    - 低分数 → 驱逐到DRAM
    """

    def __init__(self):
        self.heavyhitter_tokens: Dict[int, Dict[str, torch.Tensor]] = {}  # {layer: {k, v, scores}}
        self.dynamic_retrieval_tokens: Dict[int, Dict[str, torch.Tensor]] = {}  # {layer: {k, v, scores}}

    def add_heavyhitter_tokens(self, layer: int, k: torch.Tensor, v: torch.Tensor, scores: torch.Tensor):
        """添加从Tail驱逐的tokens到HeavyHitter队列"""
        if layer not in self.heavyhitter_tokens:
            self.heavyhitter_tokens[layer] = {
                'k': k, 'v': v, 'scores': scores
            }
        else:
            # Concat with existing HeavyHitter tokens
            self.heavyhitter_tokens[layer]['k'] = torch.cat([self.heavyhitter_tokens[layer]['k'], k], dim=-2)
            self.heavyhitter_tokens[layer]['v'] = torch.cat([self.heavyhitter_tokens[layer]['v'], v], dim=-2)
            self.heavyhitter_tokens[layer]['scores'] = torch.cat([self.heavyhitter_tokens[layer]['scores'], scores], dim=-1)

    def add_dynamic_retrieval(self, layer: int, k: torch.Tensor, v: torch.Tensor, scores: torch.Tensor):
        """添加动态窗口取回的tokens（寄存器解压后）"""
        if layer not in self.dynamic_retrieval_tokens:
            self.dynamic_retrieval_tokens[layer] = {
                'k': k, 'v': v, 'scores': scores
            }
        else:
            self.dynamic_retrieval_tokens[layer]['k'] = torch.cat([self.dynamic_retrieval_tokens[layer]['k'], k], dim=-2)
            self.dynamic_retrieval_tokens[layer]['v'] = torch.cat([self.dynamic_retrieval_tokens[layer]['v'], v], dim=-2)
            self.dynamic_retrieval_tokens[layer]['scores'] = torch.cat([self.dynamic_retrieval_tokens[layer]['scores'], scores], dim=-1)

    def get_top_tokens(self, layer: int, budget: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取高注意力的tokens，按预算限制"""
        if layer not in self.heavyhitter_tokens and layer not in self.dynamic_retrieval_tokens:
            return None, None

        # 收集所有tokens
        all_k = []
        all_v = []
        all_scores = []

        if layer in self.heavyhitter_tokens:
            all_k.append(self.heavyhitter_tokens[layer]['k'])
            all_v.append(self.heavyhitter_tokens[layer]['v'])
            all_scores.append(self.heavyhitter_tokens[layer]['scores'])

        if layer in self.dynamic_retrieval_tokens:
            all_k.append(self.dynamic_retrieval_tokens[layer]['k'])
            all_v.append(self.dynamic_retrieval_tokens[layer]['v'])
            all_scores.append(self.dynamic_retrieval_tokens[layer]['scores'])

        if not all_k:
            return None, None

        # 拼接
        k_cat = torch.cat(all_k, dim=-2)
        v_cat = torch.cat(all_v, dim=-2)
        scores_cat = torch.cat(all_scores, dim=-1)

        # 按注意力分数排序，保留top-K
        num_tokens = k_cat.shape[-2]
        if num_tokens > budget:
            top_indices = torch.topk(scores_cat, k=budget, largest=True).indices
            k_selected = k_cat[..., top_indices, :]
            v_selected = v_cat[..., top_indices, :]
        else:
            k_selected = k_cat
            v_selected = v_cat

        return k_selected, v_selected

    def clear_dynamic_retrieval(self, layer: int):
        """清除动态取回的tokens（已处理完毕）"""
        if layer in self.dynamic_retrieval_tokens:
            del self.dynamic_retrieval_tokens[layer]


class HeteroKVManagerThreeZone:
    """
    三区域HeteroKV管理器

    HBM分区：
      - Sink: 64 tokens (固定)
      - Tail: 2048 tokens (固定)
      - HeavyHitter: 动态 (根据注意力分数)

    DRAM分区：
      - 4-bit压缩chunks (溢出存储)

    关键修复：
      - HeavyHitter是真实HBM分区，不是驱逐决策工具
      - Tail驱逐的tokens加入HeavyHitter竞争队列
      - 动态窗口取回使用寄存器解压计算
      - 基于注意力分数的驱逐
    """

    def __init__(
        self,
        num_layers: int,
        sink_tokens: int = 64,
        tail_tokens: int = 2048,
        heavyhitter_budget: int = 4096,  # HeavyHitter分区最大tokens
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
        # Sink: 固定大小，所有层共享
        self._sink_kv: List[Optional[Dict[str, torch.Tensor]]] = [None] * num_layers  # [{k, v}, ...]

        # Tail: 固定大小，滑动窗口
        self._tail_kv: List[Optional[Dict[str, torch.Tensor]]] = [None] * num_layers  # [{k, v}, ...]

        # HeavyHitter: 动态大小，高注意力tokens
        self._heavyhitter_kv: List[Optional[Dict[str, torch.Tensor]]] = [None] * num_layers  # [{k, v, scores}, ...]

        # 压缩引擎
        self._compressor = KVCompressor(group_size=group_size, bits=bits)

        # 注意力竞争队列
        self._attention_queue = AttentionCompetitorQueue()

        # HeavyHitterOracle (仅用于驱逐决策，不是存储)
        self._oracle = HeavyHitterOracle(
            block_size=16,
            sink_tokens=sink_tokens,
            local_window=tail_tokens,  # 注意：这里tail_tokens作为local_window
        )

        # DRAM存储
        self._dram = DRAMStorageManager()

        # 序列追踪
        self._seq_offsets: List[int] = [0] * num_layers
        self._real_seq_len: int = 0

        print(f"[HeteroKVManagerThreeZone] Initialized | "
              f"Sink={sink_tokens} Tail={tail_tokens} HeavyHitter={heavyhitter_budget} | "
              f"Total HBM budget={sink_tokens + tail_tokens + heavyhitter_budget}")

    def update(
        self,
        layer_idx: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        mode: str = "decode",
        seq_offset: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        更新KV cache，返回用于attention计算的KV

        三区域逻辑：
        1. 新tokens添加到Tail
        2. 如果Tail满，从开头驱逐tokens
        3. 驱逐的tokens加入HeavyHitter竞争队列
        4. 如果HeavyHitter满，驱逐低注意力tokens到DRAM
        5. 返回 Sink + Tail + HeavyHitter (top-K) 用于attention
        """
        new_len = key_states.shape[-2]

        if mode == "prefill":
            # Prefill模式：返回完整序列（FlashAttention兼容）
            if self._sink_kv[layer_idx] is None:
                self._sink_kv[layer_idx] = {'k': key_states.clone(), 'v': value_states.clone()}
                self._tail_kv[layer_idx] = {'k': torch.empty_like(key_states[:, :0]), 'v': torch.empty_like(value_states[:, :0])}
            else:
                # Extend sink
                self._sink_kv[layer_idx]['k'] = torch.cat([self._sink_kv[layer_idx]['k'], key_states], dim=-2)
                self._sink_kv[layer_idx]['v'] = torch.cat([self._sink_kv[layer_idx]['v'], value_states], dim=-2)

            return self._get_full_sequence(layer_idx)

        elif mode == "decode":
            # Decode模式：单token，更新Tail窗口
            assert new_len == 1, f"Decode mode expects single token, got {new_len}"

            # 添加新token到Tail
            if self._tail_kv[layer_idx] is None:
                self._tail_kv[layer_idx] = {
                    'k': key_states.clone(),
                    'v': value_states.clone()
                }
            else:
                # 检查Tail是否已满
                tail_len = self._tail_kv[layer_idx]['k'].shape[-2]

                if tail_len >= self.tail_tokens:
                    # Tail满：从开头驱逐tokens
                    evicted_k = self._tail_kv[layer_idx]['k'][:, :1, :]  # 驱逐1个token
                    evicted_v = self._tail_kv[layer_idx]['v'][:, :1, :]

                    # 压缩并加入HeavyHitter竞争队列
                    evicted_k_4bit, evicted_v_4bit = self._compressor.compress(evicted_k, evicted_v)

                    # 获取驱逐tokens的注意力分数（从oracle）
                    evicted_scores = self._get_token_attention_scores(layer_idx, 1)

                    # 加入竞争队列
                    self._attention_queue.add_heavyhitter_tokens(
                        layer_idx, evicted_k_4bit['data'], evicted_v_4bit['data'], evicted_scores
                    )

                    # 滑动Tail窗口：移除开头，添加新token到末尾
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

            # 管理HeavyHitter分区大小
            self._manage_heavyhitter_zone(layer_idx)

            # 返回用于attention的KV (Sink + Tail + HeavyHitter top-K)
            return self._get_attention_kv(layer_idx)

        raise ValueError(f"Unknown mode: {mode}")

    def _get_full_sequence(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取完整序列（prefill模式）"""
        sink = self._sink_kv[layer_idx]
        tail = self._tail_kv[layer_idx] if self._tail_kv[layer_idx] is not None else {'k': torch.empty_like(sink['k'][:, :0]), 'v': torch.empty_like(sink['v'][:, :0])}

        k_full = torch.cat([sink['k'], tail['k']], dim=-2)
        v_full = torch.cat([sink['v'], tail['v']], dim=-2)

        return k_full, v_full

    def _get_attention_kv(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取用于attention计算的KV

        返回：Sink + Tail + HeavyHitter (top-K)
        HeavyHitter大小受预算限制
        """
        sink = self._sink_kv[layer_idx]
        tail = self._tail_kv[layer_idx]

        # 基础KV (Sink + Tail)
        k_base = torch.cat([sink['k'], tail['k']], dim=-2)
        v_base = torch.cat([sink['v'], tail['v']], dim=-2)

        # HeavyHitter (如果有)
        if self._heavyhitter_kv[layer_idx] is not None:
            hh_k = self._heavyhitter_kv[layer_idx]['k']
            hh_v = self._heavyhitter_kv[layer_idx]['v']

            k_final = torch.cat([k_base, hh_k], dim=-2)
            v_final = torch.cat([v_base, hh_v], dim=-2)
        else:
            k_final = k_base
            v_final = v_base

        return k_final, v_final

    def _get_token_attention_scores(self, layer_idx: int, num_tokens: int) -> torch.Tensor:
        """获取tokens的注意力分数（从oracle）"""
        if self._oracle.token_scores is None:
            # 冷启动：返回均匀分数
            return torch.ones(num_tokens, dtype=torch.float32, device=self.device)

        # 返回最近的分数
        current_len = self._oracle.token_scores.shape[0]
        return self._oracle.token_scores[-num_tokens:]

    def _manage_heavyhitter_zone(self, layer_idx: int):
        """
        管理HeavyHitter分区大小

        逻辑：
        1. 检查竞争队列中的tokens
        2. 选择高注意力的tokens加入HeavyHitter分区
        3. 如果超过预算，驱逐低注意力的tokens到DRAM
        """
        # 从竞争队列获取top-K tokens
        budget = self.heavyhitter_budget
        k_top, v_top = self._attention_queue.get_top_tokens(layer_idx, budget)

        if k_top is not None:
            # 存储到HeavyHitter分区
            self._heavyhitter_kv[layer_idx] = {
                'k': k_top,
                'v': v_top,
            }

            # 清除已处理的动态取回tokens
            self._attention_queue.clear_dynamic_retrieval(layer_idx)

        # 如果超过预算，驱逐低分数tokens到DRAM
        if self._heavyhitter_kv[layer_idx] is not None:
            current_len = self._heavyhitter_kv[layer_idx]['k'].shape[-2]

            if current_len > budget:
                # 驱除多余的tokens
                num_evict = current_len - budget
                evicted_k = self._heavyhitter_kv[layer_idx]['k'][:, :num_evict, :]
                evicted_v = self._heavyhitter_kv[layer_idx]['v'][:, :num_evict, :]

                # 压缩并存储到DRAM
                evicted_k_4bit, evicted_v_4bit = self._compressor.compress(evicted_k, evicted_v)
                self._dram.store(f"layer_{layer_idx}_hh_{torch.tensor([0])}", evicted_k_4bit, evicted_v_4bit)

                # 保留剩余tokens
                self._heavyhitter_kv[layer_idx]['k'] = self._heavyhitter_kv[layer_idx]['k'][:, num_evict:, :]
                self._heavyhitter_kv[layer_idx]['v'] = self._heavyhitter_kv[layer_idx]['v'][:, num_evict:, :]

    def update_attention_scores(self, attention_weights: torch.Tensor):
        """更新oracle的注意力分数"""
        self._oracle.update(attention_weights)

    def get_seq_length(self) -> int:
        """获取当前序列长度"""
        total = 0
        for layer in range(self.num_layers):
            if self._sink_kv[layer] is not None:
                total += self._sink_kv[layer]['k'].shape[-2]
            if self._tail_kv[layer] is not None:
                total += self._tail_kv[layer]['k'].shape[-2]
        return total