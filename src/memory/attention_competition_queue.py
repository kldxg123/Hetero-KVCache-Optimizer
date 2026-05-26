"""
src/memory/attention_competition_queue.py
=======================================
注意力竞争队列管理器

按照用户设计实现：
- Tail驱逐的tokens加入竞争队列
- 动态窗口取回的tokens（寄存器计算后）也加入队列
- 基于注意力分数竞争HeavyHitter HBM分区
- 低分数tokens驱逐到DRAM
"""

import torch
from typing import Dict, List, Optional, Tuple


class AttentionCompetitionQueue:
    """
    注意力竞争队列

    管理待定HBM空间的tokens：
    - 从Tail驱逐的tokens
    - 动态窗口取回的tokens（寄存器计算后）

    基于注意力分数决定：
    - 高分数 → HeavyHitter HBM分区
    - 低分数 → DRAM存储
    """

    def __init__(self):
        self.queue: Dict[str, Dict] = {}  # {token_id: {k, v, score, compressed, layer}}
        self._token_counter = 0

    def enqueue(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        scores: torch.Tensor,
        compressed: Optional[Dict] = None,
        layer_idx: int = 0,
        prefix: str = "token"
    ):
        """
        添加tokens到竞争队列

        Args:
            k: [batch, heads, tokens, dim] key tensor
            v: [batch, heads, tokens, dim] value tensor
            scores: [tokens] attention scores
            compressed: optional 4-bit compressed data {k_data, k_scales, k_zps, v_data, v_scales, v_zps}
            layer_idx: layer index
            prefix: token id prefix
        """
        num_tokens = k.shape[-2]

        # Handle scores: if single score, broadcast to all tokens
        if scores.numel() == 1:
            scores = scores.expand(num_tokens)

        for i in range(num_tokens):
            token_id = f"{prefix}_{layer_idx}_{self._token_counter}"

            self.queue[token_id] = {
                'k': k[..., i:i+1, :],  # [batch, heads, 1, dim]
                'v': v[..., i:i+1, :],
                'score': scores[i].item() if scores.dim() > 0 else float(scores),
                'compressed': compressed,
                'layer': layer_idx,
            }

            self._token_counter += 1

    def dequeue_top_k(self, k: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        取出top-k高分数的tokens

        Returns:
            k: [batch, heads, k, dim]
            v: [batch, heads, k, dim]
            scores: [k]
        """
        if not self.queue:
            return None, None, None

        # 按分数排序
        sorted_items = sorted(self.queue.items(), key=lambda x: x[1]['score'], reverse=True)

        # 取top-k
        top_k_items = sorted_items[:k]

        if not top_k_items:
            return None, None, None

        # 提取K, V, scores
        k_list = [item[1]['k'] for item in top_k_items]
        v_list = [item[1]['v'] for item in top_k_items]
        scores_list = [item[1]['score'] for item in top_k_items]

        k_cat = torch.cat(k_list, dim=-2)  # [batch, heads, k, dim]
        v_cat = torch.cat(v_list, dim=-2)
        scores_tensor = torch.tensor(scores_list, dtype=torch.float32, device=k_list[0].device)

        # 从队列移除已处理的tokens
        for token_id, _ in top_k_items:
            del self.queue[token_id]

        return k_cat, v_cat, scores_tensor

    def get_low_score_tokens(self, threshold: float, max_tokens: Optional[int] = None) -> List[str]:
        """
        获取低于阈值的tokens（用于驱逐）

        Args:
            threshold: 分数阈值
            max_tokens: 最大返回token数

        Returns:
            token_ids低于阈值的token ID列表
        """
        low_tokens = [
            token_id for token_id, item in self.queue.items()
            if item['score'] < threshold
        ]

        if max_tokens is not None and len(low_tokens) > max_tokens:
            # 按分数排序，取最低的max_tokens个
            sorted_low = sorted(
                [(tid, self.queue[tid]['score']) for tid in low_tokens],
                key=lambda x: x[1]
            )
            low_tokens = [tid for tid, _ in sorted_low[:max_tokens]]

        return low_tokens

    def evict_to_dram(
        self,
        token_ids: List[str],
        compressor,
        dram_storage,
        layer_idx: int
    ):
        """
        将低分数tokens压缩并驱逐到DRAM

        Args:
            token_ids: 要驱逐的token ID列表
            compressor: KVCompressor实例
            dram_storage: DRAMStorageManager实例
            layer_idx: layer index
        """
        for token_id in token_ids:
            if token_id not in self.queue:
                continue

            item = self.queue[token_id]

            # 压缩
            if item['compressed'] is None:
                k_4bit, v_4bit = compressor.compress(item['k'], item['v'])
            else:
                k_4bit = {'data': item['compressed']['k_data'], 'scales': item['compressed']['k_scales']}
                v_4bit = {'data': item['compressed']['v_data'], 'scales': item['compressed']['v_scales']}

            # 存储到DRAM
            dram_storage.store(token_id, k_4bit, v_4bit)

            # 从队列移除
            del self.queue[token_id]

    def size(self) -> int:
        """返回队列中token数量"""
        return len(self.queue)

    def clear(self):
        """清空队列"""
        self.queue.clear()
        self._token_counter = 0