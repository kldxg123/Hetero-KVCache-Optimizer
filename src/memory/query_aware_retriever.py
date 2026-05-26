"""
Query-Aware DRAM Chunk Retrieval (Method D)
============================================

独立实现方案 D：基于当前 query 的语义相似度检索 DRAM chunks。

理论依据：
- QK^T 相似度在注意力机制中天然存在
- 假设：与当前 query K 相似度高的历史 chunks 更可能被 attend 到

实现原则：
1. 独立模块，不破坏现有架构
2. 通过配置开关启用
3. Fallback 到方案 C（动态窗口）
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ChunkEmbeddingMetadata:
    """
    每个 DRAM chunk 的 embedding 元数据。

    存储：
    - chunk 的平均 K embedding（用于与 query K 计算相似度）
    - chunk 的位置信息
    - 历史注意力分数（用于 fallback 到方案 C）
    """
    chunk_key: str  # DRAM storage key
    start_pos: int
    end_pos: int
    mean_k_embedding: torch.Tensor  # [head_dim] 存在 CPU 上节省 GPU 内存
    historical_attention: float = 0.0  # 历史平均注意力分数（方案 C fallback）


class QueryAwareRetriever:
    """
    Query-aware DRAM chunk retriever.

    核心思想：
    - 在每次驱逐到 DRAM 时，计算并存储该 chunk 的平均 K embedding
    - 在 decode 步骤，将当前 query 的 K 与所有 DRAM chunk embeddings 计算余弦相似度
    - 检索相似度最高的 top-k 个 chunks

    与方案 C 的区别：
    - 方案 C：按历史注意力分数排序（静态）
    - 方案 D：按当前 query 的语义相似度排序（动态）
    """

    def __init__(
        self,
        device: str = 'cuda',
        alpha: float = 1.0,  # 1.0 = 纯 query-aware, 0.0 = 纯历史注意力
        min_similarity_threshold: float = 0.0,  # 最低相似度阈值
    ):
        """
        Args:
            device: 计算设备
            alpha: 混合系数
                1.0 = 完全使用 query 相似度（纯方案 D）
                0.0 = 完全使用历史注意力（方案 C fallback）
                0.5 = 混合两种信号
            min_similarity_threshold: 只检索相似度超过阈值的 chunks
        """
        self.device = device
        self.alpha = alpha
        self.min_similarity_threshold = min_similarity_threshold

        # 存储所有 DRAM chunks 的元数据
        self.chunk_metadata: Dict[str, ChunkEmbeddingMetadata] = {}

    def register_chunk(
        self,
        chunk_key: str,
        start_pos: int,
        end_pos: int,
        key_states: torch.Tensor,  # [batch, heads, seq_len, head_dim]
        historical_attention: float = 0.0,
    ) -> None:
        """
        当一个 chunk 被驱逐到 DRAM 时，注册其元数据。

        计算 chunk 的平均 K embedding（跨 batch、heads、seq_len 维度）。
        存储在 CPU 上以节省 GPU 内存。
        """
        # key_states: [batch, heads, seq_len, head_dim]
        # 计算跨所有维度的平均 embedding
        mean_k = key_states.mean(dim=(0, 1, 2))  # [head_dim]

        # detach 并移到 CPU
        mean_k_cpu = mean_k.detach().cpu()

        self.chunk_metadata[chunk_key] = ChunkEmbeddingMetadata(
            chunk_key=chunk_key,
            start_pos=start_pos,
            end_pos=end_pos,
            mean_k_embedding=mean_k_cpu,
            historical_attention=historical_attention,
        )

    def compute_query_similarities(
        self,
        query_key: torch.Tensor,  # [batch, heads, 1, head_dim]
        candidate_keys: List[str],
    ) -> Dict[str, float]:
        """
        计算 query K 与所有 candidate chunks 的余弦相似度。

        Args:
            query_key: 当前 query 的 K 张量
            candidate_keys: 待评估的 DRAM chunk keys

        Returns:
            Dict: chunk_key -> similarity_score (0-1)
        """
        if not candidate_keys:
            return {}

        # 计算 query 的平均 K（跨 batch 和 heads）
        # query_key: [batch, heads, 1, head_dim]
        mean_query_k = query_key.mean(dim=(0, 1))  # [1, head_dim]

        similarities = {}
        for chunk_key in candidate_keys:
            metadata = self.chunk_metadata.get(chunk_key)
            if metadata is None:
                continue

            # 将 chunk embedding 移到 GPU 进行计算
            chunk_emb = metadata.mean_k_embedding.to(self.device)

            # 计算余弦相似度
            # similarity = (query · chunk) / (||query|| * ||chunk||)
            similarity = F.cosine_similarity(
                mean_query_k,  # [1, head_dim]
                chunk_emb.unsqueeze(0),  # [1, head_dim]
                dim=-1,
            ).item()

            # Clamp 到 [0, 1]
            similarities[chunk_key] = max(0.0, min(1.0, similarity))

        return similarities

    def rank_chunks(
        self,
        query_key: torch.Tensor,
        candidate_keys: List[str],
        top_k: Optional[int] = None,
    ) -> List[str]:
        """
        根据 query 相似度（或混合信号）对 chunks 排序。

        Args:
            query_key: 当前 query 的 K
            candidate_keys: 候选 chunk keys
            top_k: 返回前 k 个，None 返回全部

        Returns:
            List[str]: 排序后的 chunk keys
        """
        if not candidate_keys:
            return []

        # 计算语义相似度分数
        semantic_scores = self.compute_query_similarities(query_key, candidate_keys)

        # 获取历史注意力分数
        historical_scores = {
            key: self.chunk_metadata[key].historical_attention
            for key in candidate_keys
            if key in self.chunk_metadata
        }

        # 归一化历史分数到 [0, 1]
        if historical_scores:
            max_hist = max(historical_scores.values())
            min_hist = min(historical_scores.values())
            if max_hist > min_hist:
                historical_scores = {
                    key: (val - min_hist) / (max_hist - min_hist)
                    for key, val in historical_scores.items()
                }
            else:
                historical_scores = {key: 1.0 for key in historical_scores}

        # 混合两种信号
        combined_scores = {}
        for key in candidate_keys:
            semantic = semantic_scores.get(key, 0.0)
            historical = historical_scores.get(key, 0.0)

            # 加权组合
            combined_scores[key] = (
                self.alpha * semantic +
                (1 - self.alpha) * historical
            )

        # 过滤低相似度 chunks
        if self.min_similarity_threshold > 0:
            combined_scores = {
                key: score
                for key, score in combined_scores.items()
                if semantic_scores.get(key, 0.0) >= self.min_similarity_threshold
            }

        # 按分数排序
        ranked = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)

        # 返回 top-k（或全部）
        if top_k is not None:
            ranked = ranked[:top_k]

        return [key for key, _ in ranked]

    def remove_chunk(self, chunk_key: str) -> None:
        """删除一个 chunk 的元数据。"""
        if chunk_key in self.chunk_metadata:
            del self.chunk_metadata[chunk_key]

    def clear_all(self) -> None:
        """清空所有元数据。"""
        self.chunk_metadata.clear()

    def get_stats(self) -> Dict[str, any]:
        """获取统计信息。"""
        return {
            'total_chunks': len(self.chunk_metadata),
            'device': self.device,
            'alpha': self.alpha,
            'min_similarity_threshold': self.min_similarity_threshold,
        }


class HybridRetrievalStrategy:
    """
    混合检索策略：结合 Query-Aware 和历史注意力。

    这是方案 D 的完整实现，可以独立启用或禁用。
    """

    def __init__(
        self,
        device: str = 'cuda',
        enable: bool = True,
        alpha: float = 1.0,
        min_similarity_threshold: float = 0.0,
        fallback_to_method_c: bool = True,
    ):
        """
        Args:
            device: 计算设备
            enable: 是否启用 query-aware 检索
            alpha: 语义相似度权重（1.0=纯方案D, 0.0=纯方案C）
            min_similarity_threshold: 最低相似度阈值
            fallback_to_method_c: 如果没有相似 chunks，fallback 到方案 C
        """
        self.enable = enable
        self.fallback_to_method_c = fallback_to_method_c

        self.query_aware_retriever = QueryAwareRetriever(
            device=device,
            alpha=alpha,
            min_similarity_threshold=min_similarity_threshold,
        )

    def register_chunk(
        self,
        chunk_key: str,
        start_pos: int,
        end_pos: int,
        key_states: torch.Tensor,
        historical_attention: float = 0.0,
    ) -> None:
        """注册一个 chunk（驱逐到 DRAM 时调用）。"""
        if self.enable:
            self.query_aware_retriever.register_chunk(
                chunk_key=chunk_key,
                start_pos=start_pos,
                end_pos=end_pos,
                key_states=key_states,
                historical_attention=historical_attention,
            )

    def retrieve_chunks(
        self,
        query_key: torch.Tensor,
        candidate_keys: List[str],
        top_k: int,
        historical_scores: Optional[Dict[str, float]] = None,
    ) -> Tuple[List[str], str]:
        """
        检索 top-k 个最相关的 chunks。

        Returns:
            (selected_keys, method_used)
            method_used: "method_d" (query-aware) 或 "method_c" (historical)
        """
        if not self.enable or not candidate_keys:
            # Fallback 到方案 C：按历史注意力排序
            if historical_scores:
                ranked = sorted(
                    candidate_keys,
                    key=lambda k: historical_scores.get(k, 0.0),
                    reverse=True,
                )
                return ranked[:top_k], "method_c_fallback"
            return candidate_keys[:top_k], "method_c_fallback"

        # 方案 D：query-aware 排序
        ranked_by_query = self.query_aware_retriever.rank_chunks(
            query_key=query_key,
            candidate_keys=candidate_keys,
            top_k=top_k,
        )

        # 检查是否找到了足够的 chunks
        if len(ranked_by_query) == 0 and self.fallback_to_method_c:
            # 没有找到高相似度的 chunks，fallback 到方案 C
            if historical_scores:
                ranked = sorted(
                    candidate_keys,
                    key=lambda k: historical_scores.get(k, 0.0),
                    reverse=True,
                )
                return ranked[:top_k], "method_c_fallback"

        return ranked_by_query, "method_d"

    def clear_all(self) -> None:
        """清空所有元数据。"""
        self.query_aware_retriever.clear_all()

    def get_stats(self) -> Dict[str, any]:
        """获取统计信息。"""
        stats = self.query_aware_retriever.get_stats()
        stats['enabled'] = self.enable
        stats['fallback_to_method_c'] = self.fallback_to_method_c
        return stats


# 用于测试的简单示例
if __name__ == "__main__":
    print("Query-Aware Retriever (Method D)")
    print("=" * 60)
    print("独立实现，不影响现有架构")
    print("通过配置开关启用/禁用")
    print("自动 fallback 到方案 C")
    print("\n核心假设:")
    print("  - 当前 query K 与历史 chunk K 的相似度")
    print("    可以预测该 chunk 会被 attend 到的程度")
    print("\n预期优势:")
    print("  - 对语义相关的 query 更准确")
    print("  - 不依赖历史注意力模式")
    print("\n潜在风险:")
    print("  - 理论依据不扎实（需要实验验证）")
    print("  - 多模态场景（VQA）效果不确定")
    print("  - 增加 CPU-GPU 数据传输")
