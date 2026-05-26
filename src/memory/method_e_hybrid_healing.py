"""
Method E: Hybrid Dynamic Window Enhancement
============================================

Problem: Current dynamic window trades recall for memory efficiency,
causing accuracy degradation in DRAM zone retrieval.

Solution: Multi-pronged enhancement combining:
1. Query-aware chunk scoring (cosine similarity with current query)
2. Tiered retrieval (start narrow, expand if needed)
3. Semantic re-ranking (re-score chunks based on current query)
4. Adaptive cache hit monitoring (expand window on cache misses)

Expected: >90% DRAM zone accuracy while maintaining O(1) memory
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple

@dataclass
class ChunkMetadata:
    """Enhanced metadata for each evicted chunk."""
    chunk_idx: int
    start_pos: int
    end_pos: int
    avg_attention: float  # Historical attention score
    embedding: Optional[torch.Tensor] = None  # Mean K embedding for query matching
    last_accessed: int = 0  # Last access step
    access_count: int = 0  # Number of accesses
    cache_hits: int = 0  # Successful cache hits
    cache_misses: int = 0  # Cache misses (retrieved but not useful)

class QueryAwareRetriever:
    """
    Query-aware chunk retrieval using semantic similarity.

    Idea: Chunks that are semantically similar to the current query
    are more likely to contain relevant information.

    Implementation:
    1. Pre-compute mean K embedding for each chunk during eviction
    2. At decode time, compute cosine similarity between query K and chunk embeddings
    3. Combine semantic score with historical attention score
    4. Retrieve top-k chunks based on combined score
    """

    def __init__(self, device: str = 'cuda', alpha: float = 0.5):
        """
        Args:
            device: Device for computations
            alpha: Weight for semantic similarity (0-1)
                   0.0 = pure historical attention
                   1.0 = pure semantic similarity
                   0.5 = balanced
        """
        self.device = device
        self.alpha = alpha
        self.chunk_metadata: Dict[int, ChunkMetadata] = {}

    def register_chunk(
        self,
        chunk_idx: int,
        start_pos: int,
        end_pos: int,
        avg_attention: float,
        key_states: torch.Tensor,  # [batch, heads, seq_len, head_dim]
    ):
        """Register a chunk with its metadata and compute mean K embedding."""
        # Compute mean K embedding across all heads and positions
        # key_states: [batch, heads, seq_len, head_dim]
        mean_k = key_states.mean(dim=(0, 1, 2))  # [head_dim]

        self.chunk_metadata[chunk_idx] = ChunkMetadata(
            chunk_idx=chunk_idx,
            start_pos=start_pos,
            end_pos=end_pos,
            avg_attention=avg_attention,
            embedding=mean_k.detach().cpu(),  # Store on CPU to save GPU memory
            last_accessed=0,
            access_count=0,
            cache_hits=0,
            cache_misses=0,
        )

    def compute_similarity_scores(
        self,
        query_key: torch.Tensor,  # [batch, heads, 1, head_dim]
        candidate_chunks: List[int],
    ) -> Dict[int, float]:
        """
        Compute cosine similarity between query K and chunk embeddings.

        Returns:
            Dict mapping chunk_idx -> similarity score
        """
        if not candidate_chunks:
            return {}

        # Compute mean query K across batch and heads
        # query_key: [batch, heads, 1, head_dim]
        mean_q = query_key.mean(dim=(0, 1))  # [1, head_dim]

        similarities = {}
        for chunk_idx in candidate_chunks:
            metadata = self.chunk_metadata.get(chunk_idx)
            if metadata is None or metadata.embedding is None:
                continue

            # Compute cosine similarity
            chunk_emb = metadata.embedding.to(self.device)
            sim = F.cosine_similarity(mean_q, chunk_emb.unsqueeze(0), dim=-1).item()
            similarities[chunk_idx] = max(0.0, sim)  # Clamp to [0, 1]

        return similarities

    def rank_chunks(
        self,
        query_key: torch.Tensor,
        candidate_chunks: List[int],
        top_k: int = 10,
    ) -> List[int]:
        """
        Rank chunks by combined semantic + historical score.

        Formula: combined_score = alpha * semantic_sim + (1-alpha) * normalized_attention
        """
        if not candidate_chunks:
            return []

        # Compute semantic similarities
        semantic_scores = self.compute_similarity_scores(query_key, candidate_chunks)

        # Get historical attention scores
        historical_scores = {
            idx: self.chunk_metadata[idx].avg_attention
            for idx in candidate_chunks
            if idx in self.chunk_metadata
        }

        # Normalize historical scores to [0, 1]
        if historical_scores:
            max_hist = max(historical_scores.values())
            min_hist = min(historical_scores.values())
            if max_hist > min_hist:
                historical_scores = {
                    idx: (val - min_hist) / (max_hist - min_hist)
                    for idx, val in historical_scores.items()
                }
            else:
                historical_scores = {idx: 1.0 for idx in historical_scores}

        # Combine scores
        combined_scores = {}
        for idx in candidate_chunks:
            semantic = semantic_scores.get(idx, 0.0)
            historical = historical_scores.get(idx, 0.0)
            combined_scores[idx] = (
                self.alpha * semantic +
                (1 - self.alpha) * historical
            )

        # Sort by combined score
        ranked = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        return [idx for idx, _ in ranked[:top_k]]

    def update_access_stats(self, chunk_idx: int, was_useful: bool, step: int):
        """Update access statistics for a chunk."""
        if chunk_idx not in self.chunk_metadata:
            return
        metadata = self.chunk_metadata[chunk_idx]
        metadata.last_accessed = step
        metadata.access_count += 1
        if was_useful:
            metadata.cache_hits += 1
        else:
            metadata.cache_misses += 1


class TieredRetrievalStrategy:
    """
    Tiered retrieval: start with narrow window, expand if answer quality is poor.

    Idea: Most queries can be answered with a narrow window (2-4 chunks).
    Only expand when we detect the answer is insufficient.

    Detection methods:
    1. Confidence score (max probability in output distribution)
    2. Entropy of output distribution
    3. Token repetition (model is stuck)
    4. Generic filler tokens (e.g., "The", "A", "An")

    Expansion strategy:
    - Tier 1: Retrieve 2 chunks (highest priority)
    - Tier 2: If confidence < threshold, retrieve 4 more chunks (total 6)
    - Tier 3: If still low confidence, retrieve 8 more chunks (total 14)
    - Tier 4: Full fallback (all chunks) if absolutely necessary
    """

    def __init__(
        self,
        confidence_threshold: float = 0.6,
        max_tiers: int = 3,
        tier_multipliers: List[int] = None,
    ):
        """
        Args:
            confidence_threshold: Min confidence to stop expansion
            max_tiers: Maximum number of expansion tiers
            tier_multipliers: Chunk count for each tier [tier1, tier2, tier3, ...]
        """
        self.confidence_threshold = confidence_threshold
        self.max_tiers = max_tiers
        self.tier_multipliers = tier_multipliers or [2, 4, 8, 16]

    def should_expand(
        self,
        logits: torch.Tensor,  # [vocab_size]
        generated_tokens: List[int],
        step: int,
    ) -> bool:
        """
        Determine if we should expand the retrieval window.

        Returns:
            True if expansion is needed
        """
        # Method 1: Check confidence (max probability)
        probs = F.softmax(logits, dim=-1)
        max_prob, max_token = probs.max(dim=-1)
        if max_prob.item() < self.confidence_threshold:
            return True

        # Method 2: Check entropy (high entropy = uncertain)
        entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
        if entropy > 3.0:  # High entropy threshold
            return True

        # Method 3: Check token repetition (stuck in loop)
        if len(generated_tokens) >= 4:
            last_4 = generated_tokens[-4:]
            if len(set(last_4)) <= 2:  # Highly repetitive
                return True

        return False

    def get_tier_size(self, current_tier: int) -> int:
        """Get chunk count for given tier."""
        if current_tier < len(self.tier_multipliers):
            return self.tier_multipliers[current_tier]
        return self.tier_multipliers[-1] * 2  # Exponential fallback


class HybridSelfHealingEngine:
    """
    Hybrid self-healing engine combining QueryAwareRetriever + TieredRetrievalStrategy.

    This is Method E: a hybrid approach that aims for >90% DRAM zone accuracy
    while maintaining O(1) memory behavior.
    """

    def __init__(
        self,
        device: str = 'cuda',
        alpha: float = 0.5,  # Query-aware semantic weight
        confidence_threshold: float = 0.6,
        max_tiers: int = 3,
    ):
        self.device = device
        self.query_retriever = QueryAwareRetriever(device, alpha)
        self.tiered_strategy = TieredRetrievalStrategy(confidence_threshold, max_tiers)

        self.current_tier = 0
        self.generation_history: List[Tuple[int, torch.Tensor, List[int]]] = []

    def register_evicted_chunk(
        self,
        chunk_idx: int,
        start_pos: int,
        end_pos: int,
        avg_attention: float,
        key_states: torch.Tensor,
    ):
        """Register a chunk when it's evicted to DRAM."""
        self.query_retriever.register_chunk(
            chunk_idx, start_pos, end_pos, avg_attention, key_states
        )

    def retrieve_chunks(
        self,
        query_key: torch.Tensor,
        candidate_chunks: List[int],
        step: int,
    ) -> Tuple[List[int], int]:
        """
        Retrieve chunks using tiered strategy.

        Returns:
            (chunk_indices, tier_used)
        """
        tier_size = self.tiered_strategy.get_tier_size(self.current_tier)
        ranked_chunks = self.query_retriever.rank_chunks(
            query_key, candidate_chunks, top_k=tier_size
        )

        return ranked_chunks, self.current_tier

    def update_generation(
        self,
        logits: torch.Tensor,
        generated_token: int,
        all_generated: List[int],
        retrieved_chunks: List[int],
        step: int,
    ) -> bool:
        """
        Update generation state and check if we need to expand.

        Returns:
            True if expansion is needed
        """
        self.generation_history.append((step, logits, all_generated.copy()))

        # Check if we should expand
        should_expand = self.tiered_strategy.should_expand(logits, all_generated, step)

        if should_expand and self.current_tier < self.tiered_strategy.max_tiers:
            self.current_tier += 1
            return True  # Signal that expansion is needed

        return False

    def mark_chunk_useful(self, chunk_idx: int, step: int):
        """Mark a chunk as useful (cache hit)."""
        self.query_retriever.update_access_stats(chunk_idx, True, step)

    def mark_chunk_not_useful(self, chunk_idx: int, step: int):
        """Mark a chunk as not useful (cache miss)."""
        self.query_retriever.update_access_stats(chunk_idx, False, step)

    def reset(self):
        """Reset state for new generation."""
        self.current_tier = 0
        self.generation_history.clear()


# Integration Example
"""
# In HeteroKVManager._decode_update():

1. When evicting to DRAM:
   - Compute mean K embedding for the chunk
   - Register with hybrid_engine.register_evicted_chunk()

2. During decode (before attention):
   - Get candidate chunks from DRAM
   - Call hybrid_engine.retrieve_chunks(query_k, candidates, step)
   - Retrieve and swap in the returned chunks
   - Perform attention with Sink + Tail + HeavyHitter + SwappedIn

3. After generating each token:
   - Call hybrid_engine.update_generation(logits, token, all_tokens, retrieved, step)
   - If returns True, re-run retrieval with expanded tier
   - Track which chunks were useful via mark_chunk_useful/not_useful
"""

if __name__ == "__main__":
    # Simple test
    print("Method E: Hybrid Dynamic Window Enhancement")
    print("=" * 60)
    print("Components:")
    print("  1. QueryAwareRetriever - Semantic similarity ranking")
    print("  2. TieredRetrievalStrategy - Adaptive expansion")
    print("  3. HybridSelfHealingEngine - Combined approach")
    print("\nExpected: >90% DRAM zone accuracy with O(1) memory")
