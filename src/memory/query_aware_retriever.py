"""
Token-level Query x Key retrieval for DRAM-resident KV chunks.

This module intentionally avoids mean-K, pooled chunk embeddings, and cosine
similarity on averaged features.  It ranks candidate chunks by dequantizing one
small DRAM chunk at a time and computing approximate attention scores:

    score = query_states @ key_states.T

The temporary dequantized Key and score tensors are released after each chunk.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch


@dataclass
class ChunkMetadata:
    chunk_key: str
    start_pos: int
    end_pos: int
    historical_attention: float = 0.0


class QueryAwareRetriever:
    """Rank DRAM chunks with token-level dot-product scoring."""

    def __init__(
        self,
        device: str = "cuda",
        alpha: float = 1.0,
        min_similarity_threshold: float = 0.0,
        score_reduce: str = "max",
        top_r: int = 8,
        use_triton_scoring: bool = False,
        triton_scoring_batch_chunks: int = 8,
    ):
        self.device = device
        self.alpha = alpha
        self.min_similarity_threshold = min_similarity_threshold
        self.score_reduce = score_reduce
        self.top_r = top_r
        self.use_triton_scoring = bool(use_triton_scoring)
        self.triton_scoring_batch_chunks = max(1, int(triton_scoring_batch_chunks))
        self.chunk_metadata: Dict[str, ChunkMetadata] = {}
        self.last_scores: Dict[str, float] = {}
        self.last_best_token_offsets: Dict[str, int] = {}
        self.last_scoring_backend: str = "torch_dequant"
        self._triton_disabled_reason: Optional[str] = None

    def _compute_triton_batch_scores(
        self,
        q: torch.Tensor,
        candidate_keys: List[str],
        dram_table: Dict[str, Dict[str, torch.Tensor]],
        compressor,
    ) -> Dict[str, float]:
        from src.quantization.kernels.int4_dot_score import score_int4_key_chunks_batch

        scores: Dict[str, float] = {}
        offsets: Dict[str, int] = {}
        index = 0
        while index < len(candidate_keys):
            first_key = candidate_keys[index]
            first_entry = dram_table.get(first_key)
            if first_entry is None:
                index += 1
                continue
            first_shape = tuple(first_entry["k_data"].shape)
            first_scale_shape = tuple(first_entry["k_scales"].shape)
            batch_keys: List[str] = []
            batch_entries: List[Dict[str, torch.Tensor]] = []
            while (
                index < len(candidate_keys)
                and len(batch_keys) < self.triton_scoring_batch_chunks
            ):
                key = candidate_keys[index]
                entry = dram_table.get(key)
                if (
                    entry is not None
                    and tuple(entry["k_data"].shape) == first_shape
                    and tuple(entry["k_scales"].shape) == first_scale_shape
                ):
                    batch_keys.append(key)
                    batch_entries.append(entry)
                    index += 1
                    continue
                break

            if not batch_keys:
                index += 1
                continue

            q_k = torch.stack(
                [entry["k_data"] for entry in batch_entries],
                dim=0,
            ).to(self.device, non_blocking=True)
            s_k = torch.stack(
                [entry["k_scales"] for entry in batch_entries],
                dim=0,
            ).to(self.device, non_blocking=True)
            z_k = torch.stack(
                [entry["k_zps"] for entry in batch_entries],
                dim=0,
            ).to(self.device, non_blocking=True)
            fused = score_int4_key_chunks_batch(
                q,
                q_k,
                s_k,
                z_k,
                group_size=getattr(compressor, "group_size", 128),
                score_reduce=self.score_reduce,
                top_r=self.top_r,
            )
            for key, score, offset in zip(
                batch_keys,
                fused["scores"],
                fused["best_token_offsets"],
            ):
                scores[key] = float(score)
                offsets[key] = int(offset)
            del q_k, s_k, z_k

        self.last_best_token_offsets = offsets
        return scores

    def register_chunk(
        self,
        chunk_key: str,
        start_pos: int,
        end_pos: int,
        historical_attention: float = 0.0,
        key_states: Optional[torch.Tensor] = None,
    ) -> None:
        # key_states is accepted for backward compatibility but is deliberately
        # ignored; the main path scores against the 4-bit DRAM Key itself.
        self.chunk_metadata[chunk_key] = ChunkMetadata(
            chunk_key=chunk_key,
            start_pos=start_pos,
            end_pos=end_pos,
            historical_attention=historical_attention,
        )

    def _reduce_token_scores(self, scores: torch.Tensor) -> float:
        if self.score_reduce == "query_top_r_mean" and scores.dim() == 4:
            per_query = scores.float().amax(dim=(0, 1, 3))
            if per_query.numel() == 0:
                return float("-inf")
            k = min(max(1, self.top_r), per_query.numel())
            return float(torch.topk(per_query, k=k, largest=True).values.mean().item())
        if self.score_reduce == "query_mean_max" and scores.dim() == 4:
            per_query = scores.float().amax(dim=(0, 1, 3))
            return float(per_query.mean().item()) if per_query.numel() else float("-inf")

        flat = scores.reshape(-1).float()
        if flat.numel() == 0:
            return float("-inf")
        if self.score_reduce == "top_r_mean":
            k = min(self.top_r, flat.numel())
            return float(torch.topk(flat, k=k, largest=True).values.mean().item())
        if self.score_reduce == "head_mean_max" and scores.dim() == 4:
            # Reduce single-head spikes by requiring a token to score well on
            # average across attention heads.  This is useful at 128K where
            # max-over-all-heads can over-rank unrelated chunks.
            per_token = scores.float().mean(dim=1).reshape(-1)
            return float(per_token.max().item()) if per_token.numel() else float("-inf")
        if self.score_reduce == "head_mean_top_r_mean" and scores.dim() == 4:
            per_token = scores.float().mean(dim=1).reshape(-1)
            if per_token.numel() == 0:
                return float("-inf")
            k = min(self.top_r, per_token.numel())
            return float(torch.topk(per_token, k=k, largest=True).values.mean().item())
        if self.score_reduce == "z_score_max":
            std = flat.std(unbiased=False)
            if not torch.isfinite(std) or float(std.item()) <= 1.0e-6:
                return float("-inf")
            return float(((flat.max() - flat.mean()) / std).item())
        if self.score_reduce == "peak_contrast":
            k = min(max(2, self.top_r + 1), flat.numel())
            vals = torch.topk(flat, k=k, largest=True).values
            return float((vals[0] - vals[1:].mean()).item())
        return float(flat.max().item())

    @staticmethod
    def _best_token_offset(scores: torch.Tensor) -> int:
        token_scores = scores.float()
        while token_scores.dim() > 1:
            token_scores = token_scores.max(dim=0).values
        if token_scores.numel() == 0:
            return 0
        return int(token_scores.argmax().item())

    @staticmethod
    def _token_dot_scores(query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        """Compute Q x K scores while respecting grouped-query attention heads."""
        if query.dim() != 4 or key.dim() != 4:
            return torch.matmul(query, key.transpose(-2, -1))

        batch_q, q_heads, q_len, head_dim = query.shape
        batch_k, kv_heads, kv_len, key_dim = key.shape
        if batch_q != batch_k or head_dim != key_dim:
            raise RuntimeError(
                f"Q/K shape mismatch: query={tuple(query.shape)} key={tuple(key.shape)}"
            )

        if q_heads == kv_heads or q_heads == 1 or kv_heads == 1:
            return torch.matmul(query, key.transpose(-2, -1))

        if q_heads % kv_heads == 0:
            groups = q_heads // kv_heads
            grouped_q = query.reshape(batch_q, kv_heads, groups, q_len, head_dim)
            grouped_scores = torch.einsum("bhgqd,bhkd->bhgqk", grouped_q, key)
            return grouped_scores.reshape(batch_q, q_heads, q_len, kv_len)

        if kv_heads % q_heads == 0:
            groups = kv_heads // q_heads
            grouped_key = key.reshape(batch_k, q_heads, groups, kv_len, key_dim).mean(dim=2)
            return torch.matmul(query, grouped_key.transpose(-2, -1))

        raise RuntimeError(
            f"Unsupported GQA head layout: query_heads={q_heads}, kv_heads={kv_heads}"
        )

    @torch.no_grad()
    def compute_dot_product_scores(
        self,
        query_states: torch.Tensor,
        candidate_keys: List[str],
        dram_table: Dict[str, Dict[str, torch.Tensor]],
        compressor,
    ) -> Dict[str, float]:
        """Score chunks by dequantizing one candidate Key chunk at a time."""
        if not candidate_keys:
            return {}

        q = query_states.detach().to(self.device, non_blocking=True).float()
        if q.dim() == 3:
            q = q.unsqueeze(2)

        scores: Dict[str, float] = {}
        self.last_best_token_offsets = {}
        self.last_scoring_backend = "triton_int4" if self.use_triton_scoring else "torch_dequant"
        if self.use_triton_scoring and q.is_cuda:
            try:
                scores = self._compute_triton_batch_scores(
                    q=q,
                    candidate_keys=candidate_keys,
                    dram_table=dram_table,
                    compressor=compressor,
                )
                self.last_scores = scores
                self.last_scoring_backend = "triton_int4_batch"
                return scores
            except Exception as exc:
                self._triton_disabled_reason = str(exc)
                self.use_triton_scoring = False
                self.last_scoring_backend = "torch_dequant_fallback"
                print(
                    "  [DotProductRetrieval] Triton batch scoring fallback: "
                    f"{self._triton_disabled_reason}"
                )
        for chunk_key in candidate_keys:
            entry = dram_table.get(chunk_key)
            if entry is None:
                continue
            try:
                q_k = entry["k_data"].to(self.device, non_blocking=True)
                s_k = entry["k_scales"].to(self.device, non_blocking=True)
                z_k = entry["k_zps"].to(self.device, non_blocking=True)
                if self.use_triton_scoring and q.is_cuda:
                    try:
                        from src.quantization.kernels.int4_dot_score import score_int4_key_chunk

                        fused = score_int4_key_chunk(
                            q,
                            q_k,
                            s_k,
                            z_k,
                            group_size=getattr(compressor, "group_size", 128),
                            score_reduce=self.score_reduce,
                            top_r=self.top_r,
                        )
                        scores[chunk_key] = float(fused["score"])
                        self.last_best_token_offsets[chunk_key] = int(
                            fused["best_token_offset"]
                        )
                        del q_k, s_k, z_k
                        continue
                    except Exception as exc:
                        self._triton_disabled_reason = str(exc)
                        self.use_triton_scoring = False
                        self.last_scoring_backend = "torch_dequant_fallback"
                        print(
                            "  [DotProductRetrieval] Triton scoring fallback: "
                            f"{self._triton_disabled_reason}"
                        )
                restored_k = compressor.decompress(
                    q_k, s_k, z_k, target_dtype=torch.float16
                ).float()
                token_scores = self._token_dot_scores(q, restored_k)
                scores[chunk_key] = self._reduce_token_scores(token_scores)
                self.last_best_token_offsets[chunk_key] = self._best_token_offset(token_scores)
                del q_k, s_k, z_k, restored_k, token_scores
                if torch.cuda.is_available() and str(self.device).startswith("cuda"):
                    torch.cuda.empty_cache()
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower() and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(f"  [DotProductRetrieval] failed scoring {chunk_key}: {exc}")
            except Exception as exc:
                print(f"  [DotProductRetrieval] failed scoring {chunk_key}: {exc}")

        self.last_scores = scores
        return scores

    def rank_chunks_by_scores(
        self,
        candidate_keys: List[str],
        dot_scores: Dict[str, float],
        top_k: Optional[int] = None,
    ) -> List[str]:
        if not candidate_keys:
            return []

        if self.alpha < 1.0:
            hist = {
                key: self.chunk_metadata.get(key, ChunkMetadata(key, 0, 0)).historical_attention
                for key in candidate_keys
            }
            if hist:
                max_hist, min_hist = max(hist.values()), min(hist.values())
                if max_hist > min_hist:
                    hist = {k: (v - min_hist) / (max_hist - min_hist) for k, v in hist.items()}
                else:
                    hist = {k: 1.0 for k in hist}
        else:
            hist = {}

        combined = {}
        for key in candidate_keys:
            dot = dot_scores.get(key, float("-inf"))
            if dot == float("-inf"):
                continue
            combined[key] = self.alpha * dot + (1.0 - self.alpha) * hist.get(key, 0.0)

        ranked = sorted(combined.items(), key=lambda item: item[1], reverse=True)
        if top_k is not None:
            ranked = ranked[:top_k]
        return [key for key, _ in ranked]

    def remove_chunk(self, chunk_key: str) -> None:
        self.chunk_metadata.pop(chunk_key, None)
        self.last_scores.pop(chunk_key, None)
        self.last_best_token_offsets.pop(chunk_key, None)

    def clear_all(self) -> None:
        self.chunk_metadata.clear()
        self.last_scores.clear()
        self.last_best_token_offsets.clear()

    def get_stats(self) -> Dict[str, object]:
        return {
            "total_chunks": len(self.chunk_metadata),
            "device": self.device,
            "alpha": self.alpha,
            "score_reduce": self.score_reduce,
            "top_r": self.top_r,
            "use_triton_scoring": self.use_triton_scoring,
            "triton_scoring_batch_chunks": self.triton_scoring_batch_chunks,
            "last_scoring_backend": self.last_scoring_backend,
            "triton_disabled_reason": self._triton_disabled_reason,
        }


class HybridRetrievalStrategy:
    """Compatibility facade for Method D retrieval."""

    def __init__(
        self,
        device: str = "cuda",
        enable: bool = True,
        alpha: float = 1.0,
        min_similarity_threshold: float = 0.0,
        fallback_to_method_c: bool = True,
        score_reduce: str = "max",
        top_r: int = 8,
        use_triton_scoring: bool = False,
        triton_scoring_batch_chunks: int = 8,
    ):
        self.enable = enable
        self.fallback_to_method_c = fallback_to_method_c
        self.query_aware_retriever = QueryAwareRetriever(
            device=device,
            alpha=alpha,
            min_similarity_threshold=min_similarity_threshold,
            score_reduce=score_reduce,
            top_r=top_r,
            use_triton_scoring=use_triton_scoring,
            triton_scoring_batch_chunks=triton_scoring_batch_chunks,
        )

    def register_chunk(
        self,
        chunk_key: str,
        start_pos: int,
        end_pos: int,
        key_states: Optional[torch.Tensor] = None,
        historical_attention: float = 0.0,
    ) -> None:
        if self.enable:
            self.query_aware_retriever.register_chunk(
                chunk_key=chunk_key,
                start_pos=start_pos,
                end_pos=end_pos,
                historical_attention=historical_attention,
                key_states=key_states,
            )

    def retrieve_chunks(
        self,
        query_key: torch.Tensor,
        candidate_keys: List[str],
        top_k: int,
        historical_scores: Optional[Dict[str, float]] = None,
        dram_table: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
        compressor=None,
    ) -> Tuple[List[str], str]:
        if not self.enable or not candidate_keys:
            return self._fallback(candidate_keys, top_k, historical_scores)
        if dram_table is None or compressor is None:
            if self.fallback_to_method_c:
                return self._fallback(candidate_keys, top_k, historical_scores)
            return [], "dot_product_missing_dram"

        dot_scores = self.query_aware_retriever.compute_dot_product_scores(
            query_states=query_key,
            candidate_keys=candidate_keys,
            dram_table=dram_table,
            compressor=compressor,
        )
        ranked = self.query_aware_retriever.rank_chunks_by_scores(
            candidate_keys=candidate_keys,
            dot_scores=dot_scores,
            top_k=top_k,
        )
        if not ranked and self.fallback_to_method_c:
            return self._fallback(candidate_keys, top_k, historical_scores)
        return ranked, "dot_product"

    def _fallback(
        self,
        candidate_keys: List[str],
        top_k: int,
        historical_scores: Optional[Dict[str, float]],
    ) -> Tuple[List[str], str]:
        if historical_scores:
            ranked = sorted(candidate_keys, key=lambda k: historical_scores.get(k, 0.0), reverse=True)
            return ranked[:top_k], "method_c_fallback"
        return candidate_keys[:top_k], "method_c_fallback"

    def clear_all(self) -> None:
        self.query_aware_retriever.clear_all()

    def get_stats(self) -> Dict[str, object]:
        stats = self.query_aware_retriever.get_stats()
        stats["enabled"] = self.enable
        stats["fallback_to_method_c"] = self.fallback_to_method_c
        return stats
