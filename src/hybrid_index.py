"""HybridIndex: Combines BM25 and vector search with reciprocal rank fusion."""

from __future__ import annotations

from dataclasses import dataclass

from src.retrieval import RetrievalResult, SearchIndex


@dataclass(frozen=True)
class RankedResult:
    """Immutable intermediate result with a fused score."""

    doc_id: str
    chunk_id: str
    text: str
    keyword_rank: int | None
    semantic_rank: int | None
    fused_score: float


class HybridIndex:
    """Combines BM25 keyword and vector semantic search using reciprocal rank fusion.

    The alpha parameter controls the weighting:
        - alpha=0.0 -> keyword-only
        - alpha=1.0 -> semantic-only
        - alpha=0.5 -> equal weighting (default)

    Uses Reciprocal Rank Fusion (RRF) formula:
        score = (1 - alpha) / (k + keyword_rank) + alpha / (k + semantic_rank)
    where k is a constant (default 60) that dampens the impact of high ranks.
    """

    def __init__(
        self,
        keyword_index: SearchIndex,
        semantic_index: SearchIndex,
        alpha: float = 0.5,
        rrf_k: int = 60,
    ) -> None:
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        if rrf_k < 1:
            raise ValueError(f"rrf_k must be >= 1, got {rrf_k}")

        self._keyword_index = keyword_index
        self._semantic_index = semantic_index
        self._alpha = alpha
        self._rrf_k = rrf_k

    @property
    def alpha(self) -> float:
        return self._alpha

    def search(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        """Run both searches and fuse results using RRF.

        Args:
            query: The search query.
            top_k: Number of final results to return.

        Returns:
            List of RetrievalResult ranked by fused score.
        """
        if top_k < 1:
            raise ValueError("top_k must be >= 1")

        keyword_results = (
            self._keyword_index.query(query, top_k) if self._alpha < 1.0 else []
        )
        semantic_results = (
            self._semantic_index.query(query, top_k) if self._alpha > 0.0 else []
        )

        ranked = reciprocal_rank_fusion(
            keyword_results=keyword_results,
            semantic_results=semantic_results,
            alpha=self._alpha,
            k=self._rrf_k,
        )

        sorted_ranked = sorted(ranked, key=lambda r: r.fused_score, reverse=True)

        return [
            RetrievalResult(
                doc_id=r.doc_id,
                chunk_id=r.chunk_id,
                text=r.text,
                score=r.fused_score,
                method="hybrid",
            )
            for r in sorted_ranked[:top_k]
        ]


def reciprocal_rank_fusion(
    keyword_results: list[dict[str, object]],
    semantic_results: list[dict[str, object]],
    alpha: float = 0.5,
    k: int = 60,
) -> list[RankedResult]:
    """Merge two ranked lists using Reciprocal Rank Fusion.

    Args:
        keyword_results: Ranked results from BM25 (position 0 = rank 1).
        semantic_results: Ranked results from vector search (position 0 = rank 1).
        alpha: Weight for semantic results. (1 - alpha) for keyword.
        k: RRF constant to prevent domination by top ranks.

    Returns:
        List of RankedResult with fused scores (unsorted).
    """
    chunk_data: dict[str, dict[str, object]] = {}
    keyword_ranks: dict[str, int] = {}
    semantic_ranks: dict[str, int] = {}

    for rank, r in enumerate(keyword_results, start=1):
        key = f"{r['doc_id']}::{r['chunk_id']}"
        keyword_ranks[key] = rank
        chunk_data[key] = r

    for rank, r in enumerate(semantic_results, start=1):
        key = f"{r['doc_id']}::{r['chunk_id']}"
        semantic_ranks[key] = rank
        chunk_data[key] = r

    all_keys = set(keyword_ranks.keys()) | set(semantic_ranks.keys())
    results: list[RankedResult] = []

    for key in all_keys:
        kw_rank = keyword_ranks.get(key)
        sem_rank = semantic_ranks.get(key)

        kw_score = (1.0 - alpha) / (k + kw_rank) if kw_rank is not None else 0.0
        sem_score = alpha / (k + sem_rank) if sem_rank is not None else 0.0
        fused = kw_score + sem_score

        data = chunk_data[key]
        results = [
            *results,
            RankedResult(
                doc_id=str(data["doc_id"]),
                chunk_id=str(data["chunk_id"]),
                text=str(data["text"]),
                keyword_rank=kw_rank,
                semantic_rank=sem_rank,
                fused_score=fused,
            ),
        ]

    return results
