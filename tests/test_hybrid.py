"""Tests for HybridIndex and reciprocal rank fusion."""

import pytest

from src.hybrid_index import HybridIndex, RankedResult, reciprocal_rank_fusion


# --- Fake indexes for testing ---

class FakeKeywordIndex:
    def __init__(self, results: list[dict]) -> None:
        self._results = results

    def query(self, query: str, top_k: int) -> list[dict]:
        return self._results[:top_k]


class FakeSemanticIndex:
    def __init__(self, results: list[dict]) -> None:
        self._results = results

    def query(self, query: str, top_k: int) -> list[dict]:
        return self._results[:top_k]


KEYWORD_RESULTS = [
    {"doc_id": "d1", "chunk_id": "c1", "text": "keyword hit 1", "score": 5.0},
    {"doc_id": "d2", "chunk_id": "c2", "text": "keyword hit 2", "score": 4.0},
    {"doc_id": "d3", "chunk_id": "c3", "text": "keyword only", "score": 3.0},
]

SEMANTIC_RESULTS = [
    {"doc_id": "d2", "chunk_id": "c2", "text": "semantic hit 1", "score": 0.95},
    {"doc_id": "d1", "chunk_id": "c1", "text": "semantic hit 2", "score": 0.90},
    {"doc_id": "d4", "chunk_id": "c4", "text": "semantic only", "score": 0.85},
]


class TestReciprocalRankFusion:
    """Tests for the reciprocal_rank_fusion function."""

    def test_merge_two_lists(self) -> None:
        ranked = reciprocal_rank_fusion(
            keyword_results=KEYWORD_RESULTS,
            semantic_results=SEMANTIC_RESULTS,
            alpha=0.5,
            k=60,
        )
        doc_ids = {r.doc_id for r in ranked}
        assert doc_ids == {"d1", "d2", "d3", "d4"}

    def test_overlapping_docs_get_both_ranks(self) -> None:
        ranked = reciprocal_rank_fusion(
            keyword_results=KEYWORD_RESULTS,
            semantic_results=SEMANTIC_RESULTS,
            alpha=0.5,
            k=60,
        )
        d2 = next(r for r in ranked if r.doc_id == "d2")
        assert d2.keyword_rank is not None
        assert d2.semantic_rank is not None

    def test_keyword_only_doc_has_no_semantic_rank(self) -> None:
        ranked = reciprocal_rank_fusion(
            keyword_results=KEYWORD_RESULTS,
            semantic_results=SEMANTIC_RESULTS,
            alpha=0.5,
            k=60,
        )
        d3 = next(r for r in ranked if r.doc_id == "d3")
        assert d3.keyword_rank is not None
        assert d3.semantic_rank is None

    def test_semantic_only_doc_has_no_keyword_rank(self) -> None:
        ranked = reciprocal_rank_fusion(
            keyword_results=KEYWORD_RESULTS,
            semantic_results=SEMANTIC_RESULTS,
            alpha=0.5,
            k=60,
        )
        d4 = next(r for r in ranked if r.doc_id == "d4")
        assert d4.keyword_rank is None
        assert d4.semantic_rank is not None

    def test_overlapping_doc_scores_higher_than_single_source(self) -> None:
        ranked = reciprocal_rank_fusion(
            keyword_results=KEYWORD_RESULTS,
            semantic_results=SEMANTIC_RESULTS,
            alpha=0.5,
            k=60,
        )
        d2 = next(r for r in ranked if r.doc_id == "d2")
        d3 = next(r for r in ranked if r.doc_id == "d3")
        d4 = next(r for r in ranked if r.doc_id == "d4")
        assert d2.fused_score > d3.fused_score
        assert d2.fused_score > d4.fused_score

    def test_empty_keyword_results(self) -> None:
        ranked = reciprocal_rank_fusion(
            keyword_results=[],
            semantic_results=SEMANTIC_RESULTS,
            alpha=0.5,
            k=60,
        )
        assert len(ranked) == 3
        assert all(r.keyword_rank is None for r in ranked)

    def test_empty_semantic_results(self) -> None:
        ranked = reciprocal_rank_fusion(
            keyword_results=KEYWORD_RESULTS,
            semantic_results=[],
            alpha=0.5,
            k=60,
        )
        assert len(ranked) == 3
        assert all(r.semantic_rank is None for r in ranked)

    def test_ranked_result_is_frozen(self) -> None:
        ranked = reciprocal_rank_fusion(
            keyword_results=KEYWORD_RESULTS,
            semantic_results=SEMANTIC_RESULTS,
            alpha=0.5,
            k=60,
        )
        with pytest.raises(AttributeError):
            ranked[0].fused_score = 999.0  # type: ignore[misc]


class TestHybridIndex:
    """Tests for the HybridIndex class."""

    def test_alpha_zero_gives_keyword_only(self) -> None:
        hybrid = HybridIndex(
            keyword_index=FakeKeywordIndex(KEYWORD_RESULTS),
            semantic_index=FakeSemanticIndex(SEMANTIC_RESULTS),
            alpha=0.0,
        )
        results = hybrid.search("test query", top_k=10)
        # alpha=0 means semantic index is not queried
        doc_ids = {r.doc_id for r in results}
        assert doc_ids == {"d1", "d2", "d3"}

    def test_alpha_one_gives_semantic_only(self) -> None:
        hybrid = HybridIndex(
            keyword_index=FakeKeywordIndex(KEYWORD_RESULTS),
            semantic_index=FakeSemanticIndex(SEMANTIC_RESULTS),
            alpha=1.0,
        )
        results = hybrid.search("test query", top_k=10)
        # alpha=1 means keyword index is not queried
        doc_ids = {r.doc_id for r in results}
        assert doc_ids == {"d2", "d1", "d4"}

    def test_balanced_alpha_returns_all_docs(self) -> None:
        hybrid = HybridIndex(
            keyword_index=FakeKeywordIndex(KEYWORD_RESULTS),
            semantic_index=FakeSemanticIndex(SEMANTIC_RESULTS),
            alpha=0.5,
        )
        results = hybrid.search("test query", top_k=10)
        doc_ids = {r.doc_id for r in results}
        assert doc_ids == {"d1", "d2", "d3", "d4"}

    def test_top_k_limits_results(self) -> None:
        hybrid = HybridIndex(
            keyword_index=FakeKeywordIndex(KEYWORD_RESULTS),
            semantic_index=FakeSemanticIndex(SEMANTIC_RESULTS),
            alpha=0.5,
        )
        results = hybrid.search("test query", top_k=2)
        assert len(results) == 2

    def test_results_sorted_by_fused_score(self) -> None:
        hybrid = HybridIndex(
            keyword_index=FakeKeywordIndex(KEYWORD_RESULTS),
            semantic_index=FakeSemanticIndex(SEMANTIC_RESULTS),
            alpha=0.5,
        )
        results = hybrid.search("test query", top_k=10)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_all_results_have_hybrid_method(self) -> None:
        hybrid = HybridIndex(
            keyword_index=FakeKeywordIndex(KEYWORD_RESULTS),
            semantic_index=FakeSemanticIndex(SEMANTIC_RESULTS),
            alpha=0.5,
        )
        results = hybrid.search("test query", top_k=10)
        assert all(r.method == "hybrid" for r in results)

    def test_invalid_alpha_raises(self) -> None:
        with pytest.raises(ValueError, match="alpha"):
            HybridIndex(
                keyword_index=FakeKeywordIndex([]),
                semantic_index=FakeSemanticIndex([]),
                alpha=1.5,
            )

    def test_invalid_top_k_raises(self) -> None:
        hybrid = HybridIndex(
            keyword_index=FakeKeywordIndex([]),
            semantic_index=FakeSemanticIndex([]),
            alpha=0.5,
        )
        with pytest.raises(ValueError, match="top_k"):
            hybrid.search("test", top_k=0)
