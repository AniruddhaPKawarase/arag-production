"""Tests for retrieval tools and RetrievalResult."""

import pytest

from src.retrieval import RetrievalResult


class TestRetrievalResult:
    """Tests for the RetrievalResult frozen dataclass."""

    def test_create_keyword_result(self) -> None:
        result = RetrievalResult(
            doc_id="doc-1",
            chunk_id="chunk-3",
            text="Solar winds interact with magnetosphere.",
            score=0.85,
            method="keyword",
        )
        assert result.doc_id == "doc-1"
        assert result.chunk_id == "chunk-3"
        assert result.score == 0.85
        assert result.method == "keyword"

    def test_create_semantic_result(self) -> None:
        result = RetrievalResult(
            doc_id="doc-2",
            chunk_id="chunk-1",
            text="Charged particles from the sun.",
            score=0.92,
            method="semantic",
        )
        assert result.method == "semantic"
        assert result.score == 0.92

    def test_create_chunk_read_result(self) -> None:
        result = RetrievalResult(
            doc_id="doc-1",
            chunk_id="chunk-0",
            text="Introduction to aurora borealis.",
            score=1.0,
            method="chunk_read",
        )
        assert result.method == "chunk_read"
        assert result.score == 1.0

    def test_create_hybrid_result(self) -> None:
        result = RetrievalResult(
            doc_id="doc-1",
            chunk_id="chunk-0",
            text="Hybrid search result.",
            score=0.5,
            method="hybrid",
        )
        assert result.method == "hybrid"

    def test_immutability_doc_id(self) -> None:
        result = RetrievalResult(
            doc_id="doc-1",
            chunk_id="chunk-0",
            text="Some text.",
            score=0.5,
            method="keyword",
        )
        with pytest.raises(AttributeError):
            result.doc_id = "doc-2"  # type: ignore[misc]

    def test_immutability_score(self) -> None:
        result = RetrievalResult(
            doc_id="doc-1",
            chunk_id="chunk-0",
            text="Some text.",
            score=0.5,
            method="keyword",
        )
        with pytest.raises(AttributeError):
            result.score = 0.99  # type: ignore[misc]

    def test_immutability_method(self) -> None:
        result = RetrievalResult(
            doc_id="doc-1",
            chunk_id="chunk-0",
            text="Some text.",
            score=0.5,
            method="keyword",
        )
        with pytest.raises(AttributeError):
            result.method = "semantic"  # type: ignore[misc]

    def test_invalid_method_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid method"):
            RetrievalResult(
                doc_id="doc-1",
                chunk_id="chunk-0",
                text="Some text.",
                score=0.5,
                method="invalid_method",
            )

    def test_empty_doc_id_raises(self) -> None:
        with pytest.raises(ValueError, match="doc_id must not be empty"):
            RetrievalResult(
                doc_id="",
                chunk_id="chunk-0",
                text="Some text.",
                score=0.5,
                method="keyword",
            )

    def test_empty_chunk_id_raises(self) -> None:
        with pytest.raises(ValueError, match="chunk_id must not be empty"):
            RetrievalResult(
                doc_id="doc-1",
                chunk_id="",
                text="Some text.",
                score=0.5,
                method="keyword",
            )
