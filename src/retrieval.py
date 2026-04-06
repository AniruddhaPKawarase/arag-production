"""Three retrieval tool classes for A-RAG hierarchical retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


VALID_METHODS = frozenset({"keyword", "semantic", "chunk_read", "hybrid"})


@dataclass(frozen=True)
class RetrievalResult:
    """Immutable result from any retrieval tool."""

    doc_id: str
    chunk_id: str
    text: str
    score: float
    method: str

    def __post_init__(self) -> None:
        if self.method not in VALID_METHODS:
            raise ValueError(
                f"Invalid method '{self.method}'. Must be one of: {sorted(VALID_METHODS)}"
            )
        if not self.doc_id:
            raise ValueError("doc_id must not be empty")
        if not self.chunk_id:
            raise ValueError("chunk_id must not be empty")


class SearchIndex(Protocol):
    """Protocol for any search index backend."""

    def query(self, query: str, top_k: int) -> list[dict[str, Any]]: ...


class DocumentStore(Protocol):
    """Protocol for any document chunk store."""

    def get_chunks(
        self, doc_id: str, start: int, end: int
    ) -> list[dict[str, Any]]: ...


class KeywordSearchTool:
    """BM25-based keyword search tool."""

    def __init__(self, index: SearchIndex) -> None:
        self._index = index

    def search(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        """Search using BM25 keyword matching.

        Args:
            query: The search query string.
            top_k: Maximum number of results to return.

        Returns:
            List of RetrievalResult ordered by BM25 score descending.
        """
        if top_k < 1:
            raise ValueError("top_k must be >= 1")

        raw_results = self._index.query(query, top_k)
        return [
            RetrievalResult(
                doc_id=r["doc_id"],
                chunk_id=r["chunk_id"],
                text=r["text"],
                score=float(r["score"]),
                method="keyword",
            )
            for r in raw_results
        ]


class SemanticSearchTool:
    """Embedding-based vector search tool."""

    def __init__(self, index: SearchIndex) -> None:
        self._index = index

    def search(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        """Search using embedding similarity.

        Args:
            query: The search query string.
            top_k: Maximum number of results to return.

        Returns:
            List of RetrievalResult ordered by cosine similarity descending.
        """
        if top_k < 1:
            raise ValueError("top_k must be >= 1")

        raw_results = self._index.query(query, top_k)
        return [
            RetrievalResult(
                doc_id=r["doc_id"],
                chunk_id=r["chunk_id"],
                text=r["text"],
                score=float(r["score"]),
                method="semantic",
            )
            for r in raw_results
        ]


class ChunkReadTool:
    """Direct document chunk access tool."""

    def __init__(self, store: DocumentStore) -> None:
        self._store = store

    def read(
        self, doc_id: str, start_chunk: int = 0, end_chunk: int = 5
    ) -> list[RetrievalResult]:
        """Read specific chunks from a document by position.

        Args:
            doc_id: Document identifier.
            start_chunk: Starting chunk index (inclusive).
            end_chunk: Ending chunk index (exclusive).

        Returns:
            List of RetrievalResult for the requested chunk range.
        """
        if start_chunk < 0:
            raise ValueError("start_chunk must be >= 0")
        if end_chunk <= start_chunk:
            raise ValueError("end_chunk must be > start_chunk")

        raw_chunks = self._store.get_chunks(doc_id, start_chunk, end_chunk)
        return [
            RetrievalResult(
                doc_id=c["doc_id"],
                chunk_id=c["chunk_id"],
                text=c["text"],
                score=1.0,
                method="chunk_read",
            )
            for c in raw_chunks
        ]
