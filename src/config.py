"""Frozen configuration dataclass for A-RAG."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ARAGConfig:
    """Immutable configuration for A-RAG agent.

    Attributes:
        keyword_weight: Weight for keyword (BM25) results in fusion. Range [0, 1].
        semantic_weight: Weight for semantic results in fusion. Range [0, 1].
        chunk_read_enabled: Whether the chunk-read tool is available to the agent.
        top_k_keyword: Number of results to retrieve from keyword search.
        top_k_semantic: Number of results to retrieve from semantic search.
        max_chunks: Maximum number of chunks to read via ChunkReadTool.
    """

    keyword_weight: float = 0.4
    semantic_weight: float = 0.6
    chunk_read_enabled: bool = True
    top_k_keyword: int = 10
    top_k_semantic: int = 10
    max_chunks: int = 5

    def __post_init__(self) -> None:
        if not (0.0 <= self.keyword_weight <= 1.0):
            raise ValueError(
                f"keyword_weight must be in [0, 1], got {self.keyword_weight}"
            )
        if not (0.0 <= self.semantic_weight <= 1.0):
            raise ValueError(
                f"semantic_weight must be in [0, 1], got {self.semantic_weight}"
            )
        if self.top_k_keyword < 1:
            raise ValueError(
                f"top_k_keyword must be >= 1, got {self.top_k_keyword}"
            )
        if self.top_k_semantic < 1:
            raise ValueError(
                f"top_k_semantic must be >= 1, got {self.top_k_semantic}"
            )
        if self.max_chunks < 1:
            raise ValueError(
                f"max_chunks must be >= 1, got {self.max_chunks}"
            )
