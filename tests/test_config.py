"""Tests for ARAGConfig frozen dataclass."""

import pytest

from src.config import ARAGConfig


class TestARAGConfig:
    """Tests for ARAGConfig validation and immutability."""

    def test_default_config_valid(self) -> None:
        config = ARAGConfig()
        assert config.keyword_weight == 0.4
        assert config.semantic_weight == 0.6
        assert config.chunk_read_enabled is True
        assert config.top_k_keyword == 10
        assert config.top_k_semantic == 10
        assert config.max_chunks == 5

    def test_custom_config(self) -> None:
        config = ARAGConfig(
            keyword_weight=0.3,
            semantic_weight=0.7,
            chunk_read_enabled=False,
            top_k_keyword=5,
            top_k_semantic=20,
            max_chunks=10,
        )
        assert config.keyword_weight == 0.3
        assert config.top_k_semantic == 20

    def test_invalid_keyword_weight_negative(self) -> None:
        with pytest.raises(ValueError, match="keyword_weight"):
            ARAGConfig(keyword_weight=-0.1)

    def test_invalid_keyword_weight_above_one(self) -> None:
        with pytest.raises(ValueError, match="keyword_weight"):
            ARAGConfig(keyword_weight=1.5)

    def test_invalid_semantic_weight(self) -> None:
        with pytest.raises(ValueError, match="semantic_weight"):
            ARAGConfig(semantic_weight=-0.1)

    def test_invalid_top_k_keyword_zero(self) -> None:
        with pytest.raises(ValueError, match="top_k_keyword"):
            ARAGConfig(top_k_keyword=0)

    def test_invalid_top_k_keyword_negative(self) -> None:
        with pytest.raises(ValueError, match="top_k_keyword"):
            ARAGConfig(top_k_keyword=-5)

    def test_invalid_top_k_semantic(self) -> None:
        with pytest.raises(ValueError, match="top_k_semantic"):
            ARAGConfig(top_k_semantic=0)

    def test_invalid_max_chunks(self) -> None:
        with pytest.raises(ValueError, match="max_chunks"):
            ARAGConfig(max_chunks=0)

    def test_immutability_weight(self) -> None:
        config = ARAGConfig()
        with pytest.raises(AttributeError):
            config.keyword_weight = 0.9  # type: ignore[misc]

    def test_immutability_top_k(self) -> None:
        config = ARAGConfig()
        with pytest.raises(AttributeError):
            config.top_k_keyword = 100  # type: ignore[misc]

    def test_immutability_chunk_enabled(self) -> None:
        config = ARAGConfig()
        with pytest.raises(AttributeError):
            config.chunk_read_enabled = False  # type: ignore[misc]

    def test_boundary_weights(self) -> None:
        """Weights at 0.0 and 1.0 should be valid."""
        config_zero = ARAGConfig(keyword_weight=0.0, semantic_weight=0.0)
        assert config_zero.keyword_weight == 0.0

        config_one = ARAGConfig(keyword_weight=1.0, semantic_weight=1.0)
        assert config_one.keyword_weight == 1.0
