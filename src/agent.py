"""ARAGAgent: Orchestrator for agentic hierarchical retrieval."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol

from src.config import ARAGConfig
from src.retrieval import ChunkReadTool, KeywordSearchTool, RetrievalResult, SemanticSearchTool


class LLMClient(Protocol):
    """Protocol for any LLM client."""

    def generate(self, prompt: str) -> str: ...


@dataclass(frozen=True)
class RetrievalPlan:
    """Immutable plan describing which tools to call and with what parameters."""

    use_keyword: bool
    use_semantic: bool
    chunk_reads: tuple[tuple[str, int, int], ...]  # (doc_id, start, end)


@dataclass(frozen=True)
class ARAGAnswer:
    """Immutable final answer from the A-RAG pipeline."""

    query: str
    answer: str
    sources: tuple[RetrievalResult, ...]
    token_estimate: int


class ARAGAgent:
    """Orchestrates hierarchical retrieval using three tools and an LLM.

    The agent follows a plan-execute-deduplicate-answer pipeline:
    1. plan_retrieval: Ask the LLM which tools to invoke
    2. execute_plan: Run the selected retrieval tools
    3. deduplicate: Remove duplicate chunks
    4. answer: Generate the final answer from unique context
    """

    def __init__(
        self,
        llm_client: LLMClient,
        keyword_tool: KeywordSearchTool,
        semantic_tool: SemanticSearchTool,
        chunk_tool: ChunkReadTool,
        config: ARAGConfig | None = None,
    ) -> None:
        self._llm = llm_client
        self._keyword = keyword_tool
        self._semantic = semantic_tool
        self._chunk = chunk_tool
        self._config = config or ARAGConfig()

    @property
    def config(self) -> ARAGConfig:
        return self._config

    def plan_retrieval(self, query: str) -> RetrievalPlan:
        """Ask the LLM to decide which retrieval tools to call.

        Args:
            query: The user's question.

        Returns:
            A RetrievalPlan describing the tools and parameters.
        """
        prompt = (
            f"Given this question, decide which retrieval tools to use.\n"
            f"Available tools: keyword_search, semantic_search, chunk_read\n"
            f"Question: {query}\n"
            f"Respond with a JSON object: "
            f'{{"use_keyword": bool, "use_semantic": bool, "chunk_reads": [[doc_id, start, end], ...]}}'
        )
        response = self._llm.generate(prompt)
        return _parse_plan(response)

    def execute_plan(
        self, query: str, plan: RetrievalPlan
    ) -> list[RetrievalResult]:
        """Execute the retrieval plan by calling the selected tools.

        Args:
            query: The user's question.
            plan: The RetrievalPlan from plan_retrieval.

        Returns:
            Combined list of RetrievalResult from all tools.
        """
        results: list[RetrievalResult] = []

        if plan.use_keyword:
            results = [
                *results,
                *self._keyword.search(query, self._config.top_k_keyword),
            ]

        if plan.use_semantic:
            results = [
                *results,
                *self._semantic.search(query, self._config.top_k_semantic),
            ]

        if self._config.chunk_read_enabled:
            for doc_id, start, end in plan.chunk_reads:
                results = [
                    *results,
                    *self._chunk.read(doc_id, start, end),
                ]

        return results

    @staticmethod
    def deduplicate(results: list[RetrievalResult]) -> list[RetrievalResult]:
        """Remove duplicate chunks, keeping the highest-scored version.

        Args:
            results: List of RetrievalResult, possibly with duplicates.

        Returns:
            Deduplicated list, preserving order of first occurrence.
        """
        seen: dict[str, RetrievalResult] = {}
        for r in results:
            key = f"{r.doc_id}::{r.chunk_id}"
            existing = seen.get(key)
            if existing is None or r.score > existing.score:
                seen[key] = r
        return list(seen.values())

    def answer(self, query: str) -> ARAGAnswer:
        """Full A-RAG pipeline: plan -> execute -> deduplicate -> answer.

        Args:
            query: The user's question.

        Returns:
            An ARAGAnswer with the final answer and sources.
        """
        plan = self.plan_retrieval(query)
        raw_results = self.execute_plan(query, plan)
        unique_results = self.deduplicate(raw_results)

        context = "\n\n".join(
            f"[{r.method}] {r.doc_id}/{r.chunk_id}: {r.text}"
            for r in unique_results
        )
        prompt = (
            f"Answer the question using ONLY the context below.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n"
            f"Answer:"
        )
        answer_text = self._llm.generate(prompt)
        token_estimate = len(context.split())

        return ARAGAnswer(
            query=query,
            answer=answer_text,
            sources=tuple(unique_results),
            token_estimate=token_estimate,
        )


def _parse_plan(response: str) -> RetrievalPlan:
    """Parse LLM response into a RetrievalPlan.

    This is a simplified parser. In production, use structured output
    or a more robust JSON parser with error handling.
    """
    try:
        data: dict[str, Any] = json.loads(response)
    except json.JSONDecodeError:
        return RetrievalPlan(
            use_keyword=True,
            use_semantic=True,
            chunk_reads=(),
        )

    chunk_reads = tuple(
        (str(c[0]), int(c[1]), int(c[2]))
        for c in data.get("chunk_reads", [])
    )
    return RetrievalPlan(
        use_keyword=bool(data.get("use_keyword", True)),
        use_semantic=bool(data.get("use_semantic", True)),
        chunk_reads=chunk_reads,
    )
