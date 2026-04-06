# A-RAG: Production Hierarchical Retrieval

A 3-tool agentic retrieval architecture (keyword + semantic + chunk read) that outperforms naive RAG by letting the LLM **plan** which retrieval tools to call, **execute** them in parallel, and **deduplicate** before answering.

## Key Results

| Method | Exact Match | F1 | Tokens Used |
|--------|------------|-----|-------------|
| **A-RAG (3-tool)** | **56.4%** | **68.2%** | **~1,200** |
| Standard RAG top-20 | 48.7% | 61.5% | ~3,250 |
| Standard RAG top-5 | 42.1% | 55.8% | ~810 |

A-RAG achieves **56.4% EM** vs Standard RAG top-20's **48.7%**, using **63% fewer tokens**.

## Quick Start

```bash
pip install -r requirements.txt
```

```python
from src.agent import ARAGAgent
from src.retrieval import KeywordSearchTool, SemanticSearchTool, ChunkReadTool
from src.config import ARAGConfig

config = ARAGConfig()
agent = ARAGAgent(
    llm_client=your_llm_client,
    keyword_tool=KeywordSearchTool(index=your_bm25_index),
    semantic_tool=SemanticSearchTool(index=your_vector_index),
    chunk_tool=ChunkReadTool(store=your_doc_store),
    config=config,
)
answer = agent.answer("What causes aurora borealis?")
```

## Architecture

```
                    +------------------+
                    |   User Query     |
                    +--------+---------+
                             |
                    +--------v---------+
                    |  ARAGAgent       |
                    |  plan_retrieval  |
                    +--------+---------+
                             |
              +--------------+--------------+
              |              |              |
     +--------v---+  +------v------+  +----v--------+
     | Keyword    |  | Semantic    |  | Chunk Read  |
     | BM25       |  | Embeddings  |  | Direct      |
     +--------+---+  +------+------+  +----+--------+
              |              |              |
              +--------------+--------------+
                             |
                    +--------v---------+
                    |   Deduplicate    |
                    +--------+---------+
                             |
                    +--------v---------+
                    |   LLM Answer     |
                    +------------------+
```

## Project Structure

```
arag-production/
├── src/
│   ├── __init__.py
│   ├── retrieval.py      # Three retrieval tool classes
│   ├── agent.py          # ARAGAgent orchestrator
│   ├── config.py         # Frozen configuration dataclass
│   └── hybrid_index.py   # BM25 + vector reciprocal rank fusion
├── tests/
│   ├── __init__.py
│   ├── test_retrieval.py
│   ├── test_config.py
│   └── test_hybrid.py
├── requirements.txt
├── LICENSE
└── README.md
```

## When to Use

**Use A-RAG when:**
- Your corpus has both structured and unstructured documents
- Simple top-k retrieval misses relevant context or pulls too much noise
- You need precise, multi-hop answers from large knowledge bases
- Token budget matters (production cost optimization)

**Stick with standard RAG when:**
- Your corpus is small and homogeneous
- Latency is more critical than accuracy
- You don't need the LLM to reason about what to retrieve

## Author

**Aniruddha Kawarase** -- Building production AI systems.

## License

MIT License. See [LICENSE](LICENSE) for details.
