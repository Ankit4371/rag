# Production RAG — ArXiv Research Explorer

A production-grade **Retrieval-Augmented Generation** system built from scratch without LangChain or LlamaIndex. Queries 15,000+ ML/NLP/AI research papers from ArXiv with sub-2-second latency.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688)
![Qdrant](https://img.shields.io/badge/Qdrant-Cloud-red)

## Architecture

```
Query → BGE-M3 Embedding → Qdrant Cloud (Dense Search)
      → Cohere Reranker (Cross-Encoder)
      → [Optional] LLMLingua Compression
      → Groq Llama-3.1 (Generation)
      → Structured Response + Source Attribution
```

### Key Design Decisions

| Component | Choice | Rationale |
|---|---|---|
| **Embedder** | `BAAI/bge-m3` (1024d) | State-of-the-art multilingual dense embeddings |
| **Vector DB** | Qdrant Cloud | Scalable managed vector search, zero local infra |
| **Reranker** | Cohere `rerank-v3` API | Offloaded to cloud for zero local GPU requirement; auto-fallback to local `bge-reranker-v2-m3` |
| **LLM** | Groq `llama-3.1-8b-instant` | Free tier, ~200ms generation latency |
| **Compression** | LLMLingua-2 (optional) | Toggle-able context compression for token savings |
| **Ingestion** | Google Colab + T4 GPU | 15k+ documents embedded and pushed to Qdrant Cloud without straining local resources |
| **Sparse Index** | BM25S (paused) | Code retained; replaced by Dense+Reranker for cloud-scale corpus |

## Features

- **Modular Pipeline**: Toggle Reranker and Compression on/off from the UI
- **Pipeline Trace**: Optional step-by-step visualization showing timing at each stage (Embed → Retrieve → Rerank → Generate)
- **Cloud-Native**: All heavy compute offloaded — Qdrant Cloud for search, Cohere for reranking, Groq for generation
- **Source Attribution**: Every answer links back to specific ArXiv papers
- **Dark-Themed Glassmorphic UI**: Premium web interface built with vanilla HTML/CSS/JS

## Quick Start

```bash
# 1. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
make install

# 3. Configure environment
cp .env.template .env
# Fill in: GROQ_API_KEY, QDRANT_URL, QDRANT_API_KEY, COHERE_API_KEY

# 4. Run the server
make run

# 5. Open http://localhost:8000
```

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | ✅ | [Groq Console](https://console.groq.com/) — free tier |
| `QDRANT_URL` | ✅ | [Qdrant Cloud](https://cloud.qdrant.io/) cluster URL |
| `QDRANT_API_KEY` | ✅ | Qdrant Cloud API key |
| `COHERE_API_KEY` | Optional | [Cohere Dashboard](https://dashboard.cohere.com/) — enables cloud reranking (falls back to local model if absent) |

## Project Structure

```
app/
├── main.py              # FastAPI app, /query and /health endpoints
├── ingest/
│   ├── loaders.py       # ArxivLoader — streaming from HuggingFace
│   ├── chunker.py       # RecursiveChunker — 512-token sliding window
│   └── build_indices.py # CLI for building dense + sparse indices
└── rag/
    ├── pipeline.py      # RAGPipeline — orchestrates the full flow
    ├── embedder.py      # BGEEmbedder — BAAI/bge-m3 encoding
    ├── generator.py     # GroqGenerator — Llama-3.1 via Groq API
    ├── reranker.py      # Reranker — Cohere API or local BGE CrossEncoder
    ├── compressor.py    # ContextCompressor — LLMLingua-2
    ├── retriever.py     # HybridRetriever — Dense+BM25 via RRF (paused)
    └── sparse.py        # BM25Index — BM25S build/save/load (paused)
static/
├── index.html           # Web UI
└── styles.css           # Dark-themed glassmorphic styles
tests/                   # pytest unit & integration tests
evals/                   # RAGAS evaluation framework
```

## Pipeline Trace

When enabled via the UI toggle, every query returns a detailed breakdown:

| Step | Metrics |
|---|---|
| **Embedding** | Time to encode query to 1024-dim vector |
| **Dense Retrieval** | Candidates retrieved from Qdrant Cloud |
| **Reranking** | Documents reranked via Cohere/local model |
| **Compression** | Character reduction stats (if enabled) |
| **LLM Generation** | Groq synthesis time |

## Data Ingestion (Google Colab)

The 15k+ document corpus was embedded on a **free T4 GPU** in Google Colab and pushed directly to Qdrant Cloud. This approach:
- Avoids straining local CPU/RAM
- Leverages GPU for fast BGE-M3 encoding
- Scales to 100k+ documents without local storage

## License

MIT