import time
from typing import Dict, Any
from qdrant_client import QdrantClient
from app.rag.embedder import BGEEmbedder
from app.rag.generator import GroqGenerator
import os

class RAGPipeline:
    def __init__(self):
        self.embedder = BGEEmbedder()
        # Initialize Qdrant assuming we already built it
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        
        if qdrant_url and qdrant_api_key:
            self.qdrant = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        else:
            db_path = "data/qdrant_db"
            if not os.path.exists(db_path):
                os.makedirs(db_path, exist_ok=True)
            self.qdrant = QdrantClient(path=db_path)
            
        self.generator = GroqGenerator()
        self.collection_name = "rag_docs"
        
        # --- DEPRECATED/PAUSED: Local BM25 Sparse Indexing ---
        # Note: We have paused using the local BM25s index because we successfully scaled
        # to a 15k+ document corpus uploaded to Qdrant Cloud via Colab. Keeping a local
        # sparse index for 15k+ docs is memory intensive and out-of-sync with the cloud.
        # Instead, we now rely entirely on Dense Search (BGE-M3) paired with an elite 
        # Cross-Encoder (Cohere), which handles semantic matching beautifully without BM25.
        # The code is kept here for reference if we ever move to Qdrant Native Sparse Vectors.
        
        # from app.rag.sparse import BM25Index
        # self.bm25 = BM25Index()
        # self.bm25.load()
        # 
        # from app.rag.retriever import HybridRetriever
        # self.retriever = HybridRetriever(self.embedder, self.qdrant, self.bm25)
        self.retriever = None
        
        from app.rag.reranker import Reranker
        self.reranker = Reranker()
        
        from app.rag.compressor import ContextCompressor
        self.compressor = ContextCompressor()
        
        from app.rag.rewriter import QueryRewriter
        self.rewriter = QueryRewriter()
        
    def query(self, query_text: str, k: int = 5, use_hybrid: bool = True, use_reranker: bool = True, use_compression: bool = False, use_hyde: bool = False, use_multi_query: bool = False) -> Dict[str, Any]:
        start_time = time.time()
        trace = []  # Pipeline trace for UI visualization
        
        # 0. Query Rewriting (HyDE + Multi-Query) — runs on Groq, zero local compute
        all_query_texts = [query_text]
        hyde_doc = ""
        
        if use_hyde or use_multi_query:
            t0 = time.time()
            rewrite_result = self.rewriter.rewrite(query_text, use_hyde=use_hyde, use_multi_query=use_multi_query)
            rewrite_ms = int((time.time() - t0) * 1000)
            
            hyde_doc = rewrite_result.get("hyde_doc", "")
            alt_queries = rewrite_result.get("alt_queries", [])
            
            if alt_queries:
                all_query_texts.extend(alt_queries)
            
            desc_parts = []
            if use_hyde and hyde_doc:
                desc_parts.append(f"HyDE doc generated ({len(hyde_doc)} chars)")
            if use_multi_query and alt_queries:
                desc_parts.append(f"{len(alt_queries)} alternative queries")
            
            trace.append({
                "step": "Query Rewriting",
                "description": f"{' + '.join(desc_parts)} via Groq Llama-3.1",
                "duration_ms": rewrite_ms,
                "count": len(all_query_texts),
                "docs": [{"title": q} for q in all_query_texts] + ([{"title": f"[HyDE] {hyde_doc[:80]}..."}] if hyde_doc else []),
                "icon": "rewrite"
            })
        
        # 1. Retrieve — embed all query variants + HyDE doc, merge results
        if use_hybrid:
            print("Warning: Hybrid retrieval is currently paused. Falling back to Pure Dense Retrieval.")
            use_hybrid = False
            
        fetch_k = k * 3 if use_reranker else k
        
        # Build list of texts to embed: all queries + HyDE doc (if available)
        texts_to_embed = list(all_query_texts)
        if hyde_doc:
            texts_to_embed.append(hyde_doc)
        
        t0 = time.time()
        all_vectors = self.embedder.encode(texts_to_embed)
        embed_ms = int((time.time() - t0) * 1000)
        
        # Retrieve for each vector and merge (deduplicate by title)
        t0 = time.time()
        seen_titles = set()
        retrieved_docs = []
        per_query_k = max(fetch_k // len(all_vectors), 5)
        
        for vec in all_vectors:
            try:
                hits = self.qdrant.query_points(
                    collection_name=self.collection_name, query=vec.tolist(), limit=per_query_k
                ).points
                for h in hits:
                    title = h.payload.get("metadata", {}).get("title", "")
                    if title not in seen_titles:
                        seen_titles.add(title)
                        retrieved_docs.append(h.payload)
            except Exception as e:
                print(f"Qdrant Query Error: {e}")
        retrieve_ms = int((time.time() - t0) * 1000)
        
        embed_desc = f"Encoded {len(texts_to_embed)} text(s) into 1024-dim vectors using BGE-M3"
        trace.append({
            "step": "Embedding",
            "description": embed_desc,
            "duration_ms": embed_ms,
            "count": len(texts_to_embed),
            "icon": "embed"
        })
        trace.append({
            "step": "Dense Retrieval",
            "description": f"Retrieved {len(retrieved_docs)} unique candidates from Qdrant Cloud ({len(all_vectors)} searches × {per_query_k} each)",
            "duration_ms": retrieve_ms,
            "count": len(retrieved_docs),
            "docs": [{"title": d.get("metadata", {}).get("title", "Untitled")} for d in retrieved_docs],
            "icon": "search"
        })
        
        # 2. Rerank
        if use_reranker and retrieved_docs:
            t0 = time.time()
            final_docs = self.reranker.rerank(query_text, retrieved_docs, top_k=k)
            rerank_ms = int((time.time() - t0) * 1000)
            
            reranker_mode = getattr(self.reranker, 'mode', 'local')
            trace.append({
                "step": "Reranking",
                "description": f"Reranked {len(retrieved_docs)} → {len(final_docs)} docs via {'Cohere API' if reranker_mode == 'cohere' else 'Local BGE CrossEncoder'}",
                "duration_ms": rerank_ms,
                "count": len(final_docs),
                "docs": [{"title": d.get("metadata", {}).get("title", "Untitled")} for d in final_docs],
                "icon": "rerank"
            })
        else:
            final_docs = retrieved_docs[:k]
            trace.append({
                "step": "Reranking",
                "description": "Skipped (disabled by user)",
                "duration_ms": 0,
                "count": len(final_docs),
                "icon": "rerank"
            })
            
        sources = []
        contexts = []
        for doc in final_docs:
            sources.append(doc.get("metadata", {}))
            contexts.append(doc.get("text", ""))
            
        # 3. Compress Contexts
        if use_compression and contexts:
            t0 = time.time()
            original_len = sum(len(c) for c in contexts)
            contexts = self.compressor.compress(query_text, contexts)
            compressed_len = sum(len(c) for c in contexts)
            compress_ms = int((time.time() - t0) * 1000)
            trace.append({
                "step": "Compression",
                "description": f"Compressed {original_len} → {compressed_len} chars ({int((1 - compressed_len/max(original_len,1))*100)}% reduction)",
                "duration_ms": compress_ms,
                "count": len(contexts),
                "icon": "compress"
            })
        else:
            trace.append({
                "step": "Compression",
                "description": "Skipped (disabled by user)",
                "duration_ms": 0,
                "count": len(contexts),
                "icon": "compress"
            })
            
        # 4. Generate
        t0 = time.time()
        answer = self.generator.generate(query_text, contexts)
        generate_ms = int((time.time() - t0) * 1000)
        
        trace.append({
            "step": "LLM Generation",
            "description": f"Synthesized answer from {len(contexts)} contexts via Groq Llama-3.1",
            "duration_ms": generate_ms,
            "count": 1,
            "icon": "generate"
        })
        
        latency_ms = int((time.time() - start_time) * 1000)
        
        return {
            "answer": answer,
            "sources": sources,
            "latency_ms": latency_ms,
            "trace": trace,
            "raw_contexts": contexts # Useful for evals
        }
