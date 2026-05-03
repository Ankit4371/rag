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
        db_path = "data/qdrant_db"
        if not os.path.exists(db_path):
            os.makedirs(db_path, exist_ok=True)
            
        self.qdrant = QdrantClient(path=db_path)
        self.generator = GroqGenerator()
        self.collection_name = "rag_docs"
        
        from app.rag.sparse import BM25Index
        self.bm25 = BM25Index()
        self.bm25.load()
        
        from app.rag.retriever import HybridRetriever
        self.retriever = HybridRetriever(self.embedder, self.qdrant, self.bm25)
        
        from app.rag.reranker import BGEReranker
        self.reranker = BGEReranker()
        
    def query(self, query_text: str, k: int = 5) -> Dict[str, Any]:
        start_time = time.time()
        
        # 1. Retrieve using Hybrid RRF (Fetch 3x docs for reranking)
        retrieved_docs = self.retriever.retrieve(query_text, top_k=k * 3)
        
        # 2. Rerank down to top_k
        reranked_docs = self.reranker.rerank(query_text, retrieved_docs, top_k=k)
        
        sources = []
        contexts = []
        for doc in reranked_docs:
            sources.append(doc.get("metadata", {}))
            contexts.append(doc.get("text", ""))
            
        # 3. Generate
        answer = self.generator.generate(query_text, contexts)
        
        latency_ms = int((time.time() - start_time) * 1000)
        
        return {
            "answer": answer,
            "sources": sources,
            "latency_ms": latency_ms,
            "raw_contexts": contexts # Useful for evals
        }
