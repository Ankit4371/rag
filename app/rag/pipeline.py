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
        
        from app.rag.compressor import ContextCompressor
        self.compressor = ContextCompressor()
        
    def query(self, query_text: str, k: int = 5, use_hybrid: bool = True, use_reranker: bool = True, use_compression: bool = False) -> Dict[str, Any]:
        start_time = time.time()
        
        # 1. Retrieve
        if use_hybrid:
            fetch_k = k * 3 if use_reranker else k
            retrieved_docs = self.retriever.retrieve(query_text, top_k=fetch_k)
        else:
            fetch_k = k * 3 if use_reranker else k
            query_vector = self.embedder.encode([query_text])[0]
            try:
                hits = self.qdrant.query_points(
                    collection_name=self.collection_name, query=query_vector.tolist(), limit=fetch_k
                ).points
                retrieved_docs = [h.payload for h in hits]
            except Exception:
                retrieved_docs = []
        
        # 2. Rerank
        if use_reranker and retrieved_docs:
            final_docs = self.reranker.rerank(query_text, retrieved_docs, top_k=k)
        else:
            final_docs = retrieved_docs[:k]
            
        sources = []
        contexts = []
        for doc in final_docs:
            sources.append(doc.get("metadata", {}))
            contexts.append(doc.get("text", ""))
            
        # 3. Compress Contexts
        if use_compression and contexts:
            contexts = self.compressor.compress(query_text, contexts)
            
        # 4. Generate
        answer = self.generator.generate(query_text, contexts)
        
        latency_ms = int((time.time() - start_time) * 1000)
        
        return {
            "answer": answer,
            "sources": sources,
            "latency_ms": latency_ms,
            "raw_contexts": contexts # Useful for evals
        }
