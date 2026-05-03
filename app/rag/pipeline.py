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
        
    def query(self, query_text: str, k: int = 5) -> Dict[str, Any]:
        start_time = time.time()
        
        # 1. Retrieve
        query_vector = self.embedder.encode([query_text])[0]
        try:
            search_result = self.qdrant.query_points(
                collection_name=self.collection_name,
                query=query_vector.tolist(),
                limit=k
            ).points
        except Exception as e:
            search_result = []
            print(f"Retrieval error: {e}")
        
        sources = []
        contexts = []
        for hit in search_result:
            payload = hit.payload or {}
            sources.append(payload.get("metadata", {}))
            contexts.append(payload.get("text", ""))
            
        # 2. Generate
        answer = self.generator.generate(query_text, contexts)
        
        latency_ms = int((time.time() - start_time) * 1000)
        
        return {
            "answer": answer,
            "sources": sources,
            "latency_ms": latency_ms,
            "raw_contexts": contexts # Useful for evals
        }
