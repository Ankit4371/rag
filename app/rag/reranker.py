from sentence_transformers import CrossEncoder
from typing import List, Dict, Any

class BGEReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", device: str = "cpu"):
        self.model = CrossEncoder(model_name, device=device)

    def rerank(self, query: str, docs: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        if not docs:
            return []
            
        pairs = [[query, doc.get("text", "")] for doc in docs]
        
        scores = self.model.predict(pairs)
        
        # Zip scores with docs and sort
        scored_docs = list(zip(docs, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        return [doc for doc, score in scored_docs[:top_k]]
