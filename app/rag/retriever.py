from typing import List, Dict, Any
from app.rag.embedder import BGEEmbedder
from app.rag.sparse import BM25Index
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchAny

class HybridRetriever:
    def __init__(self, embedder: BGEEmbedder, qdrant: QdrantClient, bm25: BM25Index):
        self.embedder = embedder
        self.qdrant = qdrant
        self.bm25 = bm25
        self.collection_name = "rag_docs"
        
    def _rrf(self, dense_results: List[Dict[str, Any]], sparse_results: List[Dict[str, Any]], k: int = 60) -> List[Dict[str, Any]]:
        scores = {}
        items = {}
        
        # Dense
        for rank, item in enumerate(dense_results):
            id_val = item["original_id"]
            if id_val not in scores:
                scores[id_val] = 0
                items[id_val] = item
            scores[id_val] += 1.0 / (k + rank + 1)
            
        # Sparse
        for rank, item in enumerate(sparse_results):
            id_val = item["original_id"]
            if id_val not in scores:
                scores[id_val] = 0
                items[id_val] = item
            scores[id_val] += 1.0 / (k + rank + 1)
            
        # Sort
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        return [items[id_val] for id_val in sorted_ids]
        
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        fetch_k = top_k * 2
        
        # Dense
        try:
            query_vector = self.embedder.encode([query])[0]
            dense_hits = self.qdrant.query_points(
                collection_name=self.collection_name,
                query=query_vector.tolist(),
                limit=fetch_k
            ).points
            dense_docs = [h.payload for h in dense_hits]
        except Exception as e:
            print(f"Dense retrieve err: {e}")
            dense_docs = []
            
        # Sparse
        try:
            sparse_ids = self.bm25.search(query, k=fetch_k)
            if sparse_ids:
                scroll_res = self.qdrant.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=Filter(
                        must=[
                            FieldCondition(key="original_id", match=MatchAny(any=sparse_ids))
                        ]
                    ),
                    limit=fetch_k
                )[0]
                sparse_docs = [h.payload for h in scroll_res]
                
                # We must preserve the ranking order from BM25
                sparse_dict = {d["original_id"]: d for d in sparse_docs}
                sparse_docs = [sparse_dict[sid] for sid in sparse_ids if sid in sparse_dict]
            else:
                sparse_docs = []
        except Exception as e:
            print(f"Sparse retrieve err: {e}")
            sparse_docs = []
            
        fused = self._rrf(dense_docs, sparse_docs)
        return fused[:top_k]
