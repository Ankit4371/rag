import os
from typing import List, Dict, Any

class Reranker:
    def __init__(self, top_k: int = 5):
        self.top_k = top_k
        self.cohere_api_key = os.getenv("COHERE_API_KEY")
        
        if self.cohere_api_key:
            import cohere
            print("Using Cohere for offloaded Reranking.")
            self.co_client = cohere.ClientV2(self.cohere_api_key)
            self.mode = "cohere"
        else:
            print("Using Local BGE M3 for Reranking.")
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder("BAAI/bge-reranker-v2-m3", device="cpu")
            self.mode = "local"

    def rerank(self, query: str, docs: List[Dict[str, Any]], top_k: int = None) -> List[Dict[str, Any]]:
        k = top_k or self.top_k
        if not docs:
            return []
            
        if self.mode == "cohere":
            doc_texts = [doc.get("text", "") for doc in docs]
            response = self.co_client.rerank(
                model="rerank-english-v3.0",
                query=query,
                documents=doc_texts,
                top_n=k,
            )
            # Match results back to original docs
            reranked_docs = []
            for result in response.results:
                reranked_docs.append(docs[result.index])
            return reranked_docs
        else:
            pairs = [[query, doc.get("text", "")] for doc in docs]
            scores = self.model.predict(pairs)
            
            # Zip scores with docs and sort
            scored_docs = list(zip(docs, scores))
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            
            return [doc for doc, score in scored_docs[:k]]
