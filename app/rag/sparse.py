import bm25s
from typing import List, Dict, Any
import os
import json

class BM25Index:
    def __init__(self, save_dir: str = "data/bm25_index"):
        self.save_dir = save_dir
        self.retriever = None
        self.corpus_ids = None

    def build(self, corpus: List[Dict[str, Any]]):
        texts = [doc["text"] for doc in corpus]
        self.corpus_ids = [doc["id"] for doc in corpus]
        
        corpus_tokens = bm25s.tokenize(texts)
        self.retriever = bm25s.BM25()
        self.retriever.index(corpus_tokens)
        
    def save(self):
        os.makedirs(self.save_dir, exist_ok=True)
        if self.retriever:
            self.retriever.save(self.save_dir)
            with open(os.path.join(self.save_dir, "corpus_ids.json"), "w") as f:
                json.dump(self.corpus_ids, f)
                
    def load(self):
        if os.path.exists(self.save_dir):
            self.retriever = bm25s.BM25.load(self.save_dir, load_corpus=False)
            with open(os.path.join(self.save_dir, "corpus_ids.json"), "r") as f:
                self.corpus_ids = json.load(f)

    def search(self, query: str, k: int = 10) -> List[str]:
        if not self.retriever:
            raise ValueError("Index not loaded or built")
        query_tokens = bm25s.tokenize(query)
        docs, scores = self.retriever.retrieve(query_tokens, k=k)
        
        result_ids = []
        # docs is shape (n_queries, k)
        for idx in docs[0]:
            # Convert np.int to int
            result_ids.append(self.corpus_ids[int(idx)])
            
        return result_ids
