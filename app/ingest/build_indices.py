import os
import sys
import json
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from app.rag.embedder import BGEEmbedder
from app.rag.sparse import BM25Index

def main():
    print("Loading corpus...")
    corpus = []
    if not os.path.exists("data/corpus.jsonl"):
        print("data/corpus.jsonl not found! Please run Phase 2 first.")
        return
        
    with open("data/corpus.jsonl", "r") as f:
        for line in f:
            corpus.append(json.loads(line))
            
    # For testing speed we can limit this if requested, but we'll run full 25k
    print(f"Loaded {len(corpus)} documents.")
    
    print("Building BM25S index...")
    bm25 = BM25Index()
    bm25.build(corpus)
    bm25.save()
    print("BM25S index saved to data/bm25_index.")
    
    print("Building Dense Index (Qdrant)...")
    embedder = BGEEmbedder()
    
    qdrant = QdrantClient(path="data/qdrant_db")
    
    collection_name = "rag_docs"
    # recreate_collection is safe, clears if exists
    qdrant.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
    )
    
    batch_size = 128
    for i in range(0, len(corpus), batch_size):
        batch = corpus[i:i+batch_size]
        texts = [doc["text"] for doc in batch]
        
        embeddings = embedder.encode(texts, batch_size=batch_size)
        
        points = []
        for doc, emb in zip(batch, embeddings):
            doc_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, doc["id"]))
            points.append(PointStruct(
                id=doc_uuid,
                vector=emb.tolist(),
                payload={"original_id": doc["id"], "text": doc["text"], "metadata": doc["metadata"]}
            ))
            
        qdrant.upsert(
            collection_name=collection_name,
            points=points
        )
        print(f"Upserted {min(i+batch_size, len(corpus))}/{len(corpus)} points to Qdrant...")

    print("Successfully built both indices!")

if __name__ == "__main__":
    main()
