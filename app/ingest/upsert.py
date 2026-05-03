import os
from typing import List
from pypdf import PdfReader
from app.ingest.chunker import RecursiveChunker
from app.rag.embedder import BGEEmbedder
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
import uuid

class IncrementalUpserter:
    def __init__(self, embedder: BGEEmbedder, qdrant: QdrantClient, collection_name: str = "rag_docs"):
        self.embedder = embedder
        self.qdrant = qdrant
        self.collection_name = collection_name
        self.chunker = RecursiveChunker()

    def ingest_pdf(self, file_path: str, metadata: dict = None) -> int:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        
        return self.ingest_text(text, metadata or {"source": os.path.basename(file_path)})

    def ingest_text(self, text: str, metadata: dict) -> int:
        chunks = self.chunker.chunk_text(text)
        
        points = []
        for chunk in chunks:
            vector = self.embedder.encode([chunk])[0]
            point_id = str(uuid.uuid4())
            points.append(PointStruct(
                id=point_id,
                vector=vector.tolist(),
                payload={
                    "text": chunk,
                    "metadata": metadata
                }
            ))
        
        self.qdrant.upsert(
            collection_name=self.collection_name,
            points=points
        )
        return len(points)
