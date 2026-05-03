import tiktoken
from typing import List, Dict, Any
import uuid

class RecursiveChunker:
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def chunk(self, doc: Dict[str, Any]) -> List[Dict[str, Any]]:
        text = doc["text"]
        tokens = self.tokenizer.encode(text)
        chunks = []
        
        if not tokens:
            return chunks

        # For arxiv abstracts, a simple token-based sliding window is sufficient
        # as they are typically short and single paragraphs.
        
        step_size = max(1, self.chunk_size - self.chunk_overlap)
        
        for i in range(0, len(tokens), step_size):
            chunk_tokens = tokens[i:i + self.chunk_size]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            chunks.append({
                "id": f"{doc.get('metadata', {}).get('arxiv_id', uuid.uuid4())}_{len(chunks)}",
                "text": chunk_text,
                "metadata": doc.get("metadata", {})
            })
            if i + self.chunk_size >= len(tokens):
                break
                
        return chunks
