from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List

class BGEEmbedder:
    def __init__(self, model_name: str = "BAAI/bge-m3", device: str = "cpu"):
        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        # BGE-M3 outputs 1024 dim
        embeddings = self.model.encode(texts, batch_size=batch_size, show_progress_bar=False)
        return np.array(embeddings, dtype=np.float32)
