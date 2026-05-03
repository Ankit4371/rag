from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_query_endpoint(monkeypatch):
    # Mock the generator so we don't hit the Groq API
    class MockGenerator:
        def generate(self, q, c):
            return "Mock answer"
            
    # Mock Qdrant
    class MockQdrant:
        def search(self, **kwargs):
            from qdrant_client.models import ScoredPoint
            return [ScoredPoint(id=1, version=1, score=0.9, payload={"text": "Mock context", "metadata": {"title": "Mock Title"}})]
            
    from app.rag.pipeline import RAGPipeline
    # We patch the init
    def mock_init(self):
        self.generator = MockGenerator()
        self.qdrant = MockQdrant()
        class MockEmbedder:
            def encode(self, texts):
                import numpy as np
                return np.zeros((len(texts), 1024))
        self.embedder = MockEmbedder()
        self.collection_name = "rag_docs"
        
    monkeypatch.setattr(RAGPipeline, "__init__", mock_init)
    
    response = client.post("/query", json={"query": "test query"})
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sources" in data
    assert "latency_ms" in data
    assert data["answer"] == "Mock answer"
