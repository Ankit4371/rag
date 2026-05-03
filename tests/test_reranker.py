from app.rag.reranker import BGEReranker

def test_reranker_logic():
    class MockModel:
        def predict(self, pairs):
            scores = []
            for q, t in pairs:
                if "Paris" in t:
                    scores.append(0.9)
                elif "France" in t:
                    scores.append(0.5)
                else:
                    scores.append(0.1)
            return scores
            
    reranker = BGEReranker()
    reranker.model = MockModel()
    
    docs = [
        {"text": "A completely unrelated document."},
        {"text": "Paris is the capital of France."},
        {"text": "France is a country in Europe."}
    ]
    
    reranked = reranker.rerank("What is the capital of France?", docs, top_k=2)
    assert len(reranked) == 2
    assert "Paris is the capital" in reranked[0]["text"]
    assert "France is a country" in reranked[1]["text"]
