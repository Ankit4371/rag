from app.rag.retriever import HybridRetriever

def test_rrf_logic():
    # We can mock the dependencies
    retriever = HybridRetriever(None, None, None)
    
    dense = [
        {"original_id": "A", "text": "Doc A"},
        {"original_id": "B", "text": "Doc B"},
        {"original_id": "C", "text": "Doc C"},
    ]
    
    sparse = [
        {"original_id": "B", "text": "Doc B"},
        {"original_id": "D", "text": "Doc D"},
        {"original_id": "A", "text": "Doc A"},
    ]
    
    # RRF (k=60)
    # A: dense(rank=0)->1/61, sparse(rank=2)->1/63. Total = 0.01639 + 0.01587 = 0.03226
    # B: dense(rank=1)->1/62, sparse(rank=0)->1/61. Total = 0.01612 + 0.01639 = 0.03251
    # C: dense(rank=2)->1/63. Total = 0.01587
    # D: sparse(rank=1)->1/62. Total = 0.01612
    # Expected order: B, A, D, C
    
    fused = retriever._rrf(dense, sparse, k=60)
    
    assert fused[0]["original_id"] == "B"
    assert fused[1]["original_id"] == "A"
    assert fused[2]["original_id"] == "D"
    assert fused[3]["original_id"] == "C"
