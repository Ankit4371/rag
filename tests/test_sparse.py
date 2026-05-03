from app.rag.sparse import BM25Index
import os
import shutil

def test_bm25_index():
    index_dir = "tests/test_bm25_index"
    if os.path.exists(index_dir):
        shutil.rmtree(index_dir)
        
    corpus = [
        {"id": "doc1", "text": "Machine learning is fascinating."},
        {"id": "doc2", "text": "Deep learning models scale well."},
        {"id": "doc3", "text": "Natural language processing is a subfield of AI."}
    ]
    
    index = BM25Index(save_dir=index_dir)
    index.build(corpus)
    index.save()
    
    # Test retrieve
    results = index.search("learning", k=2)
    assert len(results) == 2
    assert "doc1" in results or "doc2" in results
    
    # Test load
    index2 = BM25Index(save_dir=index_dir)
    index2.load()
    results2 = index2.search("language processing", k=1)
    assert results2[0] == "doc3"
    
    shutil.rmtree(index_dir)
