from app.rag.embedder import BGEEmbedder

def test_embedder_shape():
    # Use a small model for test speed if preferred, but we test BGE-M3
    embedder = BGEEmbedder(model_name="BAAI/bge-m3")
    texts = ["This is a test document.", "Here is another one."]
    embeddings = embedder.encode(texts)
    
    assert embeddings.shape == (2, 1024)
    assert str(embeddings.dtype) == "float32"
