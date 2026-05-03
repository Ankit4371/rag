from app.ingest.chunker import RecursiveChunker
import tiktoken

def test_chunker_bounds():
    chunker = RecursiveChunker(chunk_size=10, chunk_overlap=2)
    doc = {
        "text": "This is a very long text that should definitely be split into multiple chunks because it exceeds the ten token limit we just set.",
        "metadata": {"arxiv_id": "1234"}
    }
    chunks = chunker.chunk(doc)
    assert len(chunks) > 1
    
    enc = tiktoken.get_encoding("cl100k_base")
    for i, chunk in enumerate(chunks):
        assert len(enc.encode(chunk["text"])) <= 10
        assert chunk["metadata"]["arxiv_id"] == "1234"
        assert chunk["id"] == f"1234_{i}"
