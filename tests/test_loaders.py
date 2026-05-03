from app.ingest.loaders import ArxivLoader

def test_arxiv_loader():
    loader = ArxivLoader(categories=["cs.AI"], years=["2023"])
    docs = list(loader.load(max_docs=2))
    assert len(docs) == 2
    for doc in docs:
        assert "id" in doc
        assert "text" in doc
        assert "metadata" in doc
        assert "title" in doc["metadata"]
        assert "year" in doc["metadata"]
        assert "arxiv_id" in doc["metadata"]
        assert doc["metadata"]["year"] == "2023"
