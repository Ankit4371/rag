from fastmcp import FastMCP
from app.rag.pipeline import RAGPipeline
import os
from dotenv import load_dotenv

load_dotenv()

mcp = FastMCP("RAG-Explorer")
pipeline = None

def get_pipeline():
    global pipeline
    if pipeline is None:
        pipeline = RAGPipeline()
    return pipeline

@mcp.tool()
def query_research_papers(query: str, use_hyde: bool = True, use_reranker: bool = True) -> str:
    """
    Search and synthesize answers from 15,000+ research papers.
    """
    p = get_pipeline()
    res = p.query(
        query,
        use_hyde=use_hyde,
        use_reranker=use_reranker,
        use_guards=True # Always use guards for MCP for safety
    )
    return res["answer"]

@mcp.tool()
def ingest_text_document(text: str, filename: str) -> str:
    """
    Ingest a new text document into the research corpus.
    """
    p = get_pipeline()
    from app.ingest.upsert import IncrementalUpserter
    upserter = IncrementalUpserter(p.embedder, p.qdrant, p.collection_name)
    num_chunks = upserter.ingest_text(text, {"source": filename})
    return f"Successfully ingested {filename} ({num_chunks} chunks)."

if __name__ == "__main__":
    mcp.run()
