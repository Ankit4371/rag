from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Dict, Any
from dotenv import load_dotenv
import os

load_dotenv()

from app.rag.pipeline import RAGPipeline

app = FastAPI(
    title="RAG",
    description="Production-grade Advanced RAG System",
    version="1.0.0",
)

# Ensure static directory exists
os.makedirs("static", exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

pipeline = None

def get_pipeline():
    global pipeline
    if pipeline is None:
        pipeline = RAGPipeline()
    return pipeline

class QueryRequest(BaseModel):
    query: str
    use_hybrid: bool = True
    use_reranker: bool = True
    use_compression: bool = False
    use_hyde: bool = False
    use_multi_query: bool = False
    
class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    latency_ms: int
    trace: List[Dict[str, Any]] = []

@app.get("/")
def read_root():
    return FileResponse("static/index.html")

@app.get("/health")
def health_check():
    return {"status": "ok", "version": "1.0.0"}

@app.post("/query", response_model=QueryResponse)
def query_endpoint(req: QueryRequest):
    p = get_pipeline()
    res = p.query(
        req.query,
        use_hybrid=req.use_hybrid,
        use_reranker=req.use_reranker,
        use_compression=req.use_compression,
        use_hyde=req.use_hyde,
        use_multi_query=req.use_multi_query
    )
    # Remove raw_contexts from API response (keep trace)
    res.pop("raw_contexts", None)
    return res
