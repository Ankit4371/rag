from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

from app.rag.pipeline import RAGPipeline

app = FastAPI(
    title="RAG",
    description="Production-grade Advanced RAG System",
    version="1.0.0",
)

pipeline = None

def get_pipeline():
    global pipeline
    if pipeline is None:
        pipeline = RAGPipeline()
    return pipeline

class QueryRequest(BaseModel):
    query: str
    
class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    latency_ms: int

@app.get("/health")
def health_check():
    return {"status": "ok", "version": "1.0.0"}

@app.post("/query", response_model=QueryResponse)
def query_endpoint(req: QueryRequest):
    p = get_pipeline()
    res = p.query(req.query)
    # Remove raw_contexts from API response
    res.pop("raw_contexts", None)
    return res
