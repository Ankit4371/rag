from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import shutil
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
    use_crag: bool = False
    use_guards: bool = False
    use_cache: bool = True
    
class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    latency_ms: int
    trace: List[Dict[str, Any]] = []
    cost: Optional[Dict[str, Any]] = None
    cached: Optional[bool] = False

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
        use_multi_query=req.use_multi_query,
        use_crag=req.use_crag,
        use_guards=req.use_guards,
        use_cache=req.use_cache
    )
    # Remove raw_contexts from API response (keep trace)
    res.pop("raw_contexts", None)
    return res

@app.post("/ingest")
async def ingest_file(file: UploadFile = File(...)):
    temp_path = f"data/temp_{file.filename}"
    os.makedirs("data", exist_ok=True)
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    p = get_pipeline()
    from app.ingest.upsert import IncrementalUpserter
    upserter = IncrementalUpserter(p.embedder, p.qdrant, p.collection_name)
    
    try:
        if file.filename.endswith(".pdf"):
            num_chunks = upserter.ingest_pdf(temp_path)
        else:
            with open(temp_path, "r") as f:
                num_chunks = upserter.ingest_text(f.read(), {"source": file.filename})
        
        return {"status": "success", "chunks_ingested": num_chunks}
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
