from fastapi import FastAPI

app = FastAPI(
    title="RAG",
    description="Production-grade Advanced RAG System",
    version="1.0.0",
)

@app.get("/health")
def health_check():
    return {"status": "ok", "version": "1.0.0"}
