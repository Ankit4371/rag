import os
import sys
import pandas as pd
from datasets import Dataset

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.rag.pipeline import RAGPipeline
from dotenv import load_dotenv

load_dotenv()

def run_evaluation():
    try:
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )
        from langchain_groq import ChatGroq
        from langchain_community.embeddings import HuggingFaceEmbeddings
    except ImportError:
        print("Please install ragas and langchain-groq: uv sync --extra dev")
        return

    from evals.dataset import load_eval_dataset
    df = load_eval_dataset()
    
    pipeline = RAGPipeline()
    
    answers = []
    contexts_list = []
    
    print("Generating answers for evaluation dataset...")
    for q in df["question"]:
        res = pipeline.query(q)
        answers.append(res["answer"])
        contexts_list.append(res.get("raw_contexts", []))
        
    df["answer"] = answers
    df["contexts"] = contexts_list
    
    dataset = Dataset.from_pandas(df)
    
    # Ragas needs an LLM and Embeddings to evaluate
    if not os.environ.get("GROQ_API_KEY"):
        print("Warning: GROQ_API_KEY not set. Ragas evaluation might fail or use defaults.")
        
    groq_llm = ChatGroq(model_name="llama-3.1-8b-instant")
    hf_embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
    
    print("Running RAGAS evaluation...")
    result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
        llm=groq_llm,
        embeddings=hf_embeddings
    )
    
    metrics_dict = {
        "phase": "phase-4-naive",
        "faithfulness": result.get("faithfulness", 0),
        "answer_relevancy": result.get("answer_relevancy", 0),
        "context_precision": result.get("context_precision", 0),
        "context_recall": result.get("context_recall", 0),
    }
    
    history_file = "evals/history.csv"
    os.makedirs("evals", exist_ok=True)
    mode = 'a' if os.path.exists(history_file) else 'w'
    header = not os.path.exists(history_file)
    pd.DataFrame([metrics_dict]).to_csv(history_file, mode=mode, header=header, index=False)
    print("Evaluations saved to", history_file)
    print(metrics_dict)

if __name__ == "__main__":
    run_evaluation()
