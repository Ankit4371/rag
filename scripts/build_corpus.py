import os
import sys
import json

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.ingest.loaders import ArxivLoader
from app.ingest.chunker import RecursiveChunker

def main():
    os.makedirs("data", exist_ok=True)
    loader = ArxivLoader()
    chunker = RecursiveChunker(chunk_size=512, chunk_overlap=64)
    
    print("Starting arxiv stream. This might take a bit for the first rows...")
    # Fetching roughly 25k docs which will produce ~30k chunks (since some > 512)
    docs_generator = loader.load(max_docs=25000)
    
    count = 0
    with open("data/corpus.jsonl", "w") as f:
        for doc in docs_generator:
            chunks = chunker.chunk(doc)
            for chunk in chunks:
                f.write(json.dumps(chunk) + "\n")
                count += 1
                if count % 1000 == 0:
                    print(f"Written {count} chunks...")
                    
    print(f"Done! Total chunks written: {count}")

if __name__ == "__main__":
    main()
