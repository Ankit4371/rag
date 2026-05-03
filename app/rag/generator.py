from groq import Groq
import os
from typing import List

class GroqGenerator:
    def __init__(self, model_name: str = "llama-3.1-8b-instant"):
        self.model_name = model_name
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY", "dummy_key"))
        
    def generate(self, query: str, contexts: List[str]) -> str:
        if not contexts:
            context_str = "No relevant context found."
        else:
            context_str = "\n\n".join([f"Source {i+1}:\n{ctx}" for i, ctx in enumerate(contexts)])
            
        prompt = f"Answer the query based ONLY on the provided contexts.\n\nContexts:\n{context_str}\n\nQuery: {query}"
        
        try:
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model_name,
                temperature=0.0,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating answer: {str(e)}"
