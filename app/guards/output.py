import os
from groq import Groq
from typing import Tuple, List

class OutputGuard:
    def __init__(self, model_name: str = "llama-3.1-8b-instant"):
        self.model_name = model_name
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY", "dummy_key"))

    def validate(self, query: str, answer: str, contexts: List[str]) -> Tuple[bool, str]:
        """
        Check for Toxicity and Hallucinations (Faithfulness).
        Returns (is_safe, message_or_checked_answer).
        """
        context_str = "\n".join([f"Source {i+1}: {ctx[:300]}..." for i, ctx in enumerate(contexts)])
        
        prompt = (
            f"You are a quality control agent for a RAG system. Analyze the generated answer against the query and sources.\n\n"
            f"Query: {query}\n\n"
            f"Answer: {answer}\n\n"
            f"Sources:\n{context_str}\n\n"
            f"Check for:\n"
            f"1. Toxicity: Hate speech, harassment, or offensive content.\n"
            f"2. Faithfulness: Does the answer hallucinate or contradict the sources?\n\n"
            f"Respond in JSON format:\n"
            f"{{\n"
            f"  \"is_toxic\": boolean,\n"
            f"  \"is_faithful\": boolean,\n"
            f"  \"hallucination_reason\": \"string (if any)\",\n"
            f"  \"toxicity_reason\": \"string (if any)\"\n"
            f"}}\n"
            f"Respond ONLY with the JSON."
        )

        try:
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model_name,
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            import json
            res = json.loads(response.choices[0].message.content)
            
            if res.get("is_toxic", False):
                return False, "Error: Generated content was flagged for toxicity."
            
            if not res.get("is_faithful", True):
                # We can either block or just warn. Let's block for strict guardrails.
                return False, f"Error: Potential hallucination detected. Reason: {res.get('hallucination_reason', 'Unknown')}"
            
            return True, answer
            
        except Exception as e:
            print(f"Output Guard Error: {e}")
            return True, answer # Fallback on error
