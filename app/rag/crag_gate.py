"""
CRAG Gate — Corrective RAG Confidence Gate

Evaluates whether the retrieved documents are sufficient to answer the user's
query. Uses the LLM itself (via Groq) to classify retrieval quality into one
of three categories:

- **correct**: Retrieved docs contain relevant information → proceed normally
- **ambiguous**: Docs are partially relevant → augment with web search
- **incorrect**: Docs are not relevant at all → fall back entirely to web search

This prevents the system from hallucinating answers when the corpus doesn't
cover the query topic.

Reference: Yan et al. 2024 — https://arxiv.org/abs/2401.15884
"""

import os
from typing import List, Tuple
from groq import Groq


class CRAGGate:
    def __init__(self, model_name: str = "llama-3.1-8b-instant"):
        self.model_name = model_name
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY", "dummy_key"))

    def evaluate(self, query: str, contexts: List[str]) -> Tuple[str, str]:
        """
        Judge whether the retrieved contexts are sufficient to answer the query.
        
        Returns:
            (verdict, reasoning) where verdict is one of: 
            'correct', 'ambiguous', 'incorrect'
        """
        if not contexts:
            return ("incorrect", "No documents were retrieved.")
        
        # Build a condensed view of retrieved contexts (first 200 chars each)
        context_summaries = "\n".join(
            [f"Doc {i+1}: {ctx[:200]}..." for i, ctx in enumerate(contexts)]
        )
        
        prompt = (
            f"You are a retrieval quality judge. Given a user query and the retrieved documents, "
            f"classify the retrieval quality.\n\n"
            f"## User Query\n{query}\n\n"
            f"## Retrieved Documents (summaries)\n{context_summaries}\n\n"
            f"## Task\n"
            f"Classify as exactly ONE of:\n"
            f"- **correct**: The documents contain information that directly answers the query.\n"
            f"- **ambiguous**: The documents are partially relevant but may not fully answer the query.\n"
            f"- **incorrect**: The documents are not relevant to the query at all.\n\n"
            f"Respond in this exact format (2 lines only):\n"
            f"VERDICT: <correct|ambiguous|incorrect>\n"
            f"REASON: <one sentence explaining why>"
        )

        try:
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model_name,
                temperature=0.0,
                max_tokens=100,
            )
            text = response.choices[0].message.content.strip()
            
            # Parse verdict and reason
            verdict = "ambiguous"  # default
            reason = text
            
            for line in text.split("\n"):
                line_lower = line.strip().lower()
                if line_lower.startswith("verdict:"):
                    v = line_lower.split("verdict:")[-1].strip()
                    if "correct" in v and "incorrect" not in v:
                        verdict = "correct"
                    elif "incorrect" in v:
                        verdict = "incorrect"
                    else:
                        verdict = "ambiguous"
                elif line_lower.startswith("reason:"):
                    reason = line.split(":", 1)[-1].strip()
            
            return (verdict, reason)
            
        except Exception as e:
            print(f"CRAG gate error: {e}")
            # On failure, assume docs are fine to avoid blocking the pipeline
            return ("correct", f"Gate evaluation failed: {e}")
