"""
Query Rewriter — HyDE + Multi-Query Expansion

Improves retrieval recall by expanding the original user query before embedding.
Two techniques are combined:

1. **HyDE (Hypothetical Document Embeddings)**: Ask the LLM to generate a
   hypothetical paragraph that *would* answer the query. This synthetic doc
   is then embedded alongside the original query, producing vectors that are
   closer to actual corpus documents in the embedding space.

2. **Multi-Query**: Ask the LLM to rephrase the original query from 2 different
   angles. Each variant is embedded and searched separately, then all results
   are merged and deduplicated via a simple union.

Both techniques use Groq API (free tier) — zero local compute.

Reference:
- HyDE: Gao et al. 2022 — https://arxiv.org/abs/2212.10496
- Multi-Query: LangChain concept — query decomposition for better recall
"""

import os
import json
from typing import List, Dict, Any
from groq import Groq


class QueryRewriter:
    def __init__(self, model_name: str = "llama-3.1-8b-instant"):
        self.model_name = model_name
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY", "dummy_key"))

    def hyde(self, query: str) -> str:
        """
        Generate a hypothetical document that would answer the query.
        The synthetic doc is embedded to improve vector similarity with real corpus docs.
        """
        prompt = (
            "You are a research paper abstract generator. "
            "Given the following research question, write a short, realistic paragraph "
            "(3-5 sentences) that could appear in a machine learning research paper's abstract "
            "answering this question. Do NOT preface your answer. Just write the paragraph directly.\n\n"
            f"Research Question: {query}"
        )

        try:
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model_name,
                temperature=0.7,
                max_tokens=200,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"HyDE generation failed: {e}")
            return ""

    def multi_query(self, query: str, n: int = 2) -> List[str]:
        """
        Generate n alternative phrasings of the query to improve recall.
        Each rephrasing captures a different angle of the same information need.
        """
        prompt = (
            f"You are a search query optimizer. Given the following user query, "
            f"generate exactly {n} alternative search queries that capture different aspects "
            f"or phrasings of the same information need. Each should be a standalone search query.\n\n"
            f"Return ONLY a JSON array of strings, no explanation.\n\n"
            f"User Query: {query}\n\n"
            f"Output:"
        )

        try:
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model_name,
                temperature=0.5,
                max_tokens=200,
            )
            raw = response.choices[0].message.content.strip()

            # Parse JSON array from LLM output
            # Handle cases where LLM wraps in markdown code blocks
            if "```" in raw:
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            
            queries = json.loads(raw)
            if isinstance(queries, list):
                return [q for q in queries if isinstance(q, str)][:n]
        except Exception as e:
            print(f"Multi-query generation failed: {e}")
        
        return []

    def rewrite(self, query: str, use_hyde: bool = True, use_multi_query: bool = True) -> Dict[str, Any]:
        """
        Full rewrite pipeline. Returns:
        - hyde_doc: the synthetic hypothetical document (or empty)
        - queries: list of all query variants (original + rephrasings)
        """
        result = {
            "original_query": query,
            "hyde_doc": "",
            "alt_queries": [],
        }

        if use_hyde:
            result["hyde_doc"] = self.hyde(query)

        if use_multi_query:
            result["alt_queries"] = self.multi_query(query)

        return result
