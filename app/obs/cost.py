from typing import Dict, Any

class CostTracker:
    # Approximate pricing for Groq/Cohere (Free tier has limits, but we track 'theoretical' cost)
    PRICING = {
        "llama-3.1-8b-instant": {"input": 0.05, "output": 0.08}, # per 1M tokens
        "rerank-v3": 0.0001, # per request
    }

    def __init__(self):
        self.total_cost = 0.0

    def estimate_tokens(self, text: str) -> int:
        # Rough estimate: 1 token ~= 4 characters
        return len(text) // 4

    def track_query(self, query: str, contexts: list, answer: str, used_reranker: bool) -> Dict[str, Any]:
        input_tokens = self.estimate_tokens(query + "".join(contexts))
        output_tokens = self.estimate_tokens(answer)
        
        cost = (input_tokens * self.PRICING["llama-3.1-8b-instant"]["input"] / 1000000) + \
               (output_tokens * self.PRICING["llama-3.1-8b-instant"]["output"] / 1000000)
        
        if used_reranker:
            cost += self.PRICING["rerank-v3"]
            
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "estimated_cost_usd": round(cost, 6)
        }
