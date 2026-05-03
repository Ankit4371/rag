from groq import Groq
import os
from typing import List

SYSTEM_PROMPT = """\
You are an expert AI research assistant specializing in Machine Learning, NLP, \
and Artificial Intelligence. You synthesize information from retrieved academic \
papers to answer user queries with precision and depth.

## Rules

1. **Ground every claim in sources.** Cite using [Source N] inline. If a fact \
   comes from Source 2, write it as: "Transformers use self-attention [Source 2]."
2. **Never hallucinate.** If the provided sources do not contain enough \
   information to fully answer the query, explicitly state what is covered \
   and what is not. Do NOT invent facts.
3. **Synthesize, don't just list.** Combine insights across multiple sources \
   into a coherent narrative. Identify agreements, contradictions, or \
   complementary findings between sources.
4. **Structure your answer** using markdown:
   - Use **bold** for key terms and paper names.
   - Use bullet points for listing distinct approaches or findings.
   - Keep paragraphs short (2-3 sentences max).
5. **If sources conflict**, explicitly note the disagreement and present both \
   perspectives with their respective citations.
6. **If no relevant context is found**, say: "The retrieved sources do not \
   contain information relevant to this query." Do not guess.
7. **Be concise but comprehensive.** Aim for 150-300 words unless the query \
   demands more depth.
"""


class GroqGenerator:
    def __init__(self, model_name: str = "llama-3.1-8b-instant"):
        self.model_name = model_name
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY", "dummy_key"))
        
    def generate(self, query: str, contexts: List[str]) -> str:
        if not contexts:
            context_str = "No relevant sources were retrieved."
        else:
            context_str = "\n\n---\n\n".join(
                [f"[Source {i+1}]\n{ctx}" for i, ctx in enumerate(contexts)]
            )
            
        user_prompt = (
            f"## Retrieved Sources\n\n{context_str}\n\n"
            f"---\n\n"
            f"## User Query\n\n{query}\n\n"
            f"Synthesize a well-structured answer using ONLY the sources above. "
            f"Cite every claim with [Source N]."
        )
        
        try:
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                model=self.model_name,
                temperature=0.1,  # Low but not zero — allows natural phrasing
                max_tokens=1024,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating answer: {str(e)}"
