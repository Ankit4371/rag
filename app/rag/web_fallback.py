"""
Web Fallback — DuckDuckGo Search for out-of-corpus queries

When the CRAG Gate determines that retrieved documents are insufficient
(verdict: 'ambiguous' or 'incorrect'), this module searches the open web
via DuckDuckGo (no API key required, completely free) and returns structured
results as additional context for the generator.

Usage:
    fallback = WebFallback()
    results = fallback.search("latest prompt engineering techniques", max_results=5)
    # Returns list of dicts with 'text' and 'metadata' keys
"""

from typing import List, Dict, Any


class WebFallback:
    def __init__(self):
        self._ddgs = None

    def _get_client(self):
        """Lazy-load DuckDuckGo search client."""
        if self._ddgs is None:
            try:
                from duckduckgo_search import DDGS
                self._ddgs = DDGS()
            except ImportError:
                print("duckduckgo-search not installed. Run: uv add duckduckgo-search")
                return None
        return self._ddgs

    def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search the web via DuckDuckGo and return structured results.
        
        Returns:
            List of docs matching the pipeline's expected format:
            [{"text": "...", "metadata": {"title": "...", "source": "web"}}]
        """
        client = self._get_client()
        if client is None:
            return []

        try:
            raw_results = client.text(query, max_results=max_results)
            docs = []
            for r in raw_results:
                docs.append({
                    "text": f"Title: {r.get('title', 'N/A')}\n\n{r.get('body', '')}",
                    "metadata": {
                        "title": r.get("title", "Web Result"),
                        "source": "web",
                        "url": r.get("href", ""),
                    }
                })
            return docs
        except Exception as e:
            print(f"Web search error: {e}")
            return []
