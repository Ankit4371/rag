from datasets import load_dataset
from typing import Iterator, Dict, Any

class ArxivLoader:
    def __init__(self, categories: list[str] = None, years: list[str] = None):
        self.categories = categories or ["cs.AI", "cs.LG", "cs.CL"]
        self.years = years or ["2023", "2024", "2025", "2026"]

    def load(self, max_docs: int = None) -> Iterator[Dict[str, Any]]:
        # arxiv dataset features: id, submitter, authors, title, comments, journal-ref, doi, report-no, categories, license, abstract, versions, update_date, authors_parsed
        ds = load_dataset("librarian-bots/arxiv-metadata-snapshot", split="train", streaming=True)
        count = 0
        for row in ds:
            cats = row.get("categories", "")
            if any(c in cats for c in self.categories):
                date = row.get("update_date")
                
                # Handling both string and datetime
                if hasattr(date, "strftime"):
                    year_str = date.strftime("%Y")
                else:
                    year_str = str(date)[:4] if date else ""
                    
                if year_str in self.years:
                    yield {
                        "id": row["id"],
                        "text": row["abstract"],
                        "metadata": {
                            "title": row["title"],
                            "year": year_str,
                            "arxiv_id": row["id"]
                        }
                    }
                    count += 1
                    if max_docs and count >= max_docs:
                        break
