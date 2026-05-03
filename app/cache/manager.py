import sqlite3
import json
import hashlib
from typing import Optional, Dict, Any

class CacheManager:
    def __init__(self, db_path: str = "data/cache.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        import os
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS exact_cache (
                key_hash TEXT PRIMARY KEY,
                query TEXT,
                response_json TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()

    def get(self, query: str, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        # Hash query + config to ensure cache matches parameters
        config_str = json.dumps(config, sort_keys=True)
        key = f"{query}||{config_str}"
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute("SELECT response_json FROM exact_cache WHERE key_hash = ?", (key_hash,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return json.loads(row[0])
        return None

    def set(self, query: str, config: Dict[str, Any], response: Dict[str, Any]):
        config_str = json.dumps(config, sort_keys=True)
        key = f"{query}||{config_str}"
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        
        # Don't cache the trace or latency as they change
        cached_res = {
            "answer": response.get("answer"),
            "sources": response.get("sources"),
            "cached": True
        }
        
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT OR REPLACE INTO exact_cache (key_hash, query, response_json) VALUES (?, ?, ?)",
            (key_hash, query, json.dumps(cached_res))
        )
        conn.commit()
        conn.close()
