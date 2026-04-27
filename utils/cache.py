import sqlite3
import json
import os
from core.config import CACHE_DB


def _conn():
    c = sqlite3.connect(CACHE_DB)
    c.execute(
        "CREATE TABLE IF NOT EXISTS cache "
        "(key TEXT PRIMARY KEY, value TEXT, created_at DATETIME DEFAULT CURRENT_TIMESTAMP)"
    )
    c.commit()
    return c


def get_cached(key: str):
    try:
        with _conn() as c:
            row = c.execute("SELECT value FROM cache WHERE key=?", (key,)).fetchone()
            return json.loads(row[0]) if row else None
    except Exception:
        return None


def set_cached(key: str, value):
    try:
        with _conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO cache (key, value) VALUES (?,?)",
                (key, json.dumps(value, default=str)),
            )
    except Exception:
        pass


def cache_stats() -> dict:
    try:
        with _conn() as c:
            count = c.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
            size = os.path.getsize(CACHE_DB) if os.path.exists(CACHE_DB) else 0
            return {"entries": count, "size_kb": round(size / 1024, 1)}
    except Exception:
        return {"entries": 0, "size_kb": 0}


def clear_cache():
    try:
        with _conn() as c:
            c.execute("DELETE FROM cache")
    except Exception:
        pass
