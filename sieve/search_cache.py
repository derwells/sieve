"""A short-lived sqlite cache of backend search results.

Web results go stale, so unlike the Jev answer cache this one expires: an entry
is served for `SIEVE_SEARCH_CACHE_TTL` seconds (default one hour; 0 turns the
cache off) and the table is trimmed to the newest `MAX_ROWS` entries on write.
It exists so that repeating a search, or a variant two searches share, does not
hit the backend again within the hour.

The cache holds the caller's queries and the snippets returned for them, so
its directory is kept at mode 700 and the database and any SQLite sidecar
files at 600, whatever the umask and whatever modes an older version left.
Only the cache's own directory is chmodded, never its parents.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
from dataclasses import asdict
from pathlib import Path

from .backends import BackendResult, SearchHit

TTL_ENV = "SIEVE_SEARCH_CACHE_TTL"
DEFAULT_TTL_SECONDS = 3600.0
MAX_ROWS = 2000
CACHE_DIR = Path.home() / ".cache" / "sieve" / "search"
DB_NAME = "results.sqlite3"
#: Files SQLite may create beside the database, depending on journal mode.
SIDECAR_SUFFIXES = ("-journal", "-wal", "-shm")
DIR_MODE = 0o700
FILE_MODE = 0o600


def _make_private(directory: Path) -> Path:
    """Create the cache directory and database owner-only, tightening existing modes."""
    directory.mkdir(parents=True, exist_ok=True)
    os.chmod(directory, DIR_MODE)
    path = directory / DB_NAME
    os.close(os.open(path, os.O_RDWR | os.O_CREAT, FILE_MODE))
    for candidate in [path, *(directory / f"{DB_NAME}{suffix}" for suffix in SIDECAR_SUFFIXES)]:
        if candidate.exists() and not candidate.is_symlink():
            os.chmod(candidate, FILE_MODE)
    return path


def ttl_seconds(env: dict[str, str] | None = None) -> float:
    source = os.environ if env is None else env
    raw = source.get(TTL_ENV)
    if raw is None or not raw.strip():
        return DEFAULT_TTL_SECONDS
    try:
        return max(0.0, float(raw))
    except ValueError:
        return DEFAULT_TTL_SECONDS


def result_key(backend, query: str, count: int) -> str:
    fingerprint = getattr(backend, "fingerprint", None)
    parts = (backend.name, fingerprint() if callable(fingerprint) else "", query, str(count))
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


class SearchCache:
    """Backend results by (backend, backend config, query, count), with a TTL.

    `directory` belongs to the cache alone: it is created and kept at mode 700.
    """

    def __init__(self, directory: Path = CACHE_DIR, ttl: float | None = None) -> None:
        self.ttl = ttl_seconds() if ttl is None else ttl
        self._lock = threading.Lock()
        self.path = _make_private(directory)
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS results (key TEXT PRIMARY KEY, result TEXT NOT NULL, created REAL NOT NULL)"
        )
        self._db.commit()

    @property
    def enabled(self) -> bool:
        return self.ttl > 0

    def get(self, key: str) -> BackendResult | None:
        if not self.enabled:
            return None
        with self._lock:
            row = self._db.execute("SELECT result, created FROM results WHERE key = ?", (key,)).fetchone()
        if row is None or time.time() - row[1] > self.ttl:
            return None
        data = json.loads(row[0])
        hits = [SearchHit(**{**hit, "engines": tuple(hit.get("engines") or ())}) for hit in data["hits"]]
        return BackendResult(query=data["query"], hits=hits, usage=data["usage"], wall_seconds=data["wall_seconds"])

    def put(self, key: str, result: BackendResult) -> None:
        if not self.enabled or not result.hits:
            return
        data = {
            "query": result.query,
            "hits": [asdict(hit) for hit in result.hits],
            "usage": result.usage,
            "wall_seconds": result.wall_seconds,
        }
        now = time.time()
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO results (key, result, created) VALUES (?, ?, ?)",
                (key, json.dumps(data), now),
            )
            self._db.execute("DELETE FROM results WHERE created < ?", (now - self.ttl,))
            self._db.execute(
                "DELETE FROM results WHERE key NOT IN (SELECT key FROM results ORDER BY created DESC LIMIT ?)",
                (MAX_ROWS,),
            )
            self._db.commit()

    def close(self) -> None:
        with self._lock:
            self._db.close()
