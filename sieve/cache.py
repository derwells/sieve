"""sqlite cache of Jev answers, keyed on (model, question, unit text).

The cache lives in the repository being searched when that repository has a
`.gitignore` sieve can add itself to; otherwise it goes under
`~/.cache/sieve/<repo-hash>/` so sieve never leaves an untracked directory in
someone else's working tree.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path

#: The line sieve appends to a repository's .gitignore.
IGNORE_PATTERN = ".sieve-cache/"
CACHE_DIRNAME = ".sieve-cache"
USER_CACHE_ROOT = Path.home() / ".cache" / "sieve"
DB_NAME = "answers.sqlite3"


def answer_key(model: str, question: str, unit_text: str, spec_text: str = "") -> str:
    """Stable cache key. Any change to model, question, prompt, or text is a new key.

    `spec_text` fingerprints the QuestionSpec: the instructions and criteria are
    sent to the model too, so editing them has to invalidate the stored answers.
    """
    digest = hashlib.sha256()
    for part in (model, question, unit_text, spec_text):
        digest.update(part.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


def _repo_hash(repo: Path) -> str:
    return hashlib.sha256(str(repo).encode("utf-8")).hexdigest()[:16]


def ensure_ignored(repo: Path) -> bool:
    """Add the cache pattern to an existing .gitignore. True if it is covered now."""
    gitignore = repo / ".gitignore"
    if not gitignore.is_file():
        return False
    try:
        lines = gitignore.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        return False
    if any(line.strip().rstrip("/") == IGNORE_PATTERN.rstrip("/") for line in lines):
        return True
    try:
        with open(gitignore, "a", encoding="utf-8") as f:
            if lines and lines[-1].strip():
                f.write("\n")
            f.write(IGNORE_PATTERN + "\n")
    except OSError:
        return False
    return True


def cache_dir_for(repo: Path) -> Path:
    """Where this repository's cache belongs, creating the directory."""
    repo = repo.expanduser().resolve()
    if ensure_ignored(repo):
        directory = repo / CACHE_DIRNAME
        try:
            directory.mkdir(parents=True, exist_ok=True)
            return directory
        except OSError:
            pass  # read-only checkout: fall through to the user cache
    directory = USER_CACHE_ROOT / _repo_hash(repo)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def ask_cache() -> AnswerCache:
    """The cache for ad hoc asks, which belong to no repository."""
    directory = USER_CACHE_ROOT / "ask"
    directory.mkdir(parents=True, exist_ok=True)
    return AnswerCache(directory / DB_NAME)


class AnswerCache:
    """Key to noul, on disk. Misses are cheap; a corrupt database is not fatal."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS answers ("
            "  key TEXT PRIMARY KEY,"
            "  noul REAL NOT NULL,"
            "  created REAL NOT NULL)"
        )
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS choice_answers ("
            "  key TEXT PRIMARY KEY, distribution TEXT NOT NULL, created REAL NOT NULL)"
        )
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS route_answers ("
            "  key TEXT PRIMARY KEY, answer TEXT NOT NULL, created REAL NOT NULL)"
        )
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS score_answers ("
            "  key TEXT PRIMARY KEY, answer TEXT NOT NULL, created REAL NOT NULL)"
        )
        self._db.commit()

    @classmethod
    def for_repo(cls, repo: Path) -> AnswerCache:
        return cls(cache_dir_for(repo) / DB_NAME)

    def get(self, key: str) -> float | None:
        with self._lock:
            row = self._db.execute("SELECT noul FROM answers WHERE key = ?", (key,)).fetchone()
        return None if row is None else float(row[0])

    def get_many(self, keys: list[str]) -> dict[str, float]:
        found: dict[str, float] = {}
        for key in keys:
            value = self.get(key)
            if value is not None:
                found[key] = value
        return found

    def put(self, key: str, noul: float) -> None:
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO answers (key, noul, created) VALUES (?, ?, ?)",
                (key, float(noul), time.time()),
            )
            self._db.commit()

    def get_choice(self, key: str) -> dict[str, float] | None:
        with self._lock:
            row = self._db.execute("SELECT distribution FROM choice_answers WHERE key = ?", (key,)).fetchone()
        return None if row is None else json.loads(row[0])

    def put_choice(self, key: str, distribution: dict[str, float]) -> None:
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO choice_answers (key, distribution, created) VALUES (?, ?, ?)",
                (key, json.dumps(distribution, sort_keys=True), time.time()),
            )
            self._db.commit()

    def get_score(self, key: str) -> dict | None:
        """A stored Score answer. JSON keys are text, so the levels are cast back to int."""
        with self._lock:
            row = self._db.execute("SELECT answer FROM score_answers WHERE key = ?", (key,)).fetchone()
        if row is None:
            return None
        answer = json.loads(row[0])
        answer["probabilities"] = {int(k): float(v) for k, v in answer["probabilities"].items()}
        return answer

    def put_score(self, key: str, answer: dict) -> None:
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO score_answers (key, answer, created) VALUES (?, ?, ?)",
                (key, json.dumps(answer, sort_keys=True), time.time()),
            )
            self._db.commit()

    def get_route(self, key: str) -> dict | None:
        with self._lock:
            row = self._db.execute("SELECT answer FROM route_answers WHERE key = ?", (key,)).fetchone()
        return None if row is None else json.loads(row[0])

    def put_route(self, key: str, answer: dict) -> None:
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO route_answers (key, answer, created) VALUES (?, ?, ?)",
                (key, json.dumps(answer, sort_keys=True), time.time()),
            )
            self._db.commit()

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def __enter__(self) -> AnswerCache:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class NullCache:
    """A cache that never hits, for callers that ask for no caching."""

    def get(self, key: str) -> float | None:
        return None

    def get_many(self, keys: list[str]) -> dict[str, float]:
        return {}

    def put(self, key: str, noul: float) -> None:
        return None

    def get_choice(self, key: str) -> dict[str, float] | None:
        return None

    def put_choice(self, key: str, distribution: dict[str, float]) -> None:
        return None

    def get_score(self, key: str) -> dict | None:
        return None

    def put_score(self, key: str, answer: dict) -> None:
        return None

    def get_route(self, key: str) -> dict | None:
        return None

    def put_route(self, key: str, answer: dict) -> None:
        return None

    def close(self) -> None:
        return None
