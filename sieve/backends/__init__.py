"""Search backends and the rule that picks one.

`SIEVE_SEARCH_BACKEND` wins if it is set. Otherwise `searxng` is used whenever
`SIEVE_SEARXNG_URL` points at an instance, then `brave` whenever `BRAVE_API_KEY`
is in the environment, since those two carry real snippets; failing both,
`claude`. The chosen backend is the only one called: a failure is reported,
never retried on another backend.
"""

from __future__ import annotations

import os

from .base import (
    BackendResult,
    SearchBackend,
    SearchBackendError,
    SearchHit,
    run_cli,
    timeout_seconds,
)
from .brave import API_KEY_ENV as BRAVE_API_KEY_ENV
from .brave import BraveBackend
from .claude_cli import ClaudeSearchBackend
from .codex_cli import CodexSearchBackend
from .searxng import URL_ENV as SEARXNG_URL_ENV
from .searxng import SearxngBackend

BACKEND_ENV = "SIEVE_SEARCH_BACKEND"
BACKENDS = {
    "brave": BraveBackend,
    "claude": ClaudeSearchBackend,
    "codex": CodexSearchBackend,
    "searxng": SearxngBackend,
}

__all__ = [
    "BACKENDS",
    "BACKEND_ENV",
    "BRAVE_API_KEY_ENV",
    "BackendResult",
    "BraveBackend",
    "ClaudeSearchBackend",
    "CodexSearchBackend",
    "SEARXNG_URL_ENV",
    "SearchBackend",
    "SearchBackendError",
    "SearchHit",
    "SearxngBackend",
    "backend_name",
    "run_cli",
    "select_backend",
    "timeout_seconds",
]


def backend_name(env: dict[str, str] | None = None) -> str:
    """Which backend the current environment selects."""
    source = os.environ if env is None else env
    chosen = (source.get(BACKEND_ENV) or "").strip().lower()
    if chosen:
        if chosen not in BACKENDS:
            raise SearchBackendError(
                f"{BACKEND_ENV}={chosen!r} is not a backend; choose one of {', '.join(sorted(BACKENDS))}"
            )
        return chosen
    if (source.get(SEARXNG_URL_ENV) or "").strip():
        return "searxng"
    return "brave" if source.get(BRAVE_API_KEY_ENV) else "claude"


def select_backend(name: str | None = None, env: dict[str, str] | None = None) -> SearchBackend:
    """Build the named backend, or the one the environment selects."""
    chosen = (name or "").strip().lower() or backend_name(env)
    if chosen not in BACKENDS:
        raise SearchBackendError(f"unknown search backend {chosen!r}; choose one of {', '.join(sorted(BACKENDS))}")
    if chosen == "searxng" and env is not None:
        return SearxngBackend(env=env)
    return BACKENDS[chosen]()
