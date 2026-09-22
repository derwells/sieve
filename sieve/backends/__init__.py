"""Search backends and the rule that picks one.

`SIEVE_SEARCH_BACKEND` wins if it is set. Otherwise `brave` is used whenever
`BRAVE_API_KEY` is in the environment, because it is the only backend with real
snippets and it costs cents rather than dollars; failing that, `claude`.
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

BACKEND_ENV = "SIEVE_SEARCH_BACKEND"
BACKENDS = {
    "brave": BraveBackend,
    "claude": ClaudeSearchBackend,
    "codex": CodexSearchBackend,
}

__all__ = [
    "BACKENDS",
    "BACKEND_ENV",
    "BRAVE_API_KEY_ENV",
    "BackendResult",
    "BraveBackend",
    "ClaudeSearchBackend",
    "CodexSearchBackend",
    "SearchBackend",
    "SearchBackendError",
    "SearchHit",
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
    return "brave" if source.get(BRAVE_API_KEY_ENV) else "claude"


def select_backend(name: str | None = None, env: dict[str, str] | None = None) -> SearchBackend:
    """Build the named backend, or the one the environment selects."""
    chosen = (name or "").strip().lower() or backend_name(env)
    if chosen not in BACKENDS:
        raise SearchBackendError(f"unknown search backend {chosen!r}; choose one of {', '.join(sorted(BACKENDS))}")
    return BACKENDS[chosen]()
