"""The small interface every search backend implements.

A backend takes one query string and returns the hits it found, plus whatever
usage and timing the underlying service reported. It never ranks, dedupes, or
rewrites: that all happens in `sieve.search`.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shlex
import tempfile
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from ..errors import SieveError

#: How long one backend call may take before it is killed.
DEFAULT_TIMEOUT_SECONDS = 90.0
TIMEOUT_ENV = "SIEVE_SEARCH_TIMEOUT"
COMMAND_ENV = "SIEVE_SEARCH_CMD"


class SearchBackendError(SieveError):
    """A search backend failed and produced no usable results."""


@dataclass(frozen=True)
class SearchHit:
    """One result as the backend reported it, before canonicalisation.

    `engines` names the upstream engines that returned it, where a metasearch
    backend reports them; `published` is the date string an engine gave, if any.
    """

    url: str
    title: str = ""
    snippet: str = ""
    engines: tuple[str, ...] = ()
    published: str = ""


@dataclass
class BackendResult:
    """What one backend call returned, with its own usage and wall time."""

    query: str
    hits: list[SearchHit] = field(default_factory=list)
    usage: dict = field(default_factory=dict)
    wall_seconds: float = 0.0


@runtime_checkable
class SearchBackend(Protocol):
    """Anything that can answer one query with a list of hits."""

    name: str

    async def search(self, query: str, count: int) -> BackendResult:
        """Hits for `query`. `count` is how many are wanted; a backend may cap it,
        or return a few more when they arrived in the same response."""
        ...


def timeout_seconds(env: dict[str, str] | None = None) -> float:
    """The per-call timeout, from `SIEVE_SEARCH_TIMEOUT` or the default."""
    source = os.environ if env is None else env
    raw = source.get(TIMEOUT_ENV)
    if not raw:
        return DEFAULT_TIMEOUT_SECONDS
    try:
        value = float(raw)
    except ValueError:
        return DEFAULT_TIMEOUT_SECONDS
    return value if value > 0 else DEFAULT_TIMEOUT_SECONDS


def command_override(
    env: dict[str, str] | None = None, route: str | None = None, *, generic: bool = True
) -> list[str] | None:
    """The command line override, split with shlex, or None when none is set.

    `SIEVE_SEARCH_CMD_<ROUTE>` (for example `SIEVE_SEARCH_CMD_CODEX`) applies to
    one backend and wins. `SIEVE_SEARCH_CMD` applies to whichever CLI backend
    runs; the search chain passes `generic=False`, since it runs several.
    """
    source = os.environ if env is None else env
    candidates = [f"{COMMAND_ENV}_{route.upper()}"] if route else []
    if generic:
        candidates.append(COMMAND_ENV)
    for name in candidates:
        raw = source.get(name)
        if raw and raw.strip():
            return shlex.split(raw)
    return None


async def run_cli(argv: list[str], *, cwd: str, env: dict[str, str], timeout: float) -> tuple[int, str, str]:
    """Run a CLI to completion, killing it if it outlives `timeout`."""
    # stdin is closed: inside the MCP server it would be the JSON-RPC pipe, and
    # `codex exec` waits on a non-terminal stdin.
    process = await asyncio.create_subprocess_exec(
        *argv,
        cwd=cwd,
        env=env,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
    except (TimeoutError, asyncio.TimeoutError):
        process.kill()
        await process.wait()
        raise SearchBackendError(
            f"{argv[0]} search timed out after {timeout:.0f}s; set {TIMEOUT_ENV} to raise the limit"
        ) from None
    return process.returncode or 0, stdout.decode("utf-8", "replace"), stderr.decode("utf-8", "replace")


#: Variables that reroute a CLI to another model endpoint, key or model tier.
#: A wrapper that points Claude Code at a third-party Anthropic-compatible
#: endpoint sets these; a search child inheriting them would send the hosted
#: search to that endpoint, possibly billed per token. `CLAUDECODE` marks a
#: nested session and changes the CLI's behaviour.
ROUTING_VARIABLES = frozenset(
    {
        "CLAUDECODE",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_BASE_URL",
        "ANTHROPIC_MODEL",
        "ANTHROPIC_SMALL_FAST_MODEL",
        "API_TIMEOUT_MS",
        "CLAUDE_CODE_SUBAGENT_MODEL",
        "CLAUDE_CODE_MAX_CONTEXT_TOKENS",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
    }
)
ROUTING_PREFIXES = ("ANTHROPIC_", "CLAUDE_CODE_USE_")
#: Comma-separated extra prefixes to strip, such as a local wrapper's own variables.
STRIP_PREFIXES_ENV = "SIEVE_STRIP_ENV_PREFIXES"


def cli_environment(env: dict[str, str] | None = None) -> dict[str, str]:
    """A copy of the environment without model routing overrides.

    The CLI then uses its own login and default endpoint. `ROUTING_VARIABLES`,
    every variable starting with `ROUTING_PREFIXES`, and every variable starting
    with a prefix listed in `SIEVE_STRIP_ENV_PREFIXES` are removed.
    """
    source = dict(os.environ if env is None else env)
    extra = tuple(p.strip() for p in (source.get(STRIP_PREFIXES_ENV) or "").split(",") if p.strip())
    prefixes = ROUTING_PREFIXES + extra
    return {
        key: value
        for key, value in source.items()
        if key not in ROUTING_VARIABLES and not key.startswith(prefixes)
    }


def fingerprint(argv: list[str]) -> str:
    """A short, stable identity for a command line template."""
    return hashlib.sha256(json.dumps(argv).encode("utf-8")).hexdigest()[:12]


def stderr_tail(stderr: str, limit: int = 600) -> str:
    text = stderr.strip()
    return text[-limit:] if len(text) > limit else text


async def run_in_scratch(
    argv: list[str], *, timeout: float, runner=run_cli, env: dict[str, str] | None = None
) -> tuple[int, str, str]:
    """Run a CLI in a fresh empty directory with routing overrides stripped."""
    with tempfile.TemporaryDirectory(prefix="sieve-search-") as workdir:
        return await runner(argv, cwd=workdir, env=cli_environment(env), timeout=timeout)
