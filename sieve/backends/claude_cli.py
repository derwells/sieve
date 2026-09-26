"""Headless Claude Code backend: the WebSearch tool results, read off the stream.

`claude -p ... --output-format stream-json` emits JSONL. The WebSearch tool's
own output arrives on a `type: "user"` line under `tool_use_result`, already
parsed into `{"title", "url"}` dicts, so titles and urls here are verbatim —
no model transcribes them. WebSearch carries no snippet, so snippets are empty.

Cost floor, measured 2026-09-22: Claude Code's own system prompt and tool
schemas are ~65k tokens, about $0.05 and ~13 s per call, even with MCP, settings
and extra tools stripped. Note also that `usage.server_tool_use
.web_search_requests` came back 0 despite results arriving; count tool results
instead.

The model is `SIEVE_CLAUDE_SEARCH_MODEL` (default `claude-haiku-4-5-20251001`);
`SIEVE_SEARCH_CMD_CLAUDE` replaces the whole command line. The child runs with
the CLI's own login: endpoint, key and model routing variables are stripped.
"""

from __future__ import annotations

import json
import os
import time

from .base import (
    BackendResult,
    SearchBackendError,
    SearchHit,
    command_override,
    fingerprint,
    run_cli,
    run_in_scratch,
    stderr_tail,
    timeout_seconds,
)

MODEL_ENV = "SIEVE_CLAUDE_SEARCH_MODEL"
DEFAULT_MODEL = "claude-haiku-4-5-20251001"


def base_command(model: str = DEFAULT_MODEL) -> list[str]:
    """The flags that strip Claude Code down to one WebSearch call."""
    return [*BASE_COMMAND[:3], model, *BASE_COMMAND[4:]]


BASE_COMMAND = [
    "claude",
    "-p",
    "--model",
    DEFAULT_MODEL,
    "--allowedTools",
    "WebSearch",
    "--output-format",
    "stream-json",
    "--verbose",
    "--max-turns",
    "3",
    "--strict-mcp-config",
    "--mcp-config",
    '{"mcpServers":{}}',
    "--setting-sources",
    "",
]
#: Haiku spends one turn on a ToolSearch call before the real WebSearch, so
#: three turns buys exactly one search.
LINKS_MARKER = "Links: "


def build_prompt(query: str) -> str:
    return f"Use the WebSearch tool exactly once with the query: {query}\nThen stop."


def build_command(prompt: str, override: list[str] | None = None) -> list[str]:
    """The full argv. An override replaces every flag; the prompt is appended."""
    return [*(override or BASE_COMMAND), prompt]


def _hits_from_tool_use_result(payload: dict) -> list[SearchHit]:
    hits: list[SearchHit] = []
    for result in payload.get("results") or []:
        if not isinstance(result, dict):
            continue
        for entry in result.get("content") or []:
            if isinstance(entry, dict) and entry.get("url"):
                hits.append(SearchHit(url=str(entry["url"]), title=str(entry.get("title") or "")))
    return hits


def _hits_from_links_text(text: str) -> list[SearchHit]:
    """Fallback: `Web search results for query: "..."\\n\\nLinks: [<json>]`."""
    marker = text.find(LINKS_MARKER)
    if marker < 0:
        return []
    try:
        entries = json.loads(text[marker + len(LINKS_MARKER) :].strip())
    except json.JSONDecodeError:
        return []
    if not isinstance(entries, list):
        return []
    return [
        SearchHit(url=str(e["url"]), title=str(e.get("title") or ""))
        for e in entries
        if isinstance(e, dict) and e.get("url")
    ]


def parse_claude_stream(stdout: str) -> tuple[list[SearchHit], dict]:
    """Hits in stream order plus the usage recorded on the final result line."""
    hits: list[SearchHit] = []
    usage: dict = {}
    searches = 0
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        kind = event.get("type")
        if kind == "user":
            found: list[SearchHit] = []
            tool_use_result = event.get("tool_use_result")
            if isinstance(tool_use_result, dict) and tool_use_result.get("results"):
                found = _hits_from_tool_use_result(tool_use_result)
            if not found:
                content = (event.get("message") or {}).get("content")
                if isinstance(content, list) and content:
                    inner = content[0].get("content") if isinstance(content[0], dict) else None
                    if isinstance(inner, str):
                        found = _hits_from_links_text(inner)
            if found:
                searches += 1
                hits.extend(found)
        elif kind == "result":
            reported = event.get("usage") or {}
            usage = {
                "total_cost_usd": event.get("total_cost_usd"),
                "duration_ms": event.get("duration_ms"),
                "input_tokens": reported.get("input_tokens"),
                "output_tokens": reported.get("output_tokens"),
                "cache_read_input_tokens": reported.get("cache_read_input_tokens"),
                "cache_creation_input_tokens": reported.get("cache_creation_input_tokens"),
            }
    usage["searches"] = searches
    return hits, usage


class ClaudeSearchBackend:
    """Spawns `claude -p` headless and reads WebSearch results off the stream."""

    name = "claude"

    def __init__(
        self,
        *,
        runner=run_cli,
        timeout: float | None = None,
        command: list[str] | None = None,
        model: str | None = None,
        env: dict[str, str] | None = None,
        generic_override: bool = True,
    ) -> None:
        source = os.environ if env is None else env
        self.env = env
        self.runner = runner
        self.timeout = timeout if timeout is not None else timeout_seconds(source)
        self.command = command if command is not None else command_override(source, "claude", generic=generic_override)
        self.model = model or source.get(MODEL_ENV) or DEFAULT_MODEL

    def argv_template(self) -> list[str]:
        return self.command or base_command(self.model)

    def config(self) -> dict:
        return {
            "route": self.name,
            "model": None if self.command is not None else self.model,
            "effort": None,
            "cmd_fingerprint": fingerprint(self.argv_template()),
            "url": None,
        }

    def fingerprint(self) -> str:
        return self.config()["cmd_fingerprint"]

    async def search(self, query: str, count: int) -> BackendResult:
        started = time.monotonic()
        argv = [*self.argv_template(), build_prompt(query)]
        code, stdout, stderr = await run_in_scratch(argv, timeout=self.timeout, runner=self.runner, env=self.env)
        if code != 0:
            raise SearchBackendError(f"claude search exited {code}: {stderr_tail(stderr)}")
        hits, usage = parse_claude_stream(stdout)
        usage["provenance"] = "observed"  # WebSearch tool results, never model text
        if not hits:
            raise SearchBackendError(
                f"claude search returned no WebSearch results for {query!r}: {stderr_tail(stderr)}"
            )
        return BackendResult(
            query=query,
            hits=hits[:count],
            usage=usage,
            wall_seconds=round(time.monotonic() - started, 3),
        )
