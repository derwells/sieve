"""Headless Codex backend. WARNING: results here are model-transcribed.

`codex exec` never puts search results on the event stream. The `web_search`
`item.completed` event carries only `action.query`; the actual links exist only
in the `agent_message` the model writes afterwards. So sieve prompts for a bare
markdown bullet list and parses that, which means titles are the model's
rendering of the page titles, not verbatim, and urls can in principle be
mistyped. Snippets do not exist at all. Prefer `brave` or `claude` when
fidelity matters.

Verified 2026-09-22 against codex-cli 0.154.0: `codex exec` has no `--search`
flag; web search is turned on with `-c tools.web_search=true`.
"""

from __future__ import annotations

import json
import re
import time

from .base import (
    BackendResult,
    SearchBackendError,
    SearchHit,
    command_override,
    run_cli,
    run_in_scratch,
    stderr_tail,
    timeout_seconds,
)

BASE_COMMAND = ["codex", "exec", "-c", "tools.web_search=true", "--json", "--skip-git-repo-check"]

MARKDOWN_LINK = re.compile(r"^\s*[-*]\s*\[(?P<title>[^\]]*)\]\(\s*(?P<url>https?://[^\s)]+)\s*\)")
BARE_URL = re.compile(r"^\s*[-*]\s*<?(?P<url>https?://[^\s>]+)>?\s*$")


def build_prompt(query: str) -> str:
    return (
        f"Search the web once for: {query}\n"
        "Then answer with ONLY a markdown bullet list of the results, one per line, "
        "in the form - [title](url). No preamble, no commentary, no other text."
    )


def build_command(prompt: str, override: list[str] | None = None) -> list[str]:
    return [*(override or BASE_COMMAND), prompt]


def parse_markdown_links(text: str) -> list[SearchHit]:
    """`- [title](url)` lines, or bare `- url` lines, in document order."""
    hits: list[SearchHit] = []
    for line in text.splitlines():
        match = MARKDOWN_LINK.match(line)
        if match:
            hits.append(SearchHit(url=match.group("url"), title=match.group("title").strip()))
            continue
        match = BARE_URL.match(line)
        if match:
            hits.append(SearchHit(url=match.group("url")))
    return hits


def parse_codex_stream(stdout: str) -> tuple[list[SearchHit], dict]:
    """Hits from the last agent_message that contains links, plus turn usage."""
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
        if event.get("type") == "item.completed":
            item = event.get("item") or {}
            if item.get("type") == "web_search":
                searches += 1
            elif item.get("type") == "agent_message":
                found = parse_markdown_links(str(item.get("text") or ""))
                if found:
                    hits = found
        elif event.get("type") == "turn.completed":
            usage = dict(event.get("usage") or {})
    usage["searches"] = searches
    return hits, usage


class CodexSearchBackend:
    """Spawns `codex exec` headless and parses the markdown list it writes."""

    name = "codex"

    def __init__(self, *, runner=run_cli, timeout: float | None = None, command: list[str] | None = None) -> None:
        self.runner = runner
        self.timeout = timeout if timeout is not None else timeout_seconds()
        self.command = command if command is not None else command_override()

    async def search(self, query: str, count: int) -> BackendResult:
        started = time.monotonic()
        argv = build_command(build_prompt(query), self.command)
        code, stdout, stderr = await run_in_scratch(argv, timeout=self.timeout, runner=self.runner)
        if code != 0:
            raise SearchBackendError(f"codex search exited {code}: {stderr_tail(stderr)}")
        hits, usage = parse_codex_stream(stdout)
        if not hits:
            raise SearchBackendError(
                f"codex search returned no parseable links for {query!r}: {stderr_tail(stderr)}"
            )
        return BackendResult(
            query=query,
            hits=hits[:count],
            usage=usage,
            wall_seconds=round(time.monotonic() - started, 3),
        )
