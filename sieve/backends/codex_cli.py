"""Headless Codex backend: hits read from the observed web_search events.

Since codex-cli 0.156.1 (checked 2026-09-25), each `web_search` `item.completed`
event carries `results[]`: `{type: text_result, ref_id, url, title, snippet,
domain}`. A `search` action's results have `ref_id` `turnNsearchM` and a real
snippet. Opened pages (`open_page`, `find_in_page`, `other`; `ref_id`
`turnNviewM`) carry a real url and title, but their snippet is a placeholder
such as "Total lines: 1227", which is dropped. Hits come from these events, in
the order they were observed.

Older releases (0.154.0) put only the query on the event, so when no event has
results, sieve falls back to the markdown bullet list the model writes. Those
titles are the model's transcription and snippets are empty; `usage.hits_source`
says which path produced the hits.

`codex exec` has no `--search` flag; web search is turned on with
`-c tools.web_search=true`. It reads stdin when stdin is not a terminal, so the
child gets an empty stdin.
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
#: The snippet an opened page carries instead of text.
PLACEHOLDER_SNIPPET = re.compile(r"^\s*Total lines:\s*\d+\s*$", re.IGNORECASE)


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


def parse_event_results(item: dict) -> tuple[list[SearchHit], int, int]:
    """Hits from one completed web_search item: (hits, search results, opened pages)."""
    action = (item.get("action") or {}).get("type")
    hits: list[SearchHit] = []
    searched = opened = 0
    for result in item.get("results") or []:
        if not isinstance(result, dict):
            continue
        url = str(result.get("url") or "").strip()
        if not url.lower().startswith(("http://", "https://")):
            continue
        ref = str(result.get("ref_id") or "")
        is_view = "view" in ref or (not ref and action != "search")
        snippet = " ".join(str(result.get("snippet") or "").split())
        if is_view or PLACEHOLDER_SNIPPET.match(snippet):
            snippet = ""
        opened += is_view
        searched += not is_view
        hits.append(SearchHit(url=url, title=" ".join(str(result.get("title") or "").split()), snippet=snippet))
    return hits, searched, opened


def _merge(hits: list[SearchHit]) -> list[SearchHit]:
    """One hit per url, first seen first; a later snippet or title fills an empty one."""
    merged: dict[str, SearchHit] = {}
    for hit in hits:
        key = hit.url.rstrip("/")
        seen = merged.get(key)
        if seen is None:
            merged[key] = hit
        elif (not seen.snippet and hit.snippet) or (not seen.title and hit.title):
            merged[key] = SearchHit(url=seen.url, title=seen.title or hit.title, snippet=seen.snippet or hit.snippet)
    return list(merged.values())


def parse_codex_stream(stdout: str) -> tuple[list[SearchHit], dict]:
    """Hits from observed web_search results, else from the model's link list; plus usage."""
    observed: list[SearchHit] = []
    transcribed: list[SearchHit] = []
    usage: dict = {}
    searches = searched = opened = 0
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
                found, s, o = parse_event_results(item)
                observed.extend(found)
                searched += s
                opened += o
            elif item.get("type") == "agent_message":
                found = parse_markdown_links(str(item.get("text") or ""))
                if found:
                    transcribed = found
        elif event.get("type") == "turn.completed":
            usage = dict(event.get("usage") or {})
    usage["searches"] = searches
    if observed:
        hits = _merge(observed)
        seen = {hit.url.rstrip("/") for hit in hits}
        usage.update(
            hits_source="events",
            search_results=searched,
            opened_pages=opened,
            unobserved_links=sum(hit.url.rstrip("/") not in seen for hit in transcribed),
        )
        return hits, usage
    usage["hits_source"] = "transcript" if transcribed else "none"
    return _merge(transcribed), usage


class CodexSearchBackend:
    """Spawns `codex exec` headless and reads the web_search results it observed."""

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
