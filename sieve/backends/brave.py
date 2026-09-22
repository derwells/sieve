"""Brave Search API backend: one HTTP GET, verbatim titles and descriptions.

This is the only backend whose snippets are real: Brave returns a description
per result, so nothing is transcribed or invented by a model.
"""

from __future__ import annotations

import os
import time

import httpx2

from .base import BackendResult, SearchBackendError, SearchHit, timeout_seconds

ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
API_KEY_ENV = "BRAVE_API_KEY"
#: Brave rejects a count above this.
MAX_COUNT = 20


def parse_brave_payload(payload: dict) -> list[SearchHit]:
    """`web.results[]` → hits. Anything without a url is skipped."""
    results = (payload.get("web") or {}).get("results") or []
    hits: list[SearchHit] = []
    for entry in results:
        if not isinstance(entry, dict):
            continue
        url = (entry.get("url") or "").strip()
        if not url:
            continue
        hits.append(
            SearchHit(
                url=url,
                title=(entry.get("title") or "").strip(),
                snippet=(entry.get("description") or "").strip(),
            )
        )
    return hits


class BraveBackend:
    """Brave's web search endpoint. The key is read from the environment only."""

    name = "brave"

    def __init__(self, api_key: str | None = None, *, transport=None, timeout: float | None = None) -> None:
        self.api_key = api_key or os.environ.get(API_KEY_ENV)
        self.transport = transport
        self.timeout = timeout if timeout is not None else timeout_seconds()

    async def search(self, query: str, count: int) -> BackendResult:
        if not self.api_key:
            raise SearchBackendError(
                f"{API_KEY_ENV} is not set; the brave backend needs it. Set it or choose "
                f"another backend with SIEVE_SEARCH_BACKEND."
            )
        started = time.monotonic()
        params = {"q": query, "count": max(1, min(int(count), MAX_COUNT))}
        headers = {"X-Subscription-Token": self.api_key, "Accept": "application/json"}
        async with httpx2.AsyncClient(transport=self.transport, timeout=self.timeout) as client:
            response = await client.get(ENDPOINT, params=params, headers=headers)
            if response.status_code != 200:
                raise SearchBackendError(
                    f"brave search returned HTTP {response.status_code}: {response.text[:300]}"
                )
            payload = response.json()
        hits = parse_brave_payload(payload)
        return BackendResult(
            query=query,
            hits=hits,
            usage={"requests": 1, "results": len(hits)},
            wall_seconds=round(time.monotonic() - started, 3),
        )
