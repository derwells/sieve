"""SearXNG backend: a self-hosted metasearch instance's JSON API.

SearXNG fans one query out to its configured engines and merges their results,
so a single page usually carries 20-40 hits with the engines' own titles and
snippets. This backend pages through `/search?format=json` until it has the
hits it was asked for, a page adds nothing new, the page cap is reached, or the
deadline passes. It never pads: a short answer is returned short, with the
engines SearXNG reported as unresponsive.

The instance must have `json` in `search.formats`; `bin/sieve-searxng` writes
settings that do.
"""

from __future__ import annotations

import os
import time

import httpx2

from .base import BackendResult, SearchBackendError, SearchHit

URL_ENV = "SIEVE_SEARXNG_URL"
TIMEOUT_ENV = "SIEVE_SEARXNG_TIMEOUT"
MAX_PAGES_ENV = "SIEVE_SEARXNG_MAX_PAGES"
DEADLINE_ENV = "SIEVE_SEARXNG_DEADLINE"

DEFAULT_URL = "http://127.0.0.1:8888"
#: One HTTP request to the instance. SearXNG's own engine timeout sits below it.
DEFAULT_TIMEOUT_SECONDS = 12.0
#: Pages fetched for one query, at most. A hard ceiling stops a bad env value.
DEFAULT_MAX_PAGES = 3
MAX_PAGES_CEILING = 5
#: Wall time for one query across every page and retry.
DEFAULT_DEADLINE_SECONDS = 30.0
#: Page 1 is retried once on a connection error, timeout or 5xx. Later pages are not.
FIRST_PAGE_RETRIES = 1


def _env_float(source: dict[str, str], name: str, default: float) -> float:
    try:
        value = float(source.get(name) or default)
    except ValueError:
        return default
    return value if value > 0 else default


def _env_int(source: dict[str, str], name: str, default: int, ceiling: int) -> int:
    try:
        value = int(source.get(name) or default)
    except ValueError:
        return default
    return max(1, min(value, ceiling))


def parse_searxng_payload(payload: dict) -> list[SearchHit]:
    """`results[]` → hits. Entries without an http(s) url are skipped.

    `content` is the snippet the engines returned; `engines` is every engine
    that found the page, which is kept as provenance.
    """
    hits: list[SearchHit] = []
    for entry in payload.get("results") or []:
        if not isinstance(entry, dict):
            continue
        url = (entry.get("url") or "").strip()
        if not url.lower().startswith(("http://", "https://")):
            continue
        engines = entry.get("engines")
        if not isinstance(engines, list) or not engines:
            engines = [entry["engine"]] if entry.get("engine") else []
        hits.append(
            SearchHit(
                url=url,
                title=" ".join(str(entry.get("title") or "").split()),
                snippet=" ".join(str(entry.get("content") or "").split()),
                engines=tuple(str(e) for e in engines),
                published=str(entry.get("publishedDate") or ""),
            )
        )
    return hits


def unresponsive_engines(payload: dict) -> list[dict[str, str]]:
    """`unresponsive_engines` is a list of `[engine, reason]` pairs."""
    found = []
    for entry in payload.get("unresponsive_engines") or []:
        if isinstance(entry, (list, tuple)) and entry:
            found.append({"engine": str(entry[0]), "reason": str(entry[1]) if len(entry) > 1 else ""})
    return found


def _url_key(url: str) -> str:
    return url.strip().rstrip("/").lower()


class SearxngBackend:
    """Pages a SearXNG instance's JSON API. No key; the instance is the caller's."""

    name = "searxng"
    #: Deeper results cost another page, not another engine fan-out per variant.
    paginates = True

    def __init__(
        self,
        base_url: str | None = None,
        *,
        transport=None,
        timeout: float | None = None,
        max_pages: int | None = None,
        deadline: float | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        source = os.environ if env is None else env
        self.base_url = (base_url or source.get(URL_ENV) or DEFAULT_URL).rstrip("/")
        self.transport = transport
        self.timeout = timeout if timeout is not None else _env_float(source, TIMEOUT_ENV, DEFAULT_TIMEOUT_SECONDS)
        self.max_pages = (
            max(1, min(int(max_pages), MAX_PAGES_CEILING))
            if max_pages is not None
            else _env_int(source, MAX_PAGES_ENV, DEFAULT_MAX_PAGES, MAX_PAGES_CEILING)
        )
        self.deadline = deadline if deadline is not None else _env_float(source, DEADLINE_ENV, DEFAULT_DEADLINE_SECONDS)

    def fingerprint(self) -> str:
        """What makes two calls with the same query comparable, for the retrieval cache."""
        return f"{self.base_url}|pages={self.max_pages}"

    def config(self) -> dict:
        return {"route": self.name, "model": None, "effort": None, "cmd_fingerprint": None, "url": self.base_url}

    async def _page(self, client, query: str, pageno: int) -> dict:
        params = {"q": query, "format": "json", "pageno": pageno, "categories": "general"}
        response = await client.get(f"{self.base_url}/search", params=params, headers={"Accept": "application/json"})
        if response.status_code == 403:
            raise SearchBackendError(
                f"searxng at {self.base_url} returned HTTP 403; enable the json format under search.formats"
            )
        if response.status_code != 200:
            raise SearchBackendError(f"searxng returned HTTP {response.status_code}: {response.text[:300]}")
        try:
            payload = response.json()
        except ValueError as e:
            raise SearchBackendError(f"searxng returned non-JSON: {response.text[:200]}") from e
        if not isinstance(payload, dict):
            raise SearchBackendError("searxng returned a JSON value that is not an object")
        return payload

    async def _first_page(self, client, query: str) -> tuple[dict, int]:
        attempts = 0
        while True:
            attempts += 1
            try:
                return await self._page(client, query, 1), attempts
            except (httpx2.TransportError, httpx2.TimeoutException) as e:
                error = SearchBackendError(f"searxng at {self.base_url} unreachable: {type(e).__name__}: {e}")
            except SearchBackendError as e:
                if "HTTP 5" not in str(e):
                    raise
                error = e
            if attempts > FIRST_PAGE_RETRIES:
                raise error

    async def search(self, query: str, count: int) -> BackendResult:
        """At least `count` hits if the instance has them, in page order.

        Every hit on a fetched page is returned, so the result can run past
        `count` by the rest of the last page; the caller trims the pool.
        """
        started = time.monotonic()
        want = max(1, int(count))
        hits: list[SearchHit] = []
        seen: set[str] = set()
        failures: dict[str, str] = {}
        page_errors: list[str] = []
        pages = 0
        requests = 0
        stopped = "count"

        async with httpx2.AsyncClient(transport=self.transport, timeout=self.timeout) as client:
            for pageno in range(1, self.max_pages + 1):
                if pageno > 1 and time.monotonic() - started >= self.deadline:
                    stopped = "deadline"
                    break
                try:
                    if pageno == 1:
                        payload, attempts = await self._first_page(client, query)
                        requests += attempts
                    else:
                        requests += 1
                        payload = await self._page(client, query, pageno)
                except (httpx2.TransportError, httpx2.TimeoutException, SearchBackendError) as e:
                    if pageno == 1:
                        raise  # _first_page already turned transport errors into SearchBackendError
                    page_errors.append(f"page {pageno}: {type(e).__name__}: {e}")
                    stopped = "page_error"
                    break
                pages += 1
                for item in unresponsive_engines(payload):
                    failures.setdefault(item["engine"], item["reason"])
                new = 0
                for hit in parse_searxng_payload(payload):
                    key = _url_key(hit.url)
                    if key in seen:
                        continue
                    seen.add(key)
                    hits.append(hit)
                    new += 1
                if len(hits) >= want:
                    stopped = "count"
                    break
                if new == 0:
                    stopped = "exhausted"
                    break
            else:
                stopped = "page_cap"

        if not hits:
            detail = ", ".join(f"{e} ({r})" for e, r in failures.items()) or "no engine returned results"
            raise SearchBackendError(f"searxng returned no results for {query!r}: {detail}")

        usage = {
            "requests": requests,
            "pages": pages,
            "results": len(hits),
            "stopped": stopped,
            "provenance": "observed",  # the engines' own results, via the JSON API
            "unresponsive_engines": [{"engine": e, "reason": r} for e, r in failures.items()],
        }
        if page_errors:
            usage["page_errors"] = page_errors
        return BackendResult(
            query=query,
            hits=hits,
            usage=usage,
            wall_seconds=round(time.monotonic() - started, 3),
        )
