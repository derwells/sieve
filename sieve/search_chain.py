"""An ordered chain of search routes, with one cache and honest provenance.

    python -m sieve.search_chain search --routes codex,claude,searxng --count 10 --json QUERY
    python -m sieve.search_chain preflight --routes codex,claude,searxng --json QUERY

Routes are tried in the order given. For each route, its cache entry is read,
then a live call is made, and only if both come up empty does the next route
run, so a later route's cache is never served while an earlier route works.
The result says which route answered, which earlier routes failed and why,
whether it was served by a fallback after a hosted route failed (`degraded`),
and whether the hits came from the provider's own search results (`observed`)
or were parsed from text a model wrote (`transcribed`).

Hosted routes (`codex`, `claude`) spawn the CLIs with their own login; model,
endpoint and key routing variables are stripped from the child environment
(`sieve.backends.base.cli_environment`). Nothing here falls back to a paid API.

The cache key is the route, its effective configuration (model, effort,
command fingerprint, SearXNG URL), the normalised query and the count. Entries
expire after `SIEVE_SEARCH_CACHE_TTL` seconds (default one hour).
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
import time
from pathlib import Path

from .backends import (
    BackendResult,
    ClaudeSearchBackend,
    CodexSearchBackend,
    SearchBackendError,
    SearxngBackend,
)
from .search_cache import CACHE_DIR, SearchCache, ttl_seconds

ROUTES = ("codex", "claude", "searxng")
HOSTED_ROUTES = frozenset({"codex", "claude"})
DEFAULT_ROUTES = ("codex", "claude", "searxng")
#: Chain entries live beside, not inside, the jev_search retrieval cache.
CHAIN_CACHE_DIR = CACHE_DIR.parent / "search-chain"
CACHE_VERSION = 1


def parse_routes(routes) -> list[str]:
    """A route list from a list or a comma string; unknown or repeated routes are errors."""
    if routes is None:
        return list(DEFAULT_ROUTES)
    if isinstance(routes, str):
        routes = routes.split(",")
    names = [r.strip().lower() for r in routes if r and r.strip()]
    unknown = [r for r in names if r not in ROUTES]
    if unknown:
        raise ValueError(f"unknown route(s) {unknown}; choose from {', '.join(ROUTES)}")
    if len(set(names)) != len(names):
        raise ValueError(f"route listed twice: {names}")
    if not names:
        raise ValueError("no routes given")
    return names


def build_backend(route: str, env: dict[str, str] | None = None):
    """The backend for one route, configured from `env` (the process environment by default)."""
    if route == "codex":
        return CodexSearchBackend(env=env, generic_override=False)
    if route == "claude":
        return ClaudeSearchBackend(env=env, generic_override=False)
    if route == "searxng":
        return SearxngBackend(env=env)
    raise ValueError(f"unknown route {route!r}")


def normalise_query(query: str) -> str:
    return " ".join(query.split()).casefold()


def cache_key(config: dict, query: str, count: int) -> str:
    payload = {"v": CACHE_VERSION, "config": config, "query": normalise_query(query), "count": int(count)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _is_http(url: str) -> bool:
    return url.strip().lower().startswith(("http://", "https://"))


def _hits(result: BackendResult, count: int) -> list[dict]:
    rows = []
    for hit in result.hits:
        if not _is_http(hit.url):
            continue
        row = {"rank": len(rows) + 1, "title": hit.title, "url": hit.url, "snippet": hit.snippet}
        if hit.engines:
            row["engines"] = list(hit.engines)
        rows.append(row)
        if len(rows) == count:
            break
    return rows


def _provenance(result: BackendResult) -> tuple[bool, bool]:
    """(observed, transcribed) for one backend result."""
    kind = result.usage.get("provenance")
    return kind == "observed", kind == "transcribed"


def _open_cache(cache_dir, env) -> SearchCache:
    ttl = ttl_seconds(env)
    return SearchCache(Path(cache_dir).expanduser() if cache_dir else CHAIN_CACHE_DIR, ttl=ttl)


async def _live(backend, query: str, count: int) -> BackendResult:
    try:
        return await backend.search(query, count)
    except SearchBackendError:
        raise
    except Exception as e:  # one broken route must not end the chain
        raise SearchBackendError(f"{type(e).__name__}: {e}") from e


async def asearch(
    query: str,
    count: int = 10,
    routes=None,
    cache_dir=None,
    env: dict[str, str] | None = None,
    *,
    backends: dict | None = None,
) -> dict:
    """Search `routes` in order and return the first route's hits that works.

    `backends` maps a route to a ready backend, for tests; otherwise each route's
    backend is built from `env`.
    """
    started = time.monotonic()
    if not query or not query.strip():
        raise ValueError("empty query")
    count = max(1, int(count))
    names = parse_routes(routes)
    errors: dict[str, str] = {}
    cache = _open_cache(cache_dir, env)
    try:
        for route in names:
            backend = (backends or {}).get(route) or build_backend(route, env)
            config = backend.config()
            key = cache_key(config, query, count)
            result = cache.get(key)
            cache_state = "hit"
            if result is None:
                cache_state = "miss"
                try:
                    result = await _live(backend, query, count)
                except SearchBackendError as e:
                    errors[route] = str(e)
                    continue
            hits = _hits(result, count)
            if not hits:
                errors[route] = "no http(s) results"
                continue
            if cache_state == "miss":
                cache.put(key, result)
            observed, transcribed = _provenance(result)
            failed_hosted = [r for r in errors if r in HOSTED_ROUTES]
            return {
                "hits": hits,
                "route": route,
                "fallback_from": next(iter(errors), None),
                "errors": errors,
                "degraded": bool(failed_hosted),
                "observed": observed,
                "transcribed": transcribed,
                "config": config,
                "cache": cache_state,
                "seconds": round(time.monotonic() - started, 3),
            }
    finally:
        cache.close()
    return {
        "hits": [],
        "route": None,
        "fallback_from": next(iter(errors), None),
        "errors": errors,
        "degraded": any(r in HOSTED_ROUTES for r in errors),
        "observed": False,
        "transcribed": False,
        "config": None,
        "cache": "miss",
        "seconds": round(time.monotonic() - started, 3),
    }


async def _preflight_route(route, backend, query, count, cache) -> dict:
    started = time.monotonic()
    config = backend.config()
    entry = {"ok": False, "results": 0, "observed": False, "error": None, "seconds": 0.0, "config": config}
    try:
        result = await _live(backend, query, count)
    except SearchBackendError as e:
        entry["error"] = str(e)
    else:
        observed, _ = _provenance(result)
        usable = [h for h in result.hits if _is_http(h.url) and h.title.strip()]
        entry["results"] = len(result.hits)
        entry["observed"] = observed
        if not observed:
            entry["error"] = "hits were transcribed from the model's reply, not observed from its search tool"
        elif not usable:
            entry["error"] = "no http(s) result with a title"
        else:
            entry["ok"] = True
            cache.put(cache_key(config, query, count), result)
    entry["seconds"] = round(time.monotonic() - started, 3)
    return entry


async def apreflight(
    query: str,
    routes=None,
    cache_dir=None,
    env: dict[str, str] | None = None,
    *,
    count: int = 10,
    backends: dict | None = None,
) -> dict:
    """Call every route live, concurrently, and say which can serve observed hits.

    A route is ok only with observed provenance and at least one http(s) result
    with a title. The cache is never read; a passing route's result is stored,
    so the first real search for the same query and count is a cache hit.
    """
    if not query or not query.strip():
        raise ValueError("empty query")
    names = parse_routes(routes)
    cache = _open_cache(cache_dir, env)
    try:
        chosen = {route: (backends or {}).get(route) or build_backend(route, env) for route in names}
        entries = await asyncio.gather(
            *(_preflight_route(route, chosen[route], query, count, cache) for route in names)
        )
    finally:
        cache.close()
    return {"routes": dict(zip(names, entries))}


def search(query: str, count: int = 10, routes=None, cache_dir=None, env: dict[str, str] | None = None) -> dict:
    """Synchronous `asearch`."""
    return asyncio.run(asearch(query, count, routes, cache_dir, env))


def preflight(query: str, routes=None, cache_dir=None, env: dict[str, str] | None = None, *, count: int = 10) -> dict:
    """Synchronous `apreflight`."""
    return asyncio.run(apreflight(query, routes, cache_dir, env, count=count))


def _render_search(out: dict) -> str:
    lines = [
        f"route={out['route']} cache={out['cache']} degraded={str(out['degraded']).lower()} "
        f"observed={str(out['observed']).lower()} seconds={out['seconds']}"
    ]
    lines += [f"error {route}: {message}" for route, message in out["errors"].items()]
    lines += [f"{h['rank']}. {h['title']} <{h['url']}>\n   {h['snippet']}" for h in out["hits"]]
    return "\n".join(lines)


def _render_preflight(out: dict) -> str:
    return "\n".join(
        f"{route}: {'ok' if e['ok'] else 'FAIL'} results={e['results']} observed={str(e['observed']).lower()} "
        f"seconds={e['seconds']}{' error=' + e['error'] if e['error'] else ''}"
        for route, e in out["routes"].items()
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m sieve.search_chain", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("search", "preflight"):
        cmd = sub.add_parser(name)
        cmd.add_argument("query", nargs="+")
        cmd.add_argument("--routes", default=",".join(DEFAULT_ROUTES))
        cmd.add_argument("--count", type=int, default=10)
        cmd.add_argument("--cache-dir")
        cmd.add_argument("--json", action="store_true", help="print JSON instead of text")
    args = parser.parse_args(argv)
    query = " ".join(args.query)
    try:
        if args.command == "search":
            out = search(query, args.count, args.routes, args.cache_dir)
            ok = bool(out["hits"])
            text = _render_search(out)
        else:
            out = preflight(query, args.routes, args.cache_dir, count=args.count)
            ok = any(e["ok"] for e in out["routes"].values())
            text = _render_preflight(out)
    except ValueError as e:
        parser.error(str(e))
    print(json.dumps(out, ensure_ascii=False) if args.json else text)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
