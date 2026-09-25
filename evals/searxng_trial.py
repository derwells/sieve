"""Bounded live trial of a SearXNG instance as the jev_search backend.

    uv run python evals/searxng_trial.py [--pages 3] [--jev]

For each query: fetch pages 1..N of the JSON API (N capped at 5), and record
per page the latency, cumulative unique URLs, how many carry a usable snippet
(40+ characters), and the engines SearXNG reported as unresponsive. With
`--jev`, also run the full jev_search pipeline once per query at depth 50
(this spends Jev tokens, not backend credit). Raw output goes to evals/raw/.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import time

import httpx2

from sieve.backends.searxng import DEFAULT_URL, SearxngBackend, parse_searxng_payload, unresponsive_engines
from sieve.search import canonical_url, jev_search

QUERIES = {
    "exact_docs": "python asyncio TaskGroup documentation",
    "broad_comparison": "PostgreSQL vs MySQL vs SQLite for a small web app",
    "recent": "latest Rust stable release 2026",
    "niche": "tree-sitter query predicates #match? #eq? syntax",
}
USABLE_SNIPPET_CHARS = 40


async def page_trial(backend: SearxngBackend, query: str, pages: int) -> dict:
    seen: dict[str, dict] = {}
    rows = []
    async with httpx2.AsyncClient(timeout=backend.timeout) as client:
        for pageno in range(1, pages + 1):
            started = time.monotonic()
            try:
                payload = await backend._page(client, query, pageno)
                error = None
            except Exception as e:  # recorded, not retried: the trial measures failure
                payload, error = {}, f"{type(e).__name__}: {e}"
            latency = round(time.monotonic() - started, 2)
            hits = parse_searxng_payload(payload)
            new = 0
            for hit in hits:
                key = canonical_url(hit.url)
                if key not in seen:
                    seen[key] = {"url": hit.url, "title": hit.title, "snippet": hit.snippet, "engines": list(hit.engines)}
                    new += 1
            rows.append(
                {
                    "page": pageno,
                    "latency_s": latency,
                    "page_hits": len(hits),
                    "new_unique": new,
                    "cum_unique": len(seen),
                    "cum_usable_snippets": sum(len(h["snippet"]) >= USABLE_SNIPPET_CHARS for h in seen.values()),
                    "unresponsive": unresponsive_engines(payload),
                    "error": error,
                }
            )
            if error or new == 0:
                break
    engines: dict[str, int] = {}
    for hit in seen.values():
        for engine in hit["engines"]:
            engines[engine] = engines.get(engine, 0) + 1
    return {"query": query, "pages": rows, "engines": engines, "hits": list(seen.values())}


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--pages", type=int, default=3)
    parser.add_argument("--jev", action="store_true", help="also run jev_search at depth 50 per query")
    args = parser.parse_args()
    backend = SearxngBackend(args.url)
    pages = max(1, min(args.pages, 5))

    out = {"url": args.url, "pages": pages, "queries": {}}
    for label, query in QUERIES.items():
        trial = await page_trial(backend, query, pages)
        out["queries"][label] = trial
        for row in trial["pages"]:
            failed = ",".join(f"{u['engine']}({u['reason']})" for u in row["unresponsive"]) or "-"
            print(
                f"{label:17} p{row['page']} {row['latency_s']:5.2f}s hits={row['page_hits']:2} "
                f"new={row['new_unique']:2} unique={row['cum_unique']:3} usable={row['cum_usable_snippets']:3} "
                f"failed={failed} {row['error'] or ''}"
            )
        print(f"{'':17} engines: {trial['engines']}")
        if args.jev:
            ranked = await jev_search(query, top_k=10, depth=50, backend=backend)
            out["queries"][label]["jev_search"] = ranked
            print(
                f"{'':17} jev_search depth=50: pool={ranked['candidates_scored']} found={ranked['hits_found']} "
                f"requests={ranked['backend_requests']} wall={ranked['wall_seconds']}s "
                f"jev_cost=${ranked['usage'].get('cost_usd')} errors={len(ranked.get('backend_errors', []))}"
            )
            for row in ranked["results"][:5]:
                print(f"{'':19}{row['probability']:.2f} {row['title'][:70]} <{row['url'][:80]}>")

    raw = pathlib.Path(__file__).parent / "raw"
    raw.mkdir(exist_ok=True)
    path = raw / f"searxng-trial-{time.strftime('%Y%m%dT%H%M%S')}.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"raw: {path}")


if __name__ == "__main__":
    asyncio.run(main())
