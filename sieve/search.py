"""jev_search: code proposes query variants, a backend runs them, Jev reranks.

Jev never writes a query and never writes a result. Variants come from string
rules below, the backend supplies url/title/snippet, and Jev's only job is a
relevance probability per deduped hit against the caller's original query.
"""

from __future__ import annotations

import asyncio
import math
import re
import time
from dataclasses import dataclass, field
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from . import rank as rank_module
from . import search_cache as search_cache_module
from .backends import BackendResult, SearchBackendError, SearchHit, select_backend
from .client import DEFAULT_MODEL

#: How many variants a call may propose, whatever the caller asks for.
MIN_VARIANTS = 2
MAX_VARIANTS = 4
DEFAULT_VARIANTS = 3
#: Hits requested from the backend per variant, at least.
HITS_PER_VARIANT = 10
#: Deduped candidates Jev reranks by default, and at most. The pool is a
#: target: a sparse query returns fewer, and nothing pads it.
DEFAULT_DEPTH = 30
MAX_DEPTH = 50

#: Words that carry no retrieval signal and only crowd the query.
FILLER_WORDS = frozenset(
    """
    a an the how do does did i me my we you your please can could would should
    what whats which is are was were be to of for in on at with and or that this
    it its there any some best way ways guide tell show find
    """.split()
)

#: Tokens that all but guarantee the query is about software.
SOFTWARE_WORDS = frozenset(
    """
    sdk api cli library lib framework package module plugin repo repository
    docs documentation github npm pypi pip cargo gem crate compiler runtime
    endpoint server client daemon binary
    """.split()
)

#: Package-ish shapes: dotted or dashed identifiers, file extensions, versions.
PACKAGEISH = re.compile(
    r"^(?:[a-z0-9]+(?:[-_.][a-z0-9]+)+|@[\w.-]+/[\w.-]+|[\w-]+\.(?:js|ts|py|rs|go|rb|sh|json|toml|yaml|yml))$",
    re.IGNORECASE,
)
#: Mid-sentence capitalised or CamelCase tokens read as product names.
CAMEL_CASE = re.compile(r"^[A-Z][a-z0-9]*[A-Z]\w*$")

#: Query params that identify a campaign or a referrer, never a document.
TRACKING_PARAMS = frozenset({"fbclid", "gclid", "gclsrc", "msclkid", "mc_cid", "mc_eid", "ref", "ref_src", "referrer"})
TRACKING_PREFIXES = ("utm_",)

#: Ports that are implied by the scheme.
DEFAULT_PORTS = {"http": "80", "https": "443"}


def _tokens(query: str) -> list[str]:
    return query.split()


def strip_filler(query: str) -> str:
    """The query with filler words removed. Falls back to the query if emptied."""
    kept = [t for t in _tokens(query) if t.strip("\"'?.,!").lower() not in FILLER_WORDS]
    return " ".join(kept) if kept else query.strip()


def names_software(query: str) -> bool:
    """Whether the query looks like it names a piece of software."""
    tokens = _tokens(query)
    for position, raw in enumerate(tokens):
        token = raw.strip("\"'?.,!()")
        if not token:
            continue
        lowered = token.lower()
        if lowered in SOFTWARE_WORDS:
            return True
        if PACKAGEISH.match(token):
            return True
        if CAMEL_CASE.match(token):
            return True
        if token.isupper() and len(token) >= 2 and token.isalpha():
            return True
        if position > 0 and token[0].isupper() and token[1:].islower() and len(token) > 2:
            return True
    return False


def rephrase(query: str) -> str:
    """A site-agnostic rewrite: drop quotes, then rotate the last term to the front."""
    unquoted = query.replace('"', " ").replace("'", " ")
    tokens = [t for t in _tokens(strip_filler(unquoted)) if t]
    if len(tokens) < 3:
        return " ".join(tokens)
    return " ".join(tokens[-1:] + tokens[:-1])


def propose_variants(query: str, limit: int = DEFAULT_VARIANTS) -> list[str]:
    """2-4 deterministic variants of `query`, original first, duplicates dropped."""
    limit = max(MIN_VARIANTS, min(int(limit), MAX_VARIANTS))
    original = query.strip()
    candidates = [original, strip_filler(original), rephrase(original)]
    if names_software(original):
        stripped = strip_filler(original)
        candidates.extend([f"{stripped} docs", f"{stripped} github"])

    variants: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        candidate = " ".join(candidate.split())
        key = candidate.lower()
        if not candidate or key in seen:
            continue
        seen.add(key)
        variants.append(candidate)
        if len(variants) == limit:
            break
    return variants


def canonical_url(url: str) -> str:
    """A comparison key: lowercase host, no www, no default port, no fragment,
    no tracking params, no trailing slash."""
    raw = url.strip()
    if not raw:
        return ""
    parts = urlsplit(raw)
    scheme = parts.scheme.lower() or "https"
    host = (parts.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    netloc = host
    if parts.port and str(parts.port) != DEFAULT_PORTS.get(scheme):
        netloc = f"{host}:{parts.port}"
    query = [
        (key, value)
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
        if key.lower() not in TRACKING_PARAMS and not key.lower().startswith(TRACKING_PREFIXES)
    ]
    path = parts.path.rstrip("/")
    return urlunsplit((scheme, netloc, path, urlencode(query), ""))


@dataclass
class DedupedHit:
    """One hit after merging every variant that found the same page."""

    key: str
    url: str
    title: str
    snippet: str
    engines: list[str] = field(default_factory=list)
    published: str = ""


def dedupe_hits(hits) -> list[DedupedHit]:
    """First-seen url/title wins; a later non-empty snippet fills an empty one.

    Engines that found the same page are merged as provenance.
    """
    merged: dict[str, DedupedHit] = {}
    for hit in hits:
        key = canonical_url(hit.url)
        if not key:
            continue
        engines = list(getattr(hit, "engines", ()) or ())
        existing = merged.get(key)
        if existing is None:
            merged[key] = DedupedHit(
                key=key,
                url=hit.url.strip(),
                title=(hit.title or "").strip(),
                snippet=(hit.snippet or "").strip(),
                engines=engines,
                published=getattr(hit, "published", "") or "",
            )
            continue
        if not existing.title and hit.title:
            existing.title = hit.title.strip()
        if not existing.snippet and hit.snippet:
            existing.snippet = hit.snippet.strip()
        if not existing.published and getattr(hit, "published", ""):
            existing.published = hit.published
        existing.engines.extend(e for e in engines if e not in existing.engines)
    return list(merged.values())


def interleave(results: list[BackendResult]) -> list[SearchHit]:
    """Every variant's hits, rank by rank: all first hits, then all second hits.

    Trimming this list keeps each variant's best hits rather than all of the
    first variant's and none of the last's.
    """
    ordered: list[SearchHit] = []
    longest = max((len(r.hits) for r in results), default=0)
    for rank in range(longest):
        for result in results:
            if rank < len(result.hits):
                ordered.append(result.hits[rank])
    return ordered


def candidate_text(hit: DedupedHit) -> str:
    """What Jev is shown for one hit."""
    text = f"{hit.title} — {hit.url}"
    if hit.snippet:
        text = f"{text} — {hit.snippet}"
    return text


def resolve_depth(depth: int | None, top_k: int) -> int:
    """The candidate pool size: `depth`, never below `top_k`, never above MAX_DEPTH."""
    wanted = DEFAULT_DEPTH if depth is None else int(depth)
    return max(1, min(max(wanted, int(top_k)), MAX_DEPTH))


def variant_counts(backend, depth: int, variant_count: int) -> list[int]:
    """Hits to ask each variant for.

    Each gets an even share of the pool, at least HITS_PER_VARIANT. A backend
    that pages (`paginates = True`) asks the original query for the whole pool
    instead, because variants overlap heavily and a further page of the original
    is cheaper than more variants.
    """
    share = max(HITS_PER_VARIANT, math.ceil(depth / max(1, variant_count)))
    counts = [share] * variant_count
    if counts and getattr(backend, "paginates", False):
        counts[0] = max(share, depth)
    return counts


async def _run_variant(backend, query: str, count: int, search_cache) -> tuple[BackendResult | SearchBackendError, bool]:
    key = search_cache_module.result_key(backend, query, count) if search_cache is not None else None
    if key is not None:
        cached = search_cache.get(key)
        if cached is not None:
            return cached, True
    try:
        result = await backend.search(query, count)
    except SearchBackendError as e:
        return e, False
    except Exception as e:  # a backend blowing up must not lose the other variants
        return SearchBackendError(f"{type(e).__name__}: {e}"), False
    if key is not None:
        search_cache.put(key, result)
    return result, False


def _row(hit: DedupedHit, probability: float) -> dict:
    row = {"url": hit.url, "title": hit.title, "snippet": hit.snippet, "probability": probability}
    if hit.engines:
        row["engines"] = hit.engines
    if hit.published:
        row["published"] = hit.published
    return row


async def jev_search(
    query: str,
    top_k: int = 10,
    variants: int = DEFAULT_VARIANTS,
    depth: int | None = None,
    *,
    backend=None,
    backend_name: str | None = None,
    hits_per_variant: int | None = None,
    client=None,
    cache=None,
    search_cache=None,
    model: str = DEFAULT_MODEL,
    threshold: float = 0.0,
) -> dict:
    """Search the web through `backend`, dedupe, rerank with Jev, return the top k.

    `depth` is the candidate pool: how many deduped hits, at most, Jev reranks.
    `top_k` is how many of those come back. The pool is filled from the variants
    the query already has; it is never padded and no variants are added to fill it.
    """
    started = time.monotonic()
    variant_queries = propose_variants(query, variants)
    backend = backend or select_backend(backend_name)
    pool_target = resolve_depth(depth, top_k)
    counts = (
        [hits_per_variant] * len(variant_queries)
        if hits_per_variant
        else variant_counts(backend, pool_target, len(variant_queries))
    )

    outcomes = await asyncio.gather(
        *(_run_variant(backend, v, n, search_cache) for v, n in zip(variant_queries, counts))
    )
    calls = [(variant, o, cached) for variant, (o, cached) in zip(variant_queries, outcomes)]
    results = [(o, cached) for _, o, cached in calls if isinstance(o, BackendResult)]
    errors = [{"query": variant, "error": str(o)} for variant, o, _ in calls if isinstance(o, SearchBackendError)]
    if not results:
        detail = "; ".join(e["error"] for e in errors) or "no results"
        raise SearchBackendError(f"every {backend.name} search variant failed: {detail}")

    hits = interleave([r for r, _ in results])
    deduped = dedupe_hits(hits)
    pool = deduped[:pool_target]

    ranked = await rank_module.jev_rank(
        question=query,
        candidates=[{"id": hit.key, "text": candidate_text(hit)} for hit in pool],
        top_k=top_k,
        threshold=threshold,
        client=client,
        cache=cache,
        model=model,
    )
    by_key = {hit.key: hit for hit in pool}
    rows = [_row(by_key[row["id"]], row["probability"]) for row in ranked["results"] if row["id"] in by_key]

    usage = {k: v for k, v in ranked.items() if k not in {"results", "candidates_scored"}}
    backend_calls = []
    for result, cached in results:
        call = {"query": result.query, "hits": len(result.hits), "wall_seconds": result.wall_seconds, "usage": result.usage}
        if cached:
            call["cached"] = True
        backend_calls.append(call)
    payload = {
        "results": rows,
        "variants": variant_queries,
        "backend": backend.name,
        "usage": usage,
        "depth": pool_target,
        "hits_requested": counts,
        "hits_found": len(hits),
        "hits_deduped": len(deduped),
        "candidates_scored": len(pool),
        "backend_requests": sum(int(r.usage.get("requests", 1)) for r, cached in results if not cached),
        "backend_calls": backend_calls,
        "wall_seconds": round(time.monotonic() - started, 3),
    }
    if len(pool) < pool_target:
        payload["pool_short"] = True
    if errors:
        payload["backend_errors"] = errors
    return payload
