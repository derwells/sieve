"""Live checks against the real Jev API. Skipped without a key.

Run with:  export TYPESAFE_API_KEY=...; uv run pytest -m live
"""

import os
import shutil
import time
from pathlib import Path

import pytest

from sieve.cache import NullCache
from sieve.client import API_KEY_ENV
from sieve.grep import jev_grep
from sieve.rank import jev_rank

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not os.environ.get(API_KEY_ENV), reason=f"{API_KEY_ENV} is not set"),
]

#: sieve greps its own checkout, so the test needs no repository but this one.
REPO = str(Path(__file__).resolve().parent.parent)


async def test_grep_finds_the_validation_module():
    started = time.monotonic()
    out = await jev_grep(
        question="where are Noul answers validated client-side",
        path=REPO,
        mode="files",
        top_k=10,
        cache=NullCache(),
    )
    elapsed = time.monotonic() - started
    top5 = [row["path"] for row in out["results"][:5]]
    print(
        f"\nself files pass: {out['units_scored']} units, {out['requests']} requests, "
        f"{out['tokens']} tokens, ${out['cost_usd']:.4f}, {elapsed:.1f}s"
    )
    print("top 5:", top5)
    assert "sieve/validate.py" in top5


async def test_rank_puts_the_relevant_candidate_first():
    candidates = [
        {"id": "sourdough", "text": "Feed the starter twice a day at room temperature until it doubles."},
        {"id": "tyres", "text": "Rotate your tyres every 10,000 km to even out tread wear."},
        {"id": "postgres", "text": "VACUUM reclaims storage occupied by dead tuples in a PostgreSQL table."},
        {"id": "kayak", "text": "A high-angle paddling stroke gives more acceleration in whitewater."},
        {"id": "tulips", "text": "Plant tulip bulbs in autumn, about three times as deep as the bulb is tall."},
    ]
    out = await jev_rank(
        question="how do I stop a Postgres table from bloating?",
        candidates=candidates,
    )
    print("\nrank:", out["results"], f"${out['cost_usd']:.5f}")
    assert out["results"][0]["id"] == "postgres"


SEARCH_QUERY = "TypeSafe Jev rerank cookbook"


def _report(name, out):
    print(f"\n{name}: {out['hits_found']} hits, {out['hits_deduped']} deduped, {out['wall_seconds']}s total")
    print("  variants:", out["variants"])
    for call in out["backend_calls"]:
        print(f"    {call['wall_seconds']}s {call['hits']} hits  {call['query']!r}  {call['usage']}")
    for row in out["results"][:3]:
        print(f"  {row['probability']}  {row['url']}")
    print(f"  jev: {out['usage']['requests']} requests, ${out['usage']['cost_usd']:.5f}")


@pytest.mark.skipif(not os.environ.get("BRAVE_API_KEY"), reason="BRAVE_API_KEY is not set")
async def test_search_live_brave():
    from sieve.search import jev_search

    out = await jev_search(SEARCH_QUERY, top_k=10, backend_name="brave", cache=NullCache())
    _report("brave", out)
    assert out["backend"] == "brave"
    assert out["results"]
    assert all(row["url"].startswith("http") for row in out["results"])


@pytest.mark.skipif(not shutil.which("claude"), reason="claude is not on PATH")
async def test_search_live_claude():
    from sieve.search import jev_search

    out = await jev_search(SEARCH_QUERY, top_k=10, backend_name="claude", cache=NullCache())
    _report("claude", out)
    assert out["backend"] == "claude"
    assert out["results"]


@pytest.mark.skipif(not shutil.which("codex"), reason="codex is not on PATH")
async def test_search_live_codex():
    from sieve.search import jev_search

    out = await jev_search(SEARCH_QUERY, top_k=10, backend_name="codex", cache=NullCache())
    _report("codex", out)
    assert out["backend"] == "codex"
    assert out["results"]
