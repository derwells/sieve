"""Live Noul judgments on small synthetic conversations."""

import os

import pytest

from sieve.cache import NullCache
from sieve.client import API_KEY_ENV
from sieve.triage import jev_triage_threads

pytestmark = [pytest.mark.live, pytest.mark.skipif(not os.environ.get(API_KEY_ENV), reason=f"{API_KEY_ENV} is not set")]


def event(i, role, text):
    return {"id": str(i), "ts": f"2026-01-01T00:00:0{i}Z", "role": role, "kind": "text", "text": text}


async def test_answered_and_unanswered_requests_live():
    contract = {"title": "Database choice", "first_prompt": "Set up a database", "human_amendments": []}
    threads = [
        {"thread_id": "answered", "contract": contract, "events": [event(1, "human", "Set up a database"), event(2, "assistant", "Should I use Postgres or SQLite?"), event(3, "human", "Postgres")]},
        {"thread_id": "waiting", "contract": contract, "events": [event(1, "assistant", "Should I use Postgres or SQLite?"), event(2, "tool", "Build output."), event(3, "human", "The weather is sunny today."), event(4, "assistant", "I cannot continue the database setup until you answer whether to use Postgres or SQLite.")]},
    ]
    out = await jev_triage_threads(threads, client=None, cache=NullCache())
    rows = {r["thread_id"]: r for r in out["threads"]}
    answered = rows["answered"]["requests"][0]
    waiting = rows["waiting"]["requests"][0]
    print(f"\nanswered={answered['probabilities']} waiting={waiting['probabilities']} buckets={[r['bucket'] for r in out['threads']]} usage={out['usage']}")
    assert answered["probabilities"]["answered"] >= 0.5
    assert waiting["probabilities"]["answered"] < 0.5
    assert waiting["probabilities"]["blocking"] >= 0.5
