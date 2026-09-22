"""jev_rank: 40 candidates per call, 2000-char truncation, deterministic sort."""

from sieve.rank import CANDIDATES_PER_BATCH, MAX_CANDIDATE_CHARS, jev_rank

from .conftest import FakeClient


def by_id(scores: dict[str, float], default: float = 0.1):
    def score(state, index):
        return scores.get(state["candidates"][index]["id"], default)

    return score


async def test_ranks_by_probability_descending():
    candidates = [{"id": "a", "text": "alpha"}, {"id": "b", "text": "beta"}, {"id": "c", "text": "gamma"}]
    client = FakeClient(scorer=by_id({"a": 0.2, "b": 0.95, "c": 0.5}))
    out = await jev_rank("q", candidates, client=client)
    assert out["results"] == [
        {"id": "b", "probability": 0.95},
        {"id": "c", "probability": 0.5},
        {"id": "a", "probability": 0.2},
    ]
    assert out["candidates_scored"] == 3


async def test_forty_candidates_per_request():
    candidates = [{"id": str(i), "text": f"item {i}"} for i in range(95)]
    client = FakeClient()
    out = await jev_rank("q", candidates, client=client)
    assert [len(call.state["candidates"]) for call in client.calls] == [40, 40, 15]
    assert len(out["results"]) == 95
    assert CANDIDATES_PER_BATCH == 40


async def test_long_candidates_are_truncated_before_sending():
    candidates = [{"id": "big", "text": "x" * 9000}]
    client = FakeClient()
    await jev_rank("q", candidates, client=client)
    assert len(client.calls[0].state["candidates"][0]["text"]) == MAX_CANDIDATE_CHARS


async def test_top_k_and_threshold_filter():
    candidates = [{"id": c, "text": c} for c in "abcd"]
    client = FakeClient(scorer=by_id({"a": 0.9, "b": 0.7, "c": 0.3, "d": 0.1}))
    out = await jev_rank("q", candidates, top_k=2, threshold=0.5, client=client)
    assert [r["id"] for r in out["results"]] == ["a", "b"]


async def test_missing_ids_fall_back_to_position():
    client = FakeClient(scorer=lambda state, i: 0.5)
    out = await jev_rank("q", [{"text": "one"}, {"text": "two"}], client=client)
    assert {r["id"] for r in out["results"]} == {"0", "1"}


async def test_ties_break_on_id_so_the_order_is_stable():
    candidates = [{"id": "z", "text": "z"}, {"id": "a", "text": "a"}]
    client = FakeClient(scorer=lambda state, i: 0.5)
    out = await jev_rank("q", candidates, client=client)
    assert [r["id"] for r in out["results"]] == ["a", "z"]


async def test_batches_run_concurrently():
    candidates = [{"id": str(i), "text": f"item {i}"} for i in range(200)]
    client = FakeClient()
    await jev_rank("q", candidates, client=client, concurrency=5)
    assert len(client.calls) == 5
    assert client.max_concurrent > 1
