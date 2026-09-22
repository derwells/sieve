"""Offline routing checks."""

import pytest

from sieve.cache import AnswerCache
from sieve.errors import InvalidAnswerError
from sieve.route import INSTRUCTIONS, jev_route

from .conftest import FakeChoiceAnswer, FakeClient, FakeResponse, FakeUsage


def route(route_id):
    return {"id": route_id, "description": f"Handles {route_id}", "aliases": [f"{route_id} alias"]}


def distribution(state, index):
    ids = [r["id"] for r in state["routes"]]
    pick = "billing" if "billing" in ids else ids[-1]
    return {key: (1.0 if key == pick else 0.0) for key in [*ids, "none"]}


async def test_single_stage_state_options_and_sdk_confidence(tmp_path):
    client = FakeClient(scorer=distribution)
    routes = [route("billing"), route("docs")]
    out = await jev_route("Where is my invoice?", routes, client=client)
    assert out["choice"] == "billing"
    assert out["confidence"] == 1.0 and out["confidence_source"] == "sdk"
    assert out["probabilities"] == {"billing": 1.0, "docs": 0.0, "none": 0.0}
    assert out["usage"]["stages"] == 1 and out["usage"]["batches"] == 1
    call = client.calls[0]
    assert call.state == {"ask": "Where is my invoice?", "routes": routes}
    assert call.questions["q0"].instructions == INSTRUCTIONS
    assert call.questions["q0"].criteria == {
        "billing": "Handles billing  Also known as: billing alias",
        "docs": "Handles docs  Also known as: docs alias",
        "none": "no listed route fits",
    }


async def test_none_choice_and_missing_confidence_fallback():
    class NoConfidenceClient(FakeClient):
        async def system_one(self, state, questions, **kwargs):
            from types import SimpleNamespace
            return FakeResponse({"q0": SimpleNamespace(type="choice", choice="none", probabilities={"billing": 0.1, "none": 0.9})}, FakeUsage(10, 1))

    out = await jev_route("What is the weather?", [route("billing")], client=NoConfidenceClient())
    assert out["choice"] == "none"
    assert out["confidence"] == 0.9 and out["confidence_source"] == "max_probability"


async def test_hierarchical_25_routes():
    routes = [route(f"route{i}") for i in range(25)]
    def graded(state, index):
        ids = [r["id"] for r in state["routes"]]
        return {key: (0.8 if key == ids[-1] else 0.2 if key == ids[-2] else 0.0)
                for key in [*ids, "none"]}

    client = FakeClient(scorer=graded)
    out = await jev_route("A request", routes, client=client)
    assert out["usage"]["stages"] == 2 and out["usage"]["batches"] == 3
    assert out["usage"]["requests"] == 4
    assert len(client.calls) == 4
    assert sorted(len(call.state["routes"]) for call in client.calls) == [5, 6, 10, 10]
    assert set(out["probabilities"]) == {"route8", "route9", "route18", "route19", "route23", "route24", "none"}
    assert client.max_concurrent >= 2


@pytest.mark.parametrize("routes,match", [
    ([route("a"), route("a")], "duplicate route id"),
    ([route("none")], "reserved"),
])
async def test_invalid_ids(routes, match):
    with pytest.raises(ValueError, match=match):
        await jev_route("ask", routes, client=FakeClient())


async def test_invalid_distribution_is_rejected():
    class BrokenClient(FakeClient):
        async def system_one(self, state, questions, **kwargs):
            return FakeResponse({"q0": FakeChoiceAnswer("a", {"a": 1.0})}, FakeUsage(10, 1))

    with pytest.raises(InvalidAnswerError, match="omit offered options"):
        await jev_route("ask", [route("a")], client=BrokenClient())


async def test_route_cache_keys_include_all_route_details_and_instructions(tmp_path):
    cache = AnswerCache(tmp_path / "answers.sqlite3")
    client = FakeClient(scorer=distribution)
    try:
        original = [route("a")]
        await jev_route("ask", original, client=client, cache=cache)
        repeated = await jev_route("ask", original, client=client, cache=cache)
        assert repeated["usage"]["cache_hits"] == 1
        assert repeated["confidence_source"] == "sdk"
        changed = [route("a") | {"aliases": ["another alias"]}]
        await jev_route("ask", changed, client=client, cache=cache)
        assert len(client.calls) == 2
    finally:
        cache.close()
