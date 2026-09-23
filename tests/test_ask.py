"""sieve.ask: ad hoc judge/choose/score over caller-held items."""

import json

import pytest

from sieve.ask import ITEMS_PER_BATCH, MAX_ITEM_CHARS, ask, choose, judge, main, score
from sieve.cache import AnswerCache
from sieve.errors import SieveError

from .conftest import FakeClient


def by_id(scores, default=0.1):
    def score(state, index):
        return scores.get(state["items"][index]["id"], default)
    return score


async def test_judge_sorts_and_thresholds():
    items = [{"id": "a", "text": "alpha"}, {"id": "b", "text": "beta"}, "gamma"]
    client = FakeClient(scorer=by_id({"a": 0.2, "b": 0.95, "2": 0.6}))
    out = await judge(items, "q", yes="y", no="n", threshold=0.5, client=client)
    assert out["results"] == [{"id": "b", "probability": 0.95}, {"id": "2", "probability": 0.6}]
    assert out["items_scored"] == 3
    call = client.calls[0]
    q = call.questions["q0"]
    assert q.criteria["true"] == "y" and q.criteria["false"] == "n"
    assert call.state["question"] == "q"


async def test_judge_batches_and_truncates():
    items = [{"id": str(i), "text": f"item {i}"} for i in range(45)]
    client = FakeClient()
    await judge(items, "q", yes="y", no="n", client=client)
    assert [len(c.state["items"]) for c in client.calls] == [ITEMS_PER_BATCH, 5]
    client = FakeClient()
    await judge([{"id": "big", "text": "x" * 9000}], "q", yes="y", no="n", client=client)
    assert len(client.calls[0].state["items"][0]["text"]) == MAX_ITEM_CHARS


async def test_choose_returns_choice_and_distribution():
    def answer(state, index):
        text = state["items"][index]["text"]
        return {"bug": 0.8, "feature": 0.2} if "crash" in text else {"bug": 0.1, "feature": 0.9}

    client = FakeClient(scorer=answer)
    out = await choose(["app crashes on start", "add dark mode"], "kind?", {"bug": "b", "feature": "f"}, client=client)
    assert [r["choice"] for r in out["results"]] == ["bug", "feature"]
    assert out["results"][0]["probabilities"] == {"bug": 0.8, "feature": 0.2}
    assert client.calls[0].questions["q0"].criteria == {"bug": "b", "feature": "f"}


def test_cli_rejects_bad_option(capsys, monkeypatch):
    monkeypatch.setattr("sys.stdin", __import__("io").StringIO(json.dumps(["a"])))
    assert main(["choose", "q", "--option", "nodesc"]) == 2
    assert "LABEL=DESCRIPTION" in capsys.readouterr().err


async def test_score_returns_level_expected_and_distribution():
    def answer(state, index):
        return {0: 0.1, 1: 0.2, 2: 0.7} if "crash" in state["items"][index]["text"] else {0: 0.8, 1: 0.1, 2: 0.1}

    client = FakeClient(scorer=answer)
    out = await score(
        ["app crashes on start", "typo in the changelog"],
        "how urgent is the item?",
        ["not urgent", "worth doing", "drop everything"],
        client=client,
    )
    assert [r["level"] for r in out["results"]] == [2, 0]
    assert out["results"][0]["label"] == "drop everything"
    assert out["results"][0]["probabilities"] == {"0": 0.1, "1": 0.2, "2": 0.7}
    assert out["results"][0]["expected"] == 1.6
    assert out["levels"] == ["not urgent", "worth doing", "drop everything"]
    assert list(client.calls[0].questions["q0"].criteria) == out["levels"]


async def test_score_rejects_a_single_level():
    with pytest.raises(SieveError, match="two levels"):
        await score(["a"], "q", ["only one"], client=FakeClient())


async def test_ask_dispatches_on_kind_and_echoes_it():
    for kind, extra in (
        ("judge", {"yes": "y", "no": "n"}),
        ("choose", {"options": {"a": "first", "b": "second"}}),
        ("score", {"levels": ["low", "high"]}),
    ):
        client = FakeClient(scorer=lambda state, index: {"a": 0.6, "b": 0.4} if kind == "choose"
                            else ({0: 0.3, 1: 0.7} if kind == "score" else 0.9))
        out = await ask("q", ["one item"], kind, client=client, **extra)
        assert out["kind"] == kind
        assert out["items_scored"] == 1
        assert out["cost_usd"] > 0


@pytest.mark.parametrize(
    "kind, extra, message",
    [
        ("judge", {}, "yes and no"),
        ("choose", {}, "options"),
        ("score", {}, "levels"),
        ("guess", {}, "kind must be one of"),
    ],
)
async def test_ask_requires_the_answer_space_for_its_kind(kind, extra, message):
    with pytest.raises(SieveError, match=message):
        await ask("q", ["one item"], kind, client=FakeClient(), **extra)


async def test_ask_rejects_an_empty_item_list():
    with pytest.raises(SieveError, match="nothing to ask about"):
        await ask("q", [], "judge", yes="y", no="n", client=FakeClient())


async def test_score_answers_are_cached_with_integer_levels(tmp_path):
    cache = AnswerCache(tmp_path / "answers.sqlite3")
    client = FakeClient(scorer=lambda state, index: {0: 0.25, 1: 0.75})
    first = await score(["an item"], "q", ["low", "high"], client=client, cache=cache)
    second = await score(["an item"], "q", ["low", "high"], client=client, cache=cache)
    assert len(client.calls) == 1
    assert second["cache_hits"] == 1 and second["requests"] == 0
    assert second["results"] == first["results"]
    cache.close()


def test_cli_score_needs_two_levels(capsys, monkeypatch):
    monkeypatch.setattr("sys.stdin", __import__("io").StringIO(json.dumps(["a"])))
    assert main(["score", "q", "--level", "only one", "--no-cache"]) == 2
    assert "two levels" in capsys.readouterr().err
