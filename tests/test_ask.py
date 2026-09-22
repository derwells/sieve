"""sieve.ask: ad hoc judge/choose over caller-held items."""

import json

from sieve.ask import ITEMS_PER_BATCH, MAX_ITEM_CHARS, choose, judge, main

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
