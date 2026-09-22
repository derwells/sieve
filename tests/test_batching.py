"""Batching packs units under the token limit and dispatches them concurrently."""

import json

import pytest

from sieve.jev import (
    BATCH_TOKEN_LIMIT,
    STATE_TOKEN_LIMIT,
    JevScorer,
    QuestionSpec,
    ScoreItem,
    estimate_tokens,
    plan_batches,
    state_overhead_tokens,
)

from .conftest import FakeClient

SPEC = QuestionSpec(
    state_field="units",
    instructions="Is {ref} relevant to `question`?",
    criteria_true="It is relevant.",
    criteria_false="It is not.",
)


def item(index: int, chars: int) -> ScoreItem:
    payload = {"path": f"f{index}.py", "content": "x" * chars}
    return ScoreItem(id=str(index), text=json.dumps(payload, sort_keys=True), payload=payload)


def batch_context_tokens(question: str, batch: list[ScoreItem]) -> int:
    """What the request has to fit: the whole state, plus its longest question."""
    state = {"question": question, "units": [i.payload for i in batch]}
    return estimate_tokens(json.dumps(state)) + SPEC.question_tokens()


def test_every_batch_stays_under_the_state_token_limit():
    question = "how does the server route a request to a handler"
    items = [item(i, 4000) for i in range(40)]
    batches = plan_batches(question, items, SPEC)
    assert len(batches) > 1
    assert sum(len(b) for b in batches) == len(items)
    for batch in batches:
        assert batch_context_tokens(question, batch) < STATE_TOKEN_LIMIT


def test_batches_are_packed_not_one_per_item():
    question = "q"
    items = [item(i, 400) for i in range(50)]
    batches = plan_batches(question, items, SPEC)
    assert len(batches) == 1


def test_a_unit_larger_than_the_limit_still_gets_its_own_batch():
    question = "q"
    items = [item(0, 200), item(1, BATCH_TOKEN_LIMIT * 8), item(2, 200)]
    batches = plan_batches(question, items, SPEC)
    assert [len(b) for b in batches] == [1, 1, 1]
    assert batches[1][0].id == "1"


def test_max_items_caps_batch_size_below_the_token_limit():
    items = [item(i, 10) for i in range(95)]
    batches = plan_batches("q", items, SPEC, max_items=40)
    assert [len(b) for b in batches] == [40, 40, 15]


def test_state_overhead_grows_with_the_question():
    assert state_overhead_tokens("x" * 400, "units") > state_overhead_tokens("x", "units")


async def test_scorer_sends_one_noul_per_unit_over_a_shared_state():
    client = FakeClient(scorer=lambda state, i: 0.1 * i)
    scorer = JevScorer(client, concurrency=4)
    items = [item(i, 50) for i in range(5)]
    run = await scorer.score("q", items, SPEC)
    assert len(client.calls) == 1
    call = client.calls[0]
    assert call.state["question"] == "q"
    assert len(call.state["units"]) == 5
    assert set(call.questions) == {"q0", "q1", "q2", "q3", "q4"}
    assert call.questions["q3"].instructions == "Is `units[3]` relevant to `question`?"
    assert run.scores["4"] == pytest.approx(0.4)


async def test_concurrency_is_capped_by_the_semaphore():
    client = FakeClient()
    scorer = JevScorer(client, concurrency=3)
    items = [item(i, 140_000) for i in range(12)]  # each exceeds the limit alone
    run = await scorer.score("q", items, SPEC)
    assert len(client.calls) == 12
    assert client.max_concurrent <= 3
    assert len(run.scores) == 12
