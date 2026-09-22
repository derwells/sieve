"""The budget stops the run and returns partial results rather than overspending."""

import json

import pytest

from sieve.jev import COST_PER_MILLION_INPUT_TOKENS, JevScorer, QuestionSpec, ScoreItem, cost_for_input_tokens

from .conftest import FakeClient

SPEC = QuestionSpec(
    state_field="units",
    instructions="Is {ref} relevant to `question`?",
    criteria_true="yes",
    criteria_false="no",
)


def item(index: int, chars: int = 140_000) -> ScoreItem:
    """Big enough that each item is its own request."""
    payload = {"path": f"f{index}.py", "content": "x" * chars}
    return ScoreItem(id=str(index), text=json.dumps(payload, sort_keys=True), payload=payload)


def test_price_is_the_published_rate():
    assert COST_PER_MILLION_INPUT_TOKENS == 0.042
    assert cost_for_input_tokens(1_000_000) == pytest.approx(0.042)


async def test_run_stops_and_flags_when_the_budget_would_be_exceeded():
    client = FakeClient(input_tokens=1_000_000)  # $0.042 of real spend per request
    scorer = JevScorer(client, concurrency=1, budget_usd=0.10)
    items = [item(i) for i in range(10)]
    run = await scorer.score("q", items, SPEC)
    assert run.budget_exhausted is True
    assert len(client.calls) < len(items)
    assert scorer.spent_usd <= 0.10
    assert run.scores, "a budget stop still returns what was already scored"
    assert len(run.scores) == len(client.calls)


async def test_a_sufficient_budget_scores_everything():
    client = FakeClient(input_tokens=1000)
    scorer = JevScorer(client, concurrency=4, budget_usd=1.0)
    items = [item(i) for i in range(6)]
    run = await scorer.score("q", items, SPEC)
    assert run.budget_exhausted is False
    assert len(run.scores) == 6


async def test_no_budget_means_no_stop():
    client = FakeClient(input_tokens=10_000_000)
    scorer = JevScorer(client, concurrency=2, budget_usd=None)
    run = await scorer.score("q", [item(i) for i in range(4)], SPEC)
    assert run.budget_exhausted is False
    assert len(run.scores) == 4


async def test_a_budget_too_small_for_one_request_sends_nothing():
    client = FakeClient(input_tokens=1_000_000)
    scorer = JevScorer(client, concurrency=4, budget_usd=0.0000001)
    run = await scorer.score("q", [item(0)], SPEC)
    assert client.calls == []
    assert run.budget_exhausted is True
    assert run.scores == {}


async def test_spend_carries_across_passes_on_one_scorer():
    client = FakeClient(input_tokens=1_000_000)
    scorer = JevScorer(client, concurrency=1, budget_usd=0.05)
    first = await scorer.score("q", [item(0)], SPEC)
    assert first.budget_exhausted is False
    second = await scorer.score("q", [item(1)], SPEC)
    assert second.budget_exhausted is True
    assert len(client.calls) == 1


async def test_usage_reports_tokens_and_cost():
    client = FakeClient(input_tokens=500_000, output_tokens=100)
    scorer = JevScorer(client, concurrency=1, budget_usd=1.0)
    run = await scorer.score("q", [item(0), item(1)], SPEC)
    assert run.usage.requests == 2
    assert run.usage.input_tokens == 1_000_000
    assert run.usage.tokens == 1_000_200
    assert run.usage.as_dict()["cost_usd"] == pytest.approx(0.042)
