"""Shared fakes: a Jev client that answers from a rule, never over the network."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class FakeNoulAnswer:
    noul: float
    type: str = "noul"


@dataclass(frozen=True)
class FakeChoiceAnswer:
    choice: str
    probabilities: dict[str, float]
    confidence: float = 1.0
    type: str = "choice"


@dataclass(frozen=True)
class FakeScoreAnswer:
    score: float
    probabilities: dict[int, float]
    legend: dict[int, str] = field(default_factory=dict)
    confidence: float = 1.0
    type: str = "score"


@dataclass(frozen=True)
class FakeUsage:
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class FakeResponse:
    answers: dict[str, Any]
    usage: FakeUsage
    model: str = "jev-1.13.0"


@dataclass
class RecordedCall:
    state: dict
    questions: dict
    model: str | None


class FakeClient:
    """Answers every Noul with `scorer(state, index)`; records what it was asked."""

    def __init__(self, scorer=None, input_tokens: int = 1000, output_tokens: int = 20) -> None:
        self.scorer = scorer or (lambda state, index: 0.9)
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.calls: list[RecordedCall] = []
        self.concurrent = 0
        self.max_concurrent = 0

    async def system_one(self, state, questions, *, model=None, **kwargs):
        self.concurrent += 1
        self.max_concurrent = max(self.max_concurrent, self.concurrent)
        try:
            await asyncio.sleep(0)
            self.calls.append(RecordedCall(state=state, questions=dict(questions), model=model))
            answers = {
                qid: FakeNoulAnswer(noul=float(self.scorer(state, int(qid[1:]))))
                for qid in questions
            }
            return FakeResponse(
                answers=answers,
                usage=FakeUsage(self.input_tokens, self.output_tokens),
            )
        finally:
            self.concurrent -= 1

    async def aclose(self) -> None:
        return None


class FakeSearchBackend:
    """Returns canned hits per query; records what it was asked, and when."""

    def __init__(self, name: str = "fake", hits_by_query: dict | None = None, default=(), error_on=()) -> None:
        self.name = name
        self.hits_by_query = hits_by_query or {}
        self.default = list(default)
        self.error_on = set(error_on)
        self.queries: list[str] = []
        self.concurrent = 0
        self.max_concurrent = 0

    async def search(self, query, count):
        from sieve.backends import BackendResult, SearchBackendError

        self.concurrent += 1
        self.max_concurrent = max(self.max_concurrent, self.concurrent)
        try:
            await asyncio.sleep(0)
            self.queries.append(query)
            if query in self.error_on:
                raise SearchBackendError(f"no results for {query}")
            hits = list(self.hits_by_query.get(query, self.default))[:count]
            return BackendResult(query=query, hits=hits, usage={"requests": 1}, wall_seconds=0.01)
        finally:
            self.concurrent -= 1
