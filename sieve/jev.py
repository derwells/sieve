"""Batch, cache, budget, and validate Noul, Choice, or Score scoring.

One request carries one state and one question per item in it. Questions over a
shared state run in parallel server-side, so the context a request has to fit is
the state plus its longest question, not the sum of every question. Batches are
packed against that limit and dispatched concurrently under a semaphore.
"""

from __future__ import annotations

import asyncio
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field

from typesafe_sdk import Choice, Noul, NoulCriteria, Score

from .cache import NullCache, answer_key
from .client import DEFAULT_MODEL
from .validate import validate_response

#: Published input price for Jev.
COST_PER_MILLION_INPUT_TOKENS = 0.042
#: Hard context ceiling for state plus the longest question in one request.
STATE_TOKEN_LIMIT = 32_000
#: Where sieve stops packing, leaving room for the estimator being approximate.
BATCH_TOKEN_LIMIT = 30_000
#: Concurrent in-flight requests.
DEFAULT_CONCURRENCY = 16


def estimate_tokens(text: str) -> int:
    """Rough token count. Four characters per token, rounded up."""
    return math.ceil(len(text) / 4)


def cost_for_input_tokens(tokens: int) -> float:
    return tokens * COST_PER_MILLION_INPUT_TOKENS / 1_000_000


@dataclass(frozen=True)
class ScoreItem:
    """One thing to be scored, with the text the cache keys on."""

    id: str
    text: str
    payload: dict
    """What is placed in the request state for this item."""


@dataclass(frozen=True)
class QuestionSpec:
    """The Noul asked once per item, and where the items live in the state."""

    state_field: str
    instructions: str
    """Rendered with `{ref}` replaced by a backticked path into the state."""
    criteria_true: str
    criteria_false: str

    def question_for(self, index: int) -> Noul:
        ref = f"`{self.state_field}[{index}]`"
        return Noul(
            instructions=self.instructions.format(ref=ref),
            criteria=NoulCriteria(true=self.criteria_true, false=self.criteria_false),
        )

    def fingerprint(self) -> str:
        """Everything this spec contributes to a request, for the cache key."""
        return "\x1f".join((self.state_field, self.instructions, self.criteria_true, self.criteria_false))

    def question_tokens(self) -> int:
        """Tokens of one rendered question, including its criteria."""
        sample = self.instructions.format(ref=f"`{self.state_field}[999]`")
        return estimate_tokens(sample + self.criteria_true + self.criteria_false) + 16


@dataclass(frozen=True)
class ChoiceSpec:
    """A Choice over each self-contained item in a batch."""

    state_field: str
    instructions: str
    criteria: dict[str, str]

    def question_for(self, index: int) -> Choice:
        ref = f"`{self.state_field}[{index}]`"
        return Choice(instructions=self.instructions.format(ref=ref), criteria=self.criteria)

    def fingerprint(self) -> str:
        return "\x1f".join((self.state_field, self.instructions, json.dumps(self.criteria, sort_keys=True)))

    def question_tokens(self) -> int:
        sample = self.instructions.format(ref=f"`{self.state_field}[999]`")
        return estimate_tokens(sample + json.dumps(self.criteria)) + 16


@dataclass(frozen=True)
class ScoreSpec:
    """A Score over each item in a batch, against an ordered rubric."""

    state_field: str
    instructions: str
    levels: tuple[str, ...]
    """Level descriptions in order, lowest first; the answer is an index into them."""

    def question_for(self, index: int) -> Score:
        ref = f"`{self.state_field}[{index}]`"
        return Score(instructions=self.instructions.format(ref=ref), criteria=list(self.levels))

    def fingerprint(self) -> str:
        return "\x1f".join((self.state_field, self.instructions, json.dumps(list(self.levels))))

    def question_tokens(self) -> int:
        sample = self.instructions.format(ref=f"`{self.state_field}[999]`")
        return estimate_tokens(sample + json.dumps(list(self.levels))) + 16


@dataclass(frozen=True)
class DirectChoiceSpec:
    """One Choice per item, with the item's payload used as the whole state."""

    instructions: str

    def question_for(self, item: ScoreItem) -> Choice:
        criteria = {
            route["id"]: route["description"] + (
                "  Also known as: " + ", ".join(route["aliases"]) if route["aliases"] else ""
            )
            for route in item.payload["routes"]
        }
        criteria["none"] = "no listed route fits"
        return Choice(instructions=self.instructions, criteria=criteria)

    def fingerprint(self) -> str:
        return self.instructions

    def question_tokens(self) -> int:
        return estimate_tokens(self.instructions) + 16


#: Any question shape JevScorer knows how to batch and validate.
AnySpec = QuestionSpec | ChoiceSpec | ScoreSpec | DirectChoiceSpec


@dataclass
class Usage:
    """What a run actually consumed."""

    input_tokens: int = 0
    output_tokens: int = 0
    requests: int = 0
    cache_hits: int = 0

    @property
    def tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def cost_usd(self) -> float:
        return cost_for_input_tokens(self.input_tokens)

    def add(self, other: Usage) -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.requests += other.requests
        self.cache_hits += other.cache_hits

    def as_dict(self) -> dict:
        return {
            "tokens": self.tokens,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "requests": self.requests,
            "cache_hits": self.cache_hits,
            "cost_usd": round(self.cost_usd, 6),
        }


@dataclass
class ScoreRun:
    """Scores for every item that was reached, plus what it cost."""

    scores: dict[str, float | dict[str, float]] = field(default_factory=dict)
    choices: dict[str, str | float] = field(default_factory=dict)
    """The chosen label per item for a Choice, the expected score for a Score."""
    confidences: dict[str, float | None] = field(default_factory=dict)
    usage: Usage = field(default_factory=Usage)
    budget_exhausted: bool = False


def state_overhead_tokens(question: str, state_field: str) -> int:
    """Tokens of the state wrapper before any item is added to it."""
    return estimate_tokens(json.dumps({"question": question, state_field: []}))


def plan_batches(
    question: str,
    items: Sequence[ScoreItem],
    spec: AnySpec,
    token_limit: int = BATCH_TOKEN_LIMIT,
    max_items: int | None = None,
) -> list[list[ScoreItem]]:
    """Pack items into requests that stay under `token_limit` for state + longest question.

    An item that alone exceeds the limit still gets its own request: sieve never
    silently drops a unit, and enumeration already clips oversized ones.
    """
    base = state_overhead_tokens(question, spec.state_field) + spec.question_tokens()
    batches: list[list[ScoreItem]] = []
    current: list[ScoreItem] = []
    total = base
    for item in items:
        cost = estimate_tokens(json.dumps(item.payload))
        over_tokens = current and total + cost > token_limit
        over_count = max_items is not None and len(current) >= max_items
        if over_tokens or over_count:
            batches.append(current)
            current, total = [], base
        current.append(item)
        total += cost
    if current:
        batches.append(current)
    return batches


class JevScorer:
    """Scores items with one question each, caching, batching, and a budget."""

    def __init__(
        self,
        client,
        *,
        model: str = DEFAULT_MODEL,
        cache=None,
        concurrency: int = DEFAULT_CONCURRENCY,
        budget_usd: float | None = None,
        token_limit: int = BATCH_TOKEN_LIMIT,
    ) -> None:
        self.client = client
        self.model = model
        self.cache = cache if cache is not None else NullCache()
        self.concurrency = max(1, concurrency)
        self.budget_usd = budget_usd
        self.token_limit = token_limit
        self.spent_usd = 0.0
        self.usage = Usage()
        # The char-per-token estimate ignores request framing and per-question overhead,
        # so it under-predicts. Observed spend calibrates it as the run goes on.
        self._estimated_usd = 0.0
        self._actual_usd = 0.0

    def _estimated_cost(self, question: str, batch: list[ScoreItem], spec: AnySpec) -> float:
        """Billed input tokens: the state once, plus every question in the request."""
        if isinstance(spec, DirectChoiceSpec):
            item = batch[0]
            return cost_for_input_tokens(
                estimate_tokens(json.dumps(item.payload))
                + estimate_tokens(json.dumps(spec.question_for(item).criteria))
                + spec.question_tokens()
            )
        state = state_overhead_tokens(question, spec.state_field)
        state += sum(estimate_tokens(json.dumps(item.payload)) for item in batch)
        return cost_for_input_tokens(state + len(batch) * spec.question_tokens())

    def _projected_cost(self, question: str, batch: list[ScoreItem], spec: AnySpec) -> float:
        """What this batch is expected to cost, corrected by what requests have really cost."""
        estimate = self._estimated_cost(question, batch, spec)
        if self._estimated_usd <= 0.0:
            return estimate
        return estimate * max(1.0, self._actual_usd / self._estimated_usd)

    async def score(
        self,
        question: str,
        items: Sequence[ScoreItem],
        spec: AnySpec,
        max_items_per_batch: int | None = None,
    ) -> ScoreRun:
        """Return a validated Noul, Choice, or Score distribution per item."""
        run = ScoreRun()
        pending: list[ScoreItem] = []
        keys: dict[str, str] = {}
        fingerprint = spec.fingerprint()
        for item in items:
            key = answer_key(self.model, question, item.text, fingerprint)
            keys[item.id] = key
            if isinstance(spec, DirectChoiceSpec):
                hit = self.cache.get_route(key)
            elif isinstance(spec, ScoreSpec):
                hit = self.cache.get_score(key)
            elif isinstance(spec, ChoiceSpec):
                hit = self.cache.get_choice(key)
            else:
                hit = self.cache.get(key)
            if hit is None:
                pending.append(item)
            else:
                if isinstance(spec, DirectChoiceSpec):
                    run.scores[item.id] = hit["probabilities"]
                    run.choices[item.id] = hit["choice"]
                    run.confidences[item.id] = hit.get("confidence")
                elif isinstance(spec, ScoreSpec):
                    run.scores[item.id] = hit["probabilities"]
                    run.choices[item.id] = hit["score"]
                    run.confidences[item.id] = hit.get("confidence")
                else:
                    run.scores[item.id] = hit
                run.usage.cache_hits += 1
        if not pending:
            self.usage.add(run.usage)
            return run

        batches = (
            [[item] for item in pending]
            if isinstance(spec, DirectChoiceSpec)
            else plan_batches(question, pending, spec, self.token_limit, max_items_per_batch)
        )
        semaphore = asyncio.Semaphore(self.concurrency)
        reserved = 0.0
        stop = False
        lock = asyncio.Lock()

        async def run_batch(batch: list[ScoreItem]) -> None:
            nonlocal reserved, stop
            async with semaphore:
                async with lock:
                    if stop:
                        return
                    estimate = self._estimated_cost(question, batch, spec)
                    projected = self._projected_cost(question, batch, spec)
                    if self.budget_usd is not None and self.spent_usd + reserved + projected > self.budget_usd:
                        stop = True
                        run.budget_exhausted = True
                        return
                    reserved += projected
                # The reservation is released in the same critical section that records
                # what the request really cost, so a batch checking the budget can never
                # see a request that has stopped being reserved and is not yet spent.
                settled = False
                try:
                    if isinstance(spec, DirectChoiceSpec):
                        state = batch[0].payload
                        questions = {"q0": spec.question_for(batch[0])}
                    else:
                        state = {"question": question, spec.state_field: [item.payload for item in batch]}
                        questions = {f"q{i}": spec.question_for(i) for i in range(len(batch))}
                    response = await self.client.system_one(state, questions, model=self.model)
                    answers = validate_response(response, questions)
                    usage = getattr(response, "usage", None)
                    input_tokens = int(getattr(usage, "input_tokens", None) or 0)
                    output_tokens = int(getattr(usage, "output_tokens", None) or 0)
                    async with lock:
                        actual = cost_for_input_tokens(input_tokens)
                        self.spent_usd += actual
                        self._estimated_usd += estimate
                        self._actual_usd += actual
                        reserved -= projected
                        settled = True
                        run.usage.input_tokens += input_tokens
                        run.usage.output_tokens += output_tokens
                        run.usage.requests += 1
                        for i, item in enumerate(batch):
                            answer = answers[f"q{i}"]
                            if isinstance(spec, DirectChoiceSpec):
                                distribution = dict(answer.probabilities)
                                confidence = getattr(answer, "confidence", None)
                                run.scores[item.id] = distribution
                                run.choices[item.id] = answer.choice
                                run.confidences[item.id] = confidence
                                self.cache.put_route(keys[item.id], {
                                    "choice": answer.choice,
                                    "probabilities": distribution,
                                    "confidence": confidence,
                                })
                            elif isinstance(spec, ScoreSpec):
                                distribution = {int(k): float(v) for k, v in answer.probabilities.items()}
                                confidence = getattr(answer, "confidence", None)
                                run.scores[item.id] = distribution
                                run.choices[item.id] = float(answer.score)
                                run.confidences[item.id] = confidence
                                self.cache.put_score(keys[item.id], {
                                    "score": float(answer.score),
                                    "probabilities": distribution,
                                    "confidence": confidence,
                                })
                            elif isinstance(spec, ChoiceSpec):
                                distribution = dict(answer.probabilities)
                                run.scores[item.id] = distribution
                                self.cache.put_choice(keys[item.id], distribution)
                            else:
                                run.scores[item.id] = answer
                                self.cache.put(keys[item.id], answer)
                finally:
                    if not settled:
                        async with lock:
                            reserved -= projected

        await asyncio.gather(*(run_batch(batch) for batch in batches))
        self.usage.add(run.usage)
        return run
