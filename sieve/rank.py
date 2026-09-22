"""jev_rank: a generic reranker over candidates the caller already holds."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from .client import DEFAULT_MODEL, build_client
from .jev import DEFAULT_CONCURRENCY, JevScorer, QuestionSpec, ScoreItem

#: Candidates per Jev request.
CANDIDATES_PER_BATCH = 40
#: Each candidate is cut to this many characters before it is sent.
MAX_CANDIDATE_CHARS = 2000

CANDIDATE_SPEC = QuestionSpec(
    state_field="candidates",
    instructions=(
        "The candidate at {ref} is one item from a list the caller is trying to rank. "
        "Is it relevant to the question in `question` — would returning it help answer "
        "that question?"
    ),
    criteria_true=(
        "The candidate's text addresses what the question asks about, or contains the "
        "information needed to answer it."
    ),
    criteria_false=(
        "The candidate is about something else, or is only topically adjacent without "
        "bearing on what the question asks."
    ),
)


def truncate(text: str, limit: int = MAX_CANDIDATE_CHARS) -> str:
    return text if len(text) <= limit else text[:limit]


async def jev_rank(
    question: str,
    candidates: Sequence[dict[str, Any]],
    top_k: int | None = None,
    threshold: float = 0.0,
    *,
    client=None,
    cache=None,
    model: str = DEFAULT_MODEL,
    concurrency: int = DEFAULT_CONCURRENCY,
    budget_usd: float | None = None,
) -> dict:
    """Score each candidate against `question` and return them sorted by probability."""
    items: list[ScoreItem] = []
    ids: list[str] = []
    for position, candidate in enumerate(candidates):
        candidate_id = str(candidate.get("id", position))
        payload = {"id": candidate_id, "text": truncate(str(candidate.get("text", "")))}
        ids.append(candidate_id)
        items.append(
            ScoreItem(
                id=f"{position}:{candidate_id}",
                text=json.dumps(payload, sort_keys=True, ensure_ascii=False),
                payload=payload,
            )
        )

    owned_client = client is None
    client = client or build_client(model=model)
    scorer = JevScorer(
        client,
        model=model,
        cache=cache,
        concurrency=concurrency,
        budget_usd=budget_usd,
    )
    try:
        run = await scorer.score(question, items, CANDIDATE_SPEC, max_items_per_batch=CANDIDATES_PER_BATCH)
    finally:
        if owned_client:
            await client.aclose()

    rows = []
    for candidate_id, item in zip(ids, items):
        probability = run.scores.get(item.id)
        if probability is None or probability < threshold:
            continue
        rows.append({"id": candidate_id, "probability": round(probability, 4)})
    rows.sort(key=lambda row: (-row["probability"], row["id"]))
    if top_k is not None:
        rows = rows[:top_k]

    payload = {"results": rows, "candidates_scored": len(items)}
    payload.update(scorer.usage.as_dict())
    payload["budget_exhausted"] = run.budget_exhausted
    return payload
