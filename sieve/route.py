"""Select one caller supplied route with a Choice judgment."""

from __future__ import annotations

import json
import math
from typing import Any

from .client import DEFAULT_MODEL, build_client
from .errors import InvalidAnswerError, SieveError
from .jev import DEFAULT_CONCURRENCY, DirectChoiceSpec, JevScorer, ScoreItem

ROUTES_PER_REQUEST = 10
INSTRUCTIONS = "Which route in `routes` should handle `ask`? Pick `none` when no listed route fits."
CHOICE_SPEC = DirectChoiceSpec(instructions=INSTRUCTIONS)


def _validate_routes(routes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(routes, list) or not routes:
        raise ValueError("routes must be a nonempty list")
    seen: set[str] = set()
    for route in routes:
        if not isinstance(route, dict):
            raise ValueError("each route must be an object")
        route_id = route.get("id")
        if not isinstance(route_id, str) or not route_id:
            raise ValueError("each route needs a nonempty string id")
        if route_id == "none":
            raise ValueError("route id 'none' is reserved")
        if route_id in seen:
            raise ValueError(f"duplicate route id: {route_id!r}")
        seen.add(route_id)
        if not isinstance(route.get("description"), str):
            raise ValueError(f"route {route_id!r} needs a description string")
        if not isinstance(route.get("aliases"), list) or any(not isinstance(a, str) for a in route["aliases"]):
            raise ValueError(f"route {route_id!r} needs aliases as a list of strings")
    return [{"id": r["id"], "description": r["description"], "aliases": r["aliases"]} for r in routes]


def _item(ask: str, routes: list[dict[str, Any]], index: int) -> ScoreItem:
    state = {"ask": ask, "routes": routes}
    return ScoreItem(id=str(index), text=json.dumps(state, sort_keys=True, ensure_ascii=False), payload=state)


async def jev_route(
    ask: str,
    routes: list[dict[str, Any]],
    budget_usd: float = 0.10,
    *,
    client=None,
    cache=None,
    model: str = DEFAULT_MODEL,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> dict[str, Any]:
    """Choose a route or none; use a shortlist pass for more than ten routes."""
    if not isinstance(ask, str) or not ask.strip():
        raise ValueError("ask must be a nonempty string")
    if not isinstance(budget_usd, (int, float)) or not math.isfinite(budget_usd) or budget_usd <= 0:
        raise ValueError("budget_usd must be positive")
    routes = _validate_routes(routes)
    stages = 1 if len(routes) <= ROUTES_PER_REQUEST else 2
    batches = math.ceil(len(routes) / ROUTES_PER_REQUEST) if stages == 2 else 1
    owned_client = client is None
    client = client or build_client(model=model)
    scorer = JevScorer(client, model=model, cache=cache, concurrency=concurrency, budget_usd=budget_usd)
    exhausted = False
    try:
        if stages == 2:
            chunks = [routes[i:i + ROUTES_PER_REQUEST] for i in range(0, len(routes), ROUTES_PER_REQUEST)]
            run = await scorer.score(ask, [_item(ask, chunk, i) for i, chunk in enumerate(chunks)], CHOICE_SPEC)
            exhausted = run.budget_exhausted
            if len(run.scores) != len(chunks):
                raise SieveError("budget exhausted before all route batches were scored")
            survivors = []
            for i, chunk in enumerate(chunks):
                probabilities = run.scores[str(i)]
                top = sorted(chunk, key=lambda route: -probabilities[route["id"]])[:2]
                survivors.extend(top)
            # More than five initial batches can yield over ten survivors.
            # Keep the strongest ten first-pass candidates for the final Choice.
            if len(survivors) > ROUTES_PER_REQUEST:
                survivors = sorted(
                    survivors,
                    key=lambda route: -max(
                        run.scores[str(i)][route["id"]]
                        for i, chunk in enumerate(chunks) if route in chunk
                    ),
                )[:ROUTES_PER_REQUEST]
            routes = survivors
        final = await scorer.score(ask, [_item(ask, routes, batches)], CHOICE_SPEC)
        exhausted |= final.budget_exhausted
        if not final.scores:
            raise SieveError("budget exhausted before the final route choice")
        probabilities = final.scores[str(batches)]
        choice = final.choices[str(batches)]
        confidence = final.confidences.get(str(batches))
        source = "sdk" if confidence is not None else "max_probability"
        if confidence is None:
            confidence = probabilities[choice]
        elif not isinstance(confidence, (int, float)) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise InvalidAnswerError(f"route confidence {confidence!r} is outside [0, 1]")
        return {
            "choice": choice,
            "confidence": float(confidence),
            "confidence_source": source,
            "probabilities": probabilities,
            "usage": {**scorer.usage.as_dict(), "budget_exhausted": exhausted, "stages": stages, "batches": batches},
        }
    finally:
        if owned_client:
            await client.aclose()
