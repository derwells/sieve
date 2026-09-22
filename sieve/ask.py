"""Ad hoc Jev judgments over items the caller already holds.

The MCP tools are fixed recipes. This module is the open-ended path: hand it a
list of items and a question you wrote for this moment, and get one probability
(``judge``) or one distribution over options (``choose``) per item, with the
same batching, concurrency, budget, and validation the fixed tools use.

Usable from Python or as a CLI (``python -m sieve.ask``, or ``bin/sieve-ask``)
that reads items as JSON on stdin and prints JSON on stdout.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Mapping, Sequence
from typing import Any

from .client import DEFAULT_MODEL, build_client
from .errors import SieveError
from .jev import DEFAULT_CONCURRENCY, ChoiceSpec, JevScorer, QuestionSpec, ScoreItem

#: Items per Jev request. Same as jev_rank; the token packer still splits further if needed.
ITEMS_PER_BATCH = 40
#: Each item's text is cut to this many characters before it is sent.
MAX_ITEM_CHARS = 4000

Item = str | Mapping[str, Any]


def _normalise(items: Sequence[Item], max_chars: int) -> tuple[list[str], list[ScoreItem]]:
    ids: list[str] = []
    score_items: list[ScoreItem] = []
    for position, raw in enumerate(items):
        if isinstance(raw, str):
            item_id, text = str(position), raw
        else:
            item_id = str(raw.get("id", position))
            text = raw.get("text")
            if text is None:
                # Anything else JSON-shaped: send the whole object as the text.
                text = json.dumps({k: v for k, v in raw.items() if k != "id"}, sort_keys=True, ensure_ascii=False)
        text = str(text)
        if len(text) > max_chars:
            text = text[:max_chars]
        payload = {"id": item_id, "text": text}
        ids.append(item_id)
        score_items.append(
            ScoreItem(
                id=f"{position}:{item_id}",
                text=json.dumps(payload, sort_keys=True, ensure_ascii=False),
                payload=payload,
            )
        )
    return ids, score_items


async def judge(
    items: Sequence[Item],
    question: str,
    *,
    yes: str,
    no: str,
    top_k: int | None = None,
    threshold: float = 0.0,
    max_chars: int = MAX_ITEM_CHARS,
    client=None,
    cache=None,
    model: str = DEFAULT_MODEL,
    concurrency: int = DEFAULT_CONCURRENCY,
    budget_usd: float | None = None,
) -> dict:
    """Ask a yes/no question of every item and return each item's probability of yes.

    ``question`` is what you want to know about each item, written to refer to it as
    "the item". ``yes`` and ``no`` describe what makes the answer true or false; Jev
    reads them as criteria, so make them concrete and mutually exclusive.
    """
    spec = QuestionSpec(
        state_field="items",
        instructions=(
            "Consider the item at {ref} in the light of the question in `question`. "
            "Answer the question for this item alone."
        ),
        criteria_true=yes,
        criteria_false=no,
    )
    ids, score_items = _normalise(items, max_chars)
    owned = client is None
    client = client or build_client(model=model)
    scorer = JevScorer(client, model=model, cache=cache, concurrency=concurrency, budget_usd=budget_usd)
    try:
        run = await scorer.score(question, score_items, spec, max_items_per_batch=ITEMS_PER_BATCH)
    finally:
        if owned:
            await client.aclose()

    rows = []
    for item_id, item in zip(ids, score_items):
        p = run.scores.get(item.id)
        if p is None or p < threshold:
            continue
        rows.append({"id": item_id, "probability": round(p, 4)})
    rows.sort(key=lambda r: (-r["probability"], r["id"]))
    if top_k is not None:
        rows = rows[:top_k]
    out = {"results": rows, "items_scored": len(score_items)}
    out.update(scorer.usage.as_dict())
    out["budget_exhausted"] = run.budget_exhausted
    return out


async def choose(
    items: Sequence[Item],
    question: str,
    options: Mapping[str, str],
    *,
    max_chars: int = MAX_ITEM_CHARS,
    client=None,
    cache=None,
    model: str = DEFAULT_MODEL,
    concurrency: int = DEFAULT_CONCURRENCY,
    budget_usd: float | None = None,
) -> dict:
    """Pick one of ``options`` for every item and return the choice plus the full distribution.

    ``options`` maps a short label to a description of when that label applies.
    Add a fallback label such as ``"none"`` yourself if the question needs one.
    """
    if len(options) < 2:
        raise SieveError("choose needs at least two options.")
    spec = ChoiceSpec(
        state_field="items",
        instructions=(
            "Consider the item at {ref} in the light of the question in `question`. "
            "Choose the option that best answers it for this item alone."
        ),
        criteria=dict(options),
    )
    ids, score_items = _normalise(items, max_chars)
    owned = client is None
    client = client or build_client(model=model)
    scorer = JevScorer(client, model=model, cache=cache, concurrency=concurrency, budget_usd=budget_usd)
    try:
        run = await scorer.score(question, score_items, spec, max_items_per_batch=ITEMS_PER_BATCH)
    finally:
        if owned:
            await client.aclose()

    rows = []
    for item_id, item in zip(ids, score_items):
        dist = run.scores.get(item.id)
        if dist is None:
            continue
        best = max(dist, key=lambda k: (dist[k], k))
        rows.append({
            "id": item_id,
            "choice": best,
            "probabilities": {k: round(v, 4) for k, v in dist.items()},
        })
    out = {"results": rows, "items_scored": len(score_items)}
    out.update(scorer.usage.as_dict())
    out["budget_exhausted"] = run.budget_exhausted
    return out


def _read_items(path: str | None) -> list[Item]:
    raw = sys.stdin.read() if path in (None, "-") else open(path, encoding="utf-8").read()
    data = json.loads(raw)
    if isinstance(data, dict) and "items" in data:
        data = data["items"]
    if not isinstance(data, list):
        raise SieveError("items must be a JSON list of strings or {id, text} objects.")
    return data


def _parse_options(pairs: Sequence[str]) -> dict[str, str]:
    options: dict[str, str] = {}
    for pair in pairs:
        label, sep, description = pair.partition("=")
        if not sep or not label or not description:
            raise SieveError(f"--option expects LABEL=DESCRIPTION, got {pair!r}.")
        options[label] = description
    return options


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="sieve-ask",
        description="Ask Jev one question of many items. Items are JSON on stdin (or --items FILE).",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    j = sub.add_parser("judge", help="one yes/no probability per item")
    j.add_argument("question")
    j.add_argument("--yes", required=True, help="what makes the answer true")
    j.add_argument("--no", required=True, help="what makes the answer false")
    j.add_argument("--top-k", type=int)
    j.add_argument("--threshold", type=float, default=0.0)

    c = sub.add_parser("choose", help="one option per item, with the full distribution")
    c.add_argument("question")
    c.add_argument("--option", action="append", default=[], metavar="LABEL=DESCRIPTION", help="repeatable; two or more")

    for p in (j, c):
        p.add_argument("--items", help="JSON file; default stdin")
        p.add_argument("--budget-usd", type=float)
        p.add_argument("--max-chars", type=int, default=MAX_ITEM_CHARS)
        p.add_argument("--model", default=DEFAULT_MODEL)

    args = parser.parse_args(argv)
    try:
        items = _read_items(args.items)
        if args.cmd == "judge":
            out = asyncio.run(judge(
                items, args.question, yes=args.yes, no=args.no, top_k=args.top_k,
                threshold=args.threshold, max_chars=args.max_chars, model=args.model,
                budget_usd=args.budget_usd,
            ))
        else:
            out = asyncio.run(choose(
                items, args.question, _parse_options(args.option), max_chars=args.max_chars,
                model=args.model, budget_usd=args.budget_usd,
            ))
    except SieveError as exc:
        print(f"sieve-ask: {exc}", file=sys.stderr)
        return 2
    json.dump(out, sys.stdout, indent=2, ensure_ascii=False)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
