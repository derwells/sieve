"""Ad hoc Jev judgments over items the caller already holds.

This is the open-ended path: hand it a list of items and a question you wrote
for this moment, and get one probability (``judge``), one distribution over
labels (``choose``), or one level on a stated scale (``score``) per item, with
the same batching, caching, concurrency, budget, and validation the fixed tools
use. The fixed tools are conveniences over this core.

Usable over MCP (``jev_ask``), from Python, or as a CLI (``python -m sieve.ask``,
or ``bin/sieve-ask``) that reads items as JSON on stdin and prints JSON on stdout.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Mapping, Sequence
from typing import Any

from .cache import ask_cache
from .client import DEFAULT_MODEL, build_client
from .errors import SieveError
from .jev import (
    DEFAULT_CONCURRENCY,
    AnySpec,
    ChoiceSpec,
    JevScorer,
    QuestionSpec,
    ScoreItem,
    ScoreSpec,
)

#: Items per Jev request. Same as jev_rank; the token packer still splits further if needed.
ITEMS_PER_BATCH = 40
#: Each item's text is cut to this many characters before it is sent.
MAX_ITEM_CHARS = 4000
#: The question shapes jev_ask accepts.
KINDS = ("judge", "choose", "score")

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


async def _run(
    items: Sequence[Item],
    question: str,
    spec: AnySpec,
    *,
    max_chars: int,
    client,
    cache,
    model: str,
    concurrency: int,
    budget_usd: float | None,
):
    """Score every item with `spec` and hand back the ids, the items, the run, and usage."""
    ids, score_items = _normalise(items, max_chars)
    owned = client is None
    client = client or build_client(model=model)
    scorer = JevScorer(client, model=model, cache=cache, concurrency=concurrency, budget_usd=budget_usd)
    try:
        run = await scorer.score(question, score_items, spec, max_items_per_batch=ITEMS_PER_BATCH)
    finally:
        if owned:
            await client.aclose()
    return ids, score_items, run, scorer


def _envelope(rows: list[dict], score_items: list[ScoreItem], run, scorer) -> dict:
    out = {"results": rows, "items_scored": len(score_items)}
    out.update(scorer.usage.as_dict())
    out["budget_exhausted"] = run.budget_exhausted
    return out


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
    ids, score_items, run, scorer = await _run(
        items, question, spec, max_chars=max_chars, client=client, cache=cache,
        model=model, concurrency=concurrency, budget_usd=budget_usd,
    )

    rows = []
    for item_id, item in zip(ids, score_items):
        p = run.scores.get(item.id)
        if p is None or p < threshold:
            continue
        rows.append({"id": item_id, "probability": round(p, 4)})
    rows.sort(key=lambda r: (-r["probability"], r["id"]))
    if top_k is not None:
        rows = rows[:top_k]
    return _envelope(rows, score_items, run, scorer)


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
    ids, score_items, run, scorer = await _run(
        items, question, spec, max_chars=max_chars, client=client, cache=cache,
        model=model, concurrency=concurrency, budget_usd=budget_usd,
    )

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
    return _envelope(rows, score_items, run, scorer)


async def score(
    items: Sequence[Item],
    question: str,
    levels: Sequence[str],
    *,
    max_chars: int = MAX_ITEM_CHARS,
    client=None,
    cache=None,
    model: str = DEFAULT_MODEL,
    concurrency: int = DEFAULT_CONCURRENCY,
    budget_usd: float | None = None,
) -> dict:
    """Place every item on an ordered scale and return its level and the distribution.

    ``levels`` describes each rung in order, lowest first. Jev answers with a
    probability per rung; ``level`` is the most likely rung and ``expected`` is
    the probability-weighted score the model reports.
    """
    if len(levels) < 2:
        raise SieveError("score needs at least two levels.")
    spec = ScoreSpec(
        state_field="items",
        instructions=(
            "Consider the item at {ref} in the light of the question in `question`. "
            "Give this item alone the score whose description fits it best."
        ),
        levels=tuple(str(level) for level in levels),
    )
    ids, score_items, run, scorer = await _run(
        items, question, spec, max_chars=max_chars, client=client, cache=cache,
        model=model, concurrency=concurrency, budget_usd=budget_usd,
    )

    rows = []
    for item_id, item in zip(ids, score_items):
        dist = run.scores.get(item.id)
        if dist is None:
            continue
        best = max(dist, key=lambda k: (dist[k], -k))
        rows.append({
            "id": item_id,
            "level": best,
            "label": spec.levels[best] if best < len(spec.levels) else None,
            "expected": round(float(run.choices[item.id]), 4),
            "probabilities": {str(k): round(v, 4) for k, v in sorted(dist.items())},
        })
    out = _envelope(rows, score_items, run, scorer)
    out["levels"] = list(spec.levels)
    return out


async def ask(
    question: str,
    items: Sequence[Item],
    kind: str = "judge",
    *,
    yes: str | None = None,
    no: str | None = None,
    options: Mapping[str, str] | None = None,
    levels: Sequence[str] | None = None,
    top_k: int | None = None,
    threshold: float = 0.0,
    max_chars: int = MAX_ITEM_CHARS,
    client=None,
    cache=None,
    model: str = DEFAULT_MODEL,
    concurrency: int = DEFAULT_CONCURRENCY,
    budget_usd: float | None = None,
) -> dict:
    """Run one caller-written question of shape ``kind`` over ``items``.

    ``kind`` selects the primitive: ``judge`` needs ``yes`` and ``no``, ``choose``
    needs ``options``, ``score`` needs ``levels``. The result carries ``kind`` so a
    caller can tell the shapes apart.
    """
    if kind not in KINDS:
        raise SieveError(f"kind must be one of {', '.join(KINDS)}; got {kind!r}.")
    if not items:
        raise SieveError("items is empty; there is nothing to ask about.")
    common = dict(max_chars=max_chars, client=client, cache=cache, model=model,
                  concurrency=concurrency, budget_usd=budget_usd)
    if kind == "judge":
        if not yes or not no:
            raise SieveError("kind='judge' needs both yes and no criteria.")
        out = await judge(items, question, yes=yes, no=no, top_k=top_k, threshold=threshold, **common)
    elif kind == "choose":
        if not options:
            raise SieveError("kind='choose' needs options as {label: when it applies}.")
        out = await choose(items, question, options, **common)
    else:
        if not levels:
            raise SieveError("kind='score' needs levels as an ordered list, lowest first.")
        out = await score(items, question, levels, **common)
    out["kind"] = kind
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

    s = sub.add_parser("score", help="one level on an ordered scale per item")
    s.add_argument("question")
    s.add_argument("--level", action="append", default=[], metavar="DESCRIPTION",
                   help="repeatable, lowest first; two or more")

    for p in (j, c, s):
        p.add_argument("--items", help="JSON file; default stdin")
        p.add_argument("--budget-usd", type=float)
        p.add_argument("--max-chars", type=int, default=MAX_ITEM_CHARS)
        p.add_argument("--model", default=DEFAULT_MODEL)
        p.add_argument("--no-cache", action="store_true", help="skip the on-disk answer cache")

    args = parser.parse_args(argv)
    cache = None
    try:
        items = _read_items(args.items)
        cache = None if args.no_cache else ask_cache()
        kwargs = dict(
            kind=args.cmd, max_chars=args.max_chars, model=args.model,
            budget_usd=args.budget_usd, cache=cache,
        )
        if args.cmd == "judge":
            kwargs.update(yes=args.yes, no=args.no, top_k=args.top_k, threshold=args.threshold)
        elif args.cmd == "choose":
            kwargs.update(options=_parse_options(args.option))
        else:
            kwargs.update(levels=args.level)
        out = asyncio.run(ask(args.question, items, **kwargs))
    except SieveError as exc:
        print(f"sieve-ask: {exc}", file=sys.stderr)
        return 2
    finally:
        if cache is not None:
            cache.close()
    json.dump(out, sys.stdout, indent=2, ensure_ascii=False)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
