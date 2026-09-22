"""jev_grep: rank a repository's files, then its functions, against a question."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .cache import AnswerCache
from .client import DEFAULT_MODEL, build_client
from .enumerate import Unit, file_unit, split_file, walk_repo
from .jev import DEFAULT_CONCURRENCY, JevScorer, QuestionSpec, ScoreItem, ScoreRun, Usage

UNIT_SPEC = QuestionSpec(
    state_field="units",
    instructions=(
        "{ref} is one unit of a repository. `path` is where it lives, `lines` is the range it "
        "covers, `size_lines` is how long the whole file is, `kind` says whether it is a whole "
        "file or a piece of one, and `content` is its text — for a whole file, its opening lines "
        "followed by a line-numbered outline of the rest. Judge the unit as a whole, not "
        "just the excerpt shown: would a developer have to open it to answer the question in "
        "`question`?"
    ),
    criteria_true=(
        "Yes. Reading the unit gives you part of the answer to `question`: it states the rule or "
        "the steps, or it defines the index, registry, format, or configuration those steps read. "
        "Most units in a repository do not."
    ),
    criteria_false=(
        "No. The unit is a bystander: one record among many the mechanism happens to act on, an "
        "unrelated topic, or repository furniture such as lockfiles, licences, and build output. "
        "Sharing a repository or a general subject area with the question is not enough."
    ),
)

#: Units per request, on top of the token-limit packing. Judgments do blur when a
#: state gets very large: on a 46-file repository (one question) all 46 units in one
#: request spread 0.12-0.68 and ranked the two known-relevant files 20th and 5th,
#: where four per request spread 0.09-0.90 and ranked them 1st and 2nd. But that is
#: a property of the whole state, not of the count: across the five recall-eval asks
#: (evals/recall-2026-09-22.md, 1 to 16 units per request) the spread stayed
#: ~0.05-0.90 at every setting, and on the rerun files-mode recall@10 at 8 was equal
#: or better than at 4 on all five asks (packshot 0.20 -> 0.40) while every cell was
#: cheaper and faster (the 1,831-file repo $0.0557/7.8 s at 4 against $0.0539/5.6 s at 8).
#: jev_rank shows no collapse either at 40 short candidates in one 6k-token state.
#: 8 units of ~6k preview characters each still leave a request far under the 30k
#: state limit, which is where the blur starts.
UNITS_PER_REQUEST = 8

#: How many files from the files pass get split into functions, as a multiple of top_k.
SURVIVOR_MULTIPLE = 2


@dataclass
class GrepResult:
    results: list[dict]
    units_scored: int
    usage: Usage
    budget_exhausted: bool

    def as_dict(self) -> dict:
        payload = {"results": self.results, "units_scored": self.units_scored}
        payload.update(self.usage.as_dict())
        payload["budget_exhausted"] = self.budget_exhausted
        return payload


def unit_payload(unit: Unit) -> dict:
    payload = {
        "path": unit.path,
        "lines": f"{unit.line_start}-{unit.line_end}",
        "kind": unit.kind,
        "content": unit.text,
    }
    if unit.symbol:
        payload["symbol"] = unit.symbol
    if unit.size_lines is not None:
        payload["size_lines"] = unit.size_lines
    return payload


def score_item(unit: Unit, index: int) -> ScoreItem:
    payload = unit_payload(unit)
    return ScoreItem(
        id=f"{index}:{unit.path}:{unit.line_start}-{unit.line_end}",
        text=json.dumps(payload, sort_keys=True, ensure_ascii=False),
        payload=payload,
    )


def _rows(units: list[Unit], items: list[ScoreItem], run: ScoreRun, threshold: float) -> list[dict]:
    rows = []
    for unit, item in zip(units, items):
        probability = run.scores.get(item.id)
        if probability is None or probability < threshold:
            continue
        rows.append(
            {
                "path": unit.path,
                "line_start": unit.line_start,
                "line_end": unit.line_end,
                "kind": unit.kind,
                "probability": round(probability, 4),
            }
        )
    rows.sort(key=lambda row: (-row["probability"], row["path"], row["line_start"]))
    return rows


async def jev_grep(
    question: str,
    path: str,
    mode: str = "files",
    top_k: int = 20,
    threshold: float = 0.5,
    budget_usd: float = 0.50,
    *,
    client=None,
    cache=None,
    model: str = DEFAULT_MODEL,
    concurrency: int = DEFAULT_CONCURRENCY,
    units_per_request: int | None = None,
) -> dict:
    """Rank units of the repository at `path` by relevance to `question`.

    `mode="files"` scores every file on its path, opening lines, and outline.
    `mode="functions"` runs the files pass first, then splits the strongest
    surviving files into functions (or fixed line chunks where no tree-sitter
    grammar applies) and scores those.

    `units_per_request` overrides `UNITS_PER_REQUEST` for this call; the MCP tool
    does not expose it, it exists for evaluation.
    """
    per_request = UNITS_PER_REQUEST if units_per_request is None else units_per_request
    if per_request < 1:
        raise ValueError(f"units_per_request must be at least 1, got {units_per_request!r}")
    if mode not in ("files", "functions"):
        raise ValueError(f"mode must be 'files' or 'functions', got {mode!r}")
    root = Path(path).expanduser().resolve()

    owned_client = client is None
    client = client or build_client(model=model)
    owned_cache = cache is None
    cache = cache if cache is not None else AnswerCache.for_repo(root)
    scorer = JevScorer(
        client,
        model=model,
        cache=cache,
        concurrency=concurrency,
        budget_usd=budget_usd,
    )
    try:
        paths, _errors = walk_repo(root)
        units = [unit for unit in (file_unit(root, p) for p in paths) if unit is not None]
        items = [score_item(unit, i) for i, unit in enumerate(units)]
        run = await scorer.score(question, items, UNIT_SPEC, per_request)
        units_scored = len(items)
        exhausted = run.budget_exhausted

        if mode == "files":
            rows = _rows(units, items, run, threshold)[:top_k]
            return GrepResult(rows, units_scored, scorer.usage, exhausted).as_dict()

        survivors = [
            unit
            for unit, item in sorted(
                zip(units, items),
                key=lambda pair: -run.scores.get(pair[1].id, 0.0),
            )
            if run.scores.get(item.id, 0.0) >= threshold
        ][: SURVIVOR_MULTIPLE * top_k]

        sub_units: list[Unit] = []
        for unit in survivors:
            sub_units.extend(split_file(root, unit.path))
        sub_items = [score_item(unit, i) for i, unit in enumerate(sub_units)]
        sub_run = await scorer.score(question, sub_items, UNIT_SPEC, per_request)
        rows = _rows(sub_units, sub_items, sub_run, threshold)[:top_k]
        return GrepResult(
            rows,
            units_scored + len(sub_items),
            scorer.usage,
            exhausted or sub_run.budget_exhausted,
        ).as_dict()
    finally:
        if owned_cache:
            cache.close()
        if owned_client:
            await client.aclose()
