"""Recall eval for jev_grep on real past asks.

Each ask is a real request that was made to a coding agent in some repository;
the ground truth is the set of files the agent that answered it edited. Files the
agent created are listed but excluded from the denominator: they did not exist
when the ask was made, so no retriever could have found them.

The asks themselves live in `evals/asks.json`, which is not committed because it
names private repositories and file paths. `evals/asks.example.json` shows the
schema, with the anonymised asks used in `evals/recall-2026-09-22.md` and placeholder paths.

Every ask is evaluated against a snapshot of the *parent* of the commit that
answered it, extracted with `git archive <commit>^ | tar -x` into a temporary
directory. The repositories themselves are never checked out, switched, or
written to. Run 1 scored the repositories at HEAD, where the fix's own new files
competed for top-10 slots that did not exist when the ask was made.

Every run uses a NullCache and a fresh client, so cost and wall time are cold and
no score leaks between batch sizes (the cache key does not include batch context).
Runs are sequential; batches inside a run stay concurrent.

    export TYPESAFE_API_KEY=...
    uv run python evals/recall.py run       # main table, resumable, skips finished ones
    uv run python evals/recall.py summary   # markdown tables from the stored runs
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sieve.cache import NullCache  # noqa: E402
from sieve.grep import UNITS_PER_REQUEST, jev_grep  # noqa: E402

RAW = Path(__file__).resolve().parent / "raw"
#: Run 1 (repositories at HEAD, head-only previews) is kept under `raw/run1/`; the
#: rerun on parent snapshots with outline previews writes to `raw/run2/`.
DEFAULT_RUN = "run2"

#: Enough rows to see every file and its probability in files mode.
FILES_TOP_K = 100_000
#: In functions mode top_k also sets the survivor cut (SURVIVOR_MULTIPLE * top_k),
#: so it stays small enough to keep the splitting pass affordable while still
#: returning more than 20 distinct files.
FUNCTIONS_TOP_K = 40
#: Ranking is measured on its own; the filter is reported separately.
EVAL_THRESHOLD = 0.0
DEFAULT_THRESHOLD = 0.5
SWEEP = (1, 2, 4, 8)
#: The main table carries both candidate settings so 4 and 8 are compared on the
#: same snapshots and the same preview.
MAIN_UNITS = (4, 8)
#: One extra files-mode row per large-repo ask, well past anything measured before.
EXPLORATORY_UNITS = 16
EXPLORATORY_ASKS = ("repo-B-packshot", "repo-B-image-fallback")

ASKS_PATH = Path(__file__).resolve().parent / "asks.json"
EXAMPLE_PATH = Path(__file__).resolve().parent / "asks.example.json"


def load_asks(path: Path = ASKS_PATH) -> list[dict]:
    """The asks to evaluate, read from `evals/asks.json`.

    The file is a list of objects: `id`, `repo` (absolute path), `commit` (the
    commit that answered the ask), `ask` (the request text), `existed` (the
    ground-truth files that existed when the ask was made) and `created` (files
    the answer added, excluded from the denominator).
    """
    if not path.exists():
        raise SystemExit(
            f"{path} does not exist. Copy {EXAMPLE_PATH.name} to {path.name} and "
            f"fill in your own repositories, commits and file paths."
        )
    asks = json.loads(path.read_text(encoding="utf-8"))
    for ask in asks:
        ask.setdefault("created", [])
    return asks


#: Filled in by main() so the file can be imported without an asks.json present.
ASKS: list[dict] = []
ASK_BY_ID: dict[str, dict] = {}


@contextlib.contextmanager
def parent_snapshot(repo: str, commit: str):
    """The repository as it stood *before* `commit`, unpacked into a temp directory.

    `git archive` writes a tarball to stdout and touches nothing in the repository:
    no checkout, no worktree, no index. Untracked and ignored files are absent,
    which is what "the tree the ask was made against" means.
    """
    archive = subprocess.run(
        ["git", "-C", repo, "archive", f"{commit}^"],
        check=True,
        stdout=subprocess.PIPE,
    )
    with tempfile.TemporaryDirectory(prefix="sieve-eval-") as tmp:
        subprocess.run(["tar", "-x", "-C", tmp], input=archive.stdout, check=True)
        yield Path(tmp)


def parent_sha(repo: str, commit: str) -> str:
    out = subprocess.run(
        ["git", "-C", repo, "rev-parse", "--short", f"{commit}^"],
        check=True,
        capture_output=True,
        text=True,
    )
    return out.stdout.strip()


def ranked_files(payload: dict) -> list[str]:
    """Distinct file paths in result order (functions mode is deduped by file)."""
    order: list[str] = []
    seen: set[str] = set()
    for row in payload["results"]:
        if row["path"] not in seen:
            seen.add(row["path"])
            order.append(row["path"])
    return order


def row_files(payload: dict, k: int) -> set[str]:
    """Files touched by the top-k rows, without deduping before the cut."""
    return {row["path"] for row in payload["results"][:k]}


def best_probability(payload: dict) -> dict[str, float]:
    best: dict[str, float] = {}
    for row in payload["results"]:
        p = row["probability"]
        if p > best.get(row["path"], -1.0):
            best[row["path"]] = p
    return best


def score(payload: dict, ask: dict, mode: str) -> dict:
    truth = ask["existed"]
    order = ranked_files(payload)
    probs = best_probability(payload)
    all_probs = [row["probability"] for row in payload["results"]]
    out = {
        "recall@10": sum(f in order[:10] for f in truth) / len(truth),
        "recall@20": sum(f in order[:20] for f in truth) / len(truth),
        "found@10": [f for f in truth if f in order[:10]],
        "missed@10": [f for f in truth if f not in order[:10]],
        "missed@20": [f for f in truth if f not in order[:20]],
        "ranks": {f: (order.index(f) + 1 if f in order else None) for f in truth},
        "truth_probability": {f: probs.get(f) for f in truth},
        "created_probability": {f: probs.get(f) for f in ask["created"]},
        "under_default_threshold": [
            f for f in truth if probs.get(f) is not None and probs[f] < DEFAULT_THRESHOLD
        ],
        "probability_min": min(all_probs) if all_probs else None,
        "probability_max": max(all_probs) if all_probs else None,
        "distinct_files_returned": len(order),
    }
    if mode == "functions":
        out["row_recall@10"] = sum(f in row_files(payload, 10) for f in truth) / len(truth)
        out["row_recall@20"] = sum(f in row_files(payload, 20) for f in truth) / len(truth)
    return out


async def run_one(ask: dict, mode: str, units: int, budget: float) -> dict:
    top_k = FILES_TOP_K if mode == "files" else FUNCTIONS_TOP_K
    with parent_snapshot(ask["repo"], ask["commit"]) as root:
        started = time.monotonic()
        payload = await jev_grep(
            question=ask["ask"],
            path=str(root),
            mode=mode,
            top_k=top_k,
            threshold=EVAL_THRESHOLD,
            budget_usd=budget,
            cache=NullCache(),
            units_per_request=units,
        )
        elapsed = time.monotonic() - started
    record = {
        "ask_id": ask["id"],
        "snapshot": parent_sha(ask["repo"], ask["commit"]),
        "mode": mode,
        "units_per_request": units,
        "budget_usd": budget,
        "wall_seconds": round(elapsed, 1),
        "units_scored": payload["units_scored"],
        "requests": payload["requests"],
        "input_tokens": payload["input_tokens"],
        "output_tokens": payload["output_tokens"],
        "cost_usd": payload["cost_usd"],
        "budget_exhausted": payload["budget_exhausted"],
        "metrics": score(payload, ask, mode),
        "top_20": payload["results"][:20],
    }
    return record


def plan(only: str | None) -> list[tuple[dict, str, int, float]]:
    jobs: list[tuple[dict, str, int, float]] = []
    if only in (None, "main"):
        for units in MAIN_UNITS:
            for ask in ASKS:
                for mode in ("files", "functions"):
                    jobs.append((ask, mode, units, 0.50))
    if only == "sweep":
        for units in SWEEP:
            for ask in ASKS:
                if units in MAIN_UNITS:
                    continue  # the main table already has this cell
                jobs.append((ask, "files", units, 0.50))
    if only == "exploratory":
        for ask_id in EXPLORATORY_ASKS:
            jobs.append((ASK_BY_ID[ask_id], "files", EXPLORATORY_UNITS, 0.50))
    return jobs


def path_for(run: str, ask_id: str, mode: str, units: int, budget: float) -> Path:
    tag = "" if budget == 0.50 else f"-b{budget:g}"
    return RAW / run / f"{ask_id}.{mode}.u{units}{tag}.json"


async def cmd_run(args) -> None:
    (RAW / args.run).mkdir(parents=True, exist_ok=True)
    for ask, mode, units, budget in plan(args.only):
        if args.ask and ask["id"] != args.ask:
            continue
        out = path_for(args.run, ask["id"], mode, units, budget)
        if out.exists() and not args.force:
            print(f"skip {out.name}")
            continue
        print(f"run  {out.name} ... ", end="", flush=True)
        record = await run_one(ask, mode, units, budget)
        out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        m = record["metrics"]
        print(
            f"r@10={m['recall@10']:.2f} r@20={m['recall@20']:.2f} "
            f"${record['cost_usd']:.4f} {record['wall_seconds']}s "
            f"{record['requests']} reqs"
            + (" BUDGET-EXHAUSTED" if record["budget_exhausted"] else "")
        )


def load(run: str, budget: float = 0.50) -> dict[tuple[str, str, int], dict]:
    runs = {}
    for f in sorted((RAW / run).glob("*.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        if r["budget_usd"] != budget:
            continue
        runs[(r["ask_id"], r["mode"], r["units_per_request"])] = r
    return runs


def cmd_summary(args) -> None:
    runs = load(args.run, args.budget)
    total = sum(
        json.loads(f.read_text())["cost_usd"] for f in sorted((RAW / args.run).glob("*.json"))
    )

    print(f"\n### Main table (threshold={EVAL_THRESHOLD}, run={args.run})\n")
    print("| ask | mode | units/req | scored | recall@10 | recall@20 | GT ranks | in | out | reqs | $ | wall s | exh |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for ask in ASKS:
        for mode in ("files", "functions"):
            for units in MAIN_UNITS:
                r = runs.get((ask["id"], mode, units))
                if r is None:
                    continue
                m = r["metrics"]
                ranks = ", ".join(str(v) for v in m["ranks"].values())
                print(
                    f"| {ask['id']} | {mode} | {units} | {r['units_scored']} | {m['recall@10']:.2f} | "
                    f"{m['recall@20']:.2f} | {ranks} | {r['input_tokens']} | {r['output_tokens']} | "
                    f"{r['requests']} | {r['cost_usd']:.4f} | {r['wall_seconds']} | "
                    f"{'yes' if r['budget_exhausted'] else 'no'} |"
                )

    print("\n### Ground-truth probabilities (files mode)\n")
    print("| ask | units/req | prob min-max | ground-truth probabilities |")
    print("|---|---|---|---|")
    for ask in ASKS:
        for units in (*MAIN_UNITS, EXPLORATORY_UNITS):
            r = runs.get((ask["id"], "files", units))
            if r is None:
                continue
            m = r["metrics"]
            gt = ", ".join(
                f"{Path(f).name} {'-' if p is None else f'{p:.2f}'}"
                for f, p in m["truth_probability"].items()
            )
            print(
                f"| {ask['id']} | {units} | "
                f"{m['probability_min']:.2f}-{m['probability_max']:.2f} | {gt} |"
            )

    print(f"\n### Row-level variant (functions mode, units_per_request={UNITS_PER_REQUEST}, no dedup before the cut)\n")
    print("| ask | deduped r@10 | row r@10 | deduped r@20 | row r@20 |")
    print("|---|---|---|---|---|")
    for ask in ASKS:
        r = runs.get((ask["id"], "functions", UNITS_PER_REQUEST))
        if r is None:
            continue
        m = r["metrics"]
        print(
            f"| {ask['id']} | {m['recall@10']:.2f} | {m.get('row_recall@10', float('nan')):.2f} | "
            f"{m['recall@20']:.2f} | {m.get('row_recall@20', float('nan')):.2f} |"
        )

    print("\n### Gate\n")
    passing = [
        ask["id"]
        for ask in ASKS
        if (r := runs.get((ask["id"], "files", UNITS_PER_REQUEST))) is not None
        and r["metrics"]["recall@10"] >= 0.8
    ]
    print(f"asks with files-mode recall@10 >= 0.8 at units_per_request={UNITS_PER_REQUEST}: "
          f"{len(passing)}/{len(ASKS)} -> {'PASS' if len(passing) >= 4 else 'FAIL'}")
    print(f"passing: {passing}")
    print(f"\nTotal cost of every stored run: ${total:.4f}")


def main() -> None:
    global ASKS, ASK_BY_ID
    ASKS = load_asks()
    ASK_BY_ID = {ask["id"]: ask for ask in ASKS}
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run")
    run.add_argument("--only", choices=["main", "sweep", "exploratory"])
    run.add_argument("--ask")
    run.add_argument("--run", default=DEFAULT_RUN)
    run.add_argument("--force", action="store_true")
    run.set_defaults(fn=lambda a: asyncio.run(cmd_run(a)))
    summary = sub.add_parser("summary")
    summary.add_argument("--run", default=DEFAULT_RUN)
    summary.add_argument("--budget", type=float, default=0.50)
    summary.set_defaults(fn=cmd_summary)
    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
