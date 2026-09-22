"""Cold, resumable acceptance eval for jev_verify.

Run with the TypeSafe key in the environment. Each case makes exactly one
jev_verify call with a fresh NullCache. Completed JSON files are skipped.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sieve.cache import NullCache  # noqa: E402
from sieve.client import build_client  # noqa: E402
from sieve.verify import jev_verify  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CASES = Path(__file__).with_name("verify-cases.json")
RAW = Path(__file__).parent / "raw" / "verify"
DEFAULT = (0.8, 0.5)


class ModelRecordingClient:
    def __init__(self, client):
        self.client = client
        self.models: list[str] = []

    async def system_one(self, *args, **kwargs):
        response = await self.client.system_one(*args, **kwargs)
        self.models.append(response.model)
        return response


def cases():
    data = json.loads(CASES.read_text(encoding="utf-8"))
    if len(data) != 40 or [c["id"] for c in data] != [f"v{i:02d}" for i in range(1, 41)]:
        raise ValueError("expected ordered cases v01 through v40")
    return data


async def run_one(case):
    client = build_client()
    recorder = ModelRecordingClient(client)
    try:
        started = time.monotonic()
        payload = await jev_verify(records=[case["record"]], base_path=str(ROOT),
                                   cache=NullCache(), client=recorder)
        wall_seconds = time.monotonic() - started
    finally:
        await client.aclose()
    row = payload["results"][0]
    evidence = row["evidence"]
    best = max(evidence, key=lambda e: e["probabilities"]["supports_fully"]) if evidence else None
    return {
        "id": case["id"], "label": case["label"], "alteration": case["alteration"],
        "expected_verdict": case["expected_verdict"], "record": case["record"],
        "model": sorted(set(recorder.models)), "best_window_distribution": best["probabilities"] if best else None,
        "all_window_distributions": [e["probabilities"] for e in evidence],
        "evidence": evidence, "best_evidence": row["best_evidence"],
        "max_contradicts": row["max_contradicts"], "quote_match": row["quote_match"],
        "flags": row["flags"], "citations_checked": row["citations_checked"],
        "unqualified_factual_relay_blocked": payload["unqualified_factual_relay_blocked"],
        "usage": payload["usage"], "wall_seconds": wall_seconds,
    }


async def cmd_run(args):
    RAW.mkdir(parents=True, exist_ok=True)
    for case in cases():
        if args.case and case["id"] != args.case:
            continue
        path = RAW / f"{case['id']}.json"
        if path.exists() and not args.force:
            print(f"skip {case['id']}", flush=True)
            continue
        print(f"run {case['id']} ...", flush=True)
        result = await run_one(case)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        tmp.replace(path)
        print(f"  {result['usage']['requests']} req, {len(result['all_window_distributions'])} windows, "
              f"${result['usage']['cost_usd']:.6f}, {result['wall_seconds']:.2f}s", flush=True)


def load():
    all_cases = cases()
    missing = [c["id"] for c in all_cases if not (RAW / f"{c['id']}.json").exists()]
    if missing:
        raise SystemExit(f"Missing raw results: {', '.join(missing)}")
    rows = [json.loads((RAW / f"{c['id']}.json").read_text(encoding="utf-8")) for c in all_cases]
    for case, row in zip(all_cases, rows):
        if row["record"] != case["record"] or row["expected_verdict"] != case["expected_verdict"]:
            raise SystemExit(f"Case changed after scoring: {case['id']}")
    return rows


def accepted(row, support, contradict):
    d = row["best_window_distribution"]
    return d is not None and row["max_contradicts"] is not None and d["supports_fully"] >= support and row["max_contradicts"] < contradict


def metrics(rows, thresholds):
    support, contradict = thresholds
    altered_accepted = sum(r["label"] == "altered" and accepted(r, support, contradict) for r in rows)
    true_flagged = sum(r["label"] == "true" and not accepted(r, support, contradict) for r in rows)
    flagged = sum(not accepted(r, support, contradict) for r in rows)
    top_correct = sum(r["best_window_distribution"] is not None and
                      max(r["best_window_distribution"], key=r["best_window_distribution"].get) == r["expected_verdict"] for r in rows)
    return altered_accepted, true_flagged, flagged, top_correct


def fit(rows):
    grid = [i / 20 for i in range(21)]
    return min(((s, c) for s in grid for c in grid),
               key=lambda pair: (sum(metrics(rows, pair)[:2]), metrics(rows, pair)[0],
                                 abs(pair[0] - DEFAULT[0]) + abs(pair[1] - DEFAULT[1]),
                                 pair[0], pair[1]))


def table(rows, pair, name):
    bad, missed, flagged, correct = metrics(rows, pair)
    return f"| {name} | {bad}/{sum(r['label'] == 'altered' for r in rows)} | {missed}/{sum(r['label'] == 'true' for r in rows)} | {flagged}/{len(rows)} ({flagged/len(rows):.1%}) | {correct}/{len(rows)} ({correct/len(rows):.1%}) |"


def cmd_summary(_args):
    rows = load()
    fit_rows, test_rows = rows[:20], rows[20:]
    chosen = fit(fit_rows)
    print(f"Chosen thresholds: support {chosen[0]:.2f}, contradict {chosen[1]:.2f}")
    print("\n| set and policy | altered accepted | true flagged | review burden | top verdict accuracy |")
    print("|---|---:|---:|---:|---:|")
    for name, subset, pair in (("Fit, chosen", fit_rows, chosen), ("Test, chosen", test_rows, chosen),
                               ("Fit, default", fit_rows, DEFAULT), ("Test, default", test_rows, DEFAULT)):
        print(table(subset, pair, name))
    print("\n| alteration on test | chosen accepted | default accepted | top verdict correct |")
    print("|---|---:|---:|---:|")
    for kind in ("number", "negation", "scope", "date", "wrong_citation"):
        subset = [r for r in test_rows if r["alteration"] == kind]
        print(f"| {kind} | {metrics(subset, chosen)[0]}/{len(subset)} | {metrics(subset, DEFAULT)[0]}/{len(subset)} | {metrics(subset, chosen)[3]}/{len(subset)} |")
    print("\n| case | label | type | expected | top | support | max contradict | quote | blocked | windows | requests | cost $ | wall s |")
    print("|---|---|---|---|---|---:|---:|---|---|---:|---:|---:|---:|")
    for r in rows:
        d = r["best_window_distribution"]
        top = max(d, key=d.get) if d else "none"
        print(f"| {r['id']} | {r['label']} | {r['alteration'] or ''} | {r['expected_verdict']} | {top} | "
              f"{d['supports_fully'] if d else 0:.4f} | {r['max_contradicts'] if r['max_contradicts'] is not None else 0:.4f} | "
              f"{r['quote_match']} | {r['unqualified_factual_relay_blocked']} | {len(r['all_window_distributions'])} | "
              f"{r['usage']['requests']} | {r['usage']['cost_usd']:.6f} | {r['wall_seconds']:.2f} |")
    print("\nModels:", dict(Counter(model for r in rows for model in r["model"])))
    print("Blocked:", sum(r["unqualified_factual_relay_blocked"] for r in rows), "/", len(rows))
    print("Quote matches:", dict(Counter(str(r["quote_match"]) for r in rows)))
    print("Requests:", sum(r["usage"]["requests"] for r in rows), "Windows:", sum(len(r["all_window_distributions"]) for r in rows))
    print("Tokens:", sum(r["usage"]["input_tokens"] for r in rows), "input,",
          sum(r["usage"]["output_tokens"] for r in rows), "output")
    print(f"Total cost: ${sum(r['usage']['input_tokens'] for r in rows) * 0.042 / 1_000_000:.9f}")
    print(f"Total wall time: {sum(r['wall_seconds'] for r in rows):.2f}s")
    print("Quote failures:", [r["id"] for r in rows if r["quote_match"] not in ("exact", "normalised")])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--case", choices=[f"v{i:02d}" for i in range(1, 41)])
    run.add_argument("--force", action="store_true")
    run.set_defaults(func=lambda args: asyncio.run(cmd_run(args)))
    summary = sub.add_parser("summary")
    summary.set_defaults(func=cmd_summary)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
