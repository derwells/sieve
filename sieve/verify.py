"""Check cited claims against bounded source passages with one Choice per passage."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from .client import DEFAULT_MODEL, build_client
from .evidence import find_quote, windows_for
from .fetch import fetch_source
from .jev import ChoiceSpec, DEFAULT_CONCURRENCY, JevScorer, ScoreItem
from .report import extract_claims

OPTIONS = {
    "supports_fully": "The evidence establishes the entire claim, including its quantities, scope, conditions, and dates.",
    "partially_supports": "The evidence establishes some but not all of the claim, or supports a narrower claim.",
    "contradicts": "The evidence conflicts with a material part of the claim.",
    "does_not_address": "The evidence does not establish or refute the claim.",
}
QUESTION = "What does this evidence establish about the entire claim, preserving its quantities, scope, conditions and dates?"
CHOICE_SPEC = ChoiceSpec(
    state_field="pairs",
    instructions=QUESTION + " Use only the claim, claim_context, evidence_text, source_locator, and source_version in {ref}.",
    criteria=OPTIONS,
)
MAX_PAIRS_PER_BATCH = 40


def _validated_records(records: list[dict] | None, report: str | None) -> list[dict]:
    if (records is None) == (report is None):
        raise ValueError("provide exactly one of records or report")
    if report is not None:
        if not isinstance(report, str) or not report.strip():
            raise ValueError("report must be nonempty Markdown")
        return extract_claims(report)
    if not isinstance(records, list):
        raise ValueError("records must be a list")
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("claim"), str) or not record["claim"].strip():
            raise ValueError("each record needs a nonempty claim")
        if record.get("kind", "fact") not in {"fact", "recommendation"}:
            raise ValueError("kind must be fact or recommendation")
        if not isinstance(record.get("citations", []), list):
            raise ValueError("citations must be a list")
        for citation in record.get("citations", []):
            if not isinstance(citation, dict) or not isinstance(citation.get("locator"), str):
                raise ValueError("each citation needs a locator string")
            if citation.get("quote") is not None and not isinstance(citation["quote"], str):
                raise ValueError("citation quote must be a string")
        if not isinstance(record.get("premises", []), list) or any(not isinstance(p, str) or not p.strip() for p in record.get("premises", [])):
            raise ValueError("premises must be nonempty strings")
    return records


def _with_premises(records: list[dict]) -> list[tuple[dict, list[dict]]]:
    return [
        (
            record,
            [
                {"claim": premise, "claim_context": record.get("claim_context", ""), "kind": "fact", "citations": record.get("citations", [])}
                for premise in record.get("premises", [])
            ],
        )
        for record in records
    ]


async def jev_verify(
    records: list[dict] | None = None,
    report: str | None = None,
    base_path: str | None = None,
    budget_usd: float = 0.50,
    support_threshold: float = 0.6,
    contradict_threshold: float = 0.5,
    *,
    client=None,
    http_client=None,
    cache=None,
    model: str = DEFAULT_MODEL,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> dict[str, Any]:
    """Verify claims. Thresholds are provisional until the verify eval fits them."""
    if budget_usd <= 0 or not 0 <= support_threshold <= 1 or not 0 <= contradict_threshold <= 1:
        raise ValueError("budget must be positive and thresholds must be in [0, 1]")
    groups = _with_premises(_validated_records(records, report))
    flat = [item for record, premises in groups for item in [record, *premises]]
    locators = list(dict.fromkeys(c["locator"] for record in flat for c in record.get("citations", [])))
    fetched = await asyncio.gather(*(fetch_source(locator, base_path=base_path, client=http_client) for locator in locators))
    sources = dict(zip(locators, fetched))
    items: list[ScoreItem] = []
    metadata: dict[str, tuple[int, dict]] = {}
    results: list[dict] = []
    for record_index, record in enumerate(flat):
        citations_checked: list[dict] = []
        flags: list[dict] = []
        row = {
            "claim": record["claim"],
            "kind": record.get("kind", "fact"),
            "exempt": record.get("kind", "fact") == "recommendation",
            "verdict": None,
            "best_evidence": None,
            "max_contradicts": None,
            "quote_match": None,
            "flags": flags,
            "citations_checked": citations_checked,
            "evidence": [],
        }
        if record.get("extraction_uncertain"):
            row["extraction_uncertain"] = True
        if record.get("claim_context"):
            row["claim_context"] = record["claim_context"]
        results.append(row)
        for citation in record.get("citations", []):
            locator = citation["locator"]
            source = sources[locator]
            checked = {"locator": locator, "source_version": source.source_version or None, "quote_match": None, "flags": []}
            citations_checked.append(checked)
            if not source.ok:
                flag = {"flag": "fetch_failed", "locator": locator, "reason": source.error}
                flags.append(flag)
                checked["flags"].append(flag)
                continue
            quote = citation.get("quote")
            match = find_quote(source.text, quote) if quote else None
            if match is not None:
                checked["quote_match"] = match.kind
                if match.kind == "not_found":
                    flag = {"flag": "quote_not_found", "locator": locator}
                    flags.append(flag)
                    checked["flags"].append(flag)
            for window in windows_for(source.text, record["claim"], match):
                state = {
                    "claim": record["claim"],
                    "claim_context": record.get("claim_context", ""),
                    "evidence_text": window.text,
                    "source_locator": locator,
                    "source_version": source.source_version,
                }
                item_id = str(len(items))
                items.append(ScoreItem(item_id, json.dumps(state, sort_keys=True, ensure_ascii=False), state))
                metadata[item_id] = (record_index, {
                    "locator": locator,
                    "source_version": source.source_version,
                    "start": window.start,
                    "end": window.end,
                    "excerpt": window.text,
                })
        matches = {c["locator"]: c["quote_match"] for c in citations_checked if c["quote_match"] is not None}
        row["quote_match"] = next(iter(matches.values())) if len(matches) == 1 else (matches or None)
        if not citations_checked:
            flags.append({"flag": "no_citations"})

    owned_client = client is None and bool(items)
    if owned_client:
        client = build_client(model=model)
    scorer = JevScorer(client, model=model, cache=cache, budget_usd=budget_usd, concurrency=concurrency)
    try:
        run = await scorer.score("", items, CHOICE_SPEC, max_items_per_batch=MAX_PAIRS_PER_BATCH) if items else None
    finally:
        if owned_client:
            await client.aclose()
    for item in items:
        if run is None or item.id not in run.scores:
            continue
        index, evidence = metadata[item.id]
        probabilities = run.scores[item.id]
        entry = {**evidence, "probabilities": probabilities}
        results[index]["evidence"].append(entry)
    for row_index, row in enumerate(results):
        evidence = row["evidence"]
        if evidence:
            best = max(evidence, key=lambda e: e["probabilities"]["supports_fully"])
            row["best_evidence"] = {key: value for key, value in best.items() if key != "probabilities"}
            row["max_contradicts"] = max(e["probabilities"]["contradicts"] for e in evidence)
            if not row["exempt"]:
                row["verdict"] = best["probabilities"]
        if not row["exempt"]:
            if not evidence:
                row["flags"].append({"flag": "no_evidence"})
            elif row["verdict"]["supports_fully"] < support_threshold:
                row["flags"].append({"flag": "support_below_threshold"})
            if row["max_contradicts"] is not None and row["max_contradicts"] >= contradict_threshold:
                row["flags"].append({"flag": "contradiction_above_threshold"})
        if run and run.budget_exhausted and len(evidence) < sum(1 for i in metadata.values() if i[0] == row_index):
            row["flags"].append({"flag": "budget_exhausted"})

    output: list[dict] = []
    index = 0
    factual = []
    for _, premises in groups:
        primary = results[index]
        index += 1
        primary["premises"] = results[index:index + len(premises)] if premises else []
        factual.extend(primary["premises"])
        if not primary["exempt"]:
            factual.append(primary)
        index += len(premises)
        output.append(primary)
    counts = {key: 0 for key in OPTIONS}
    for row in factual:
        if row["verdict"]:
            counts[max(row["verdict"], key=row["verdict"].get)] += 1
    blocked = any(
        not any(
            sources[c["locator"]].ok and bool(sources[c["locator"]].text.strip())
            for c in row["citations_checked"]
        )
        or (row["max_contradicts"] is not None and row["max_contradicts"] >= contradict_threshold)
        for row in factual
    )
    return {
        "results": output,
        "counts": counts,
        "flagged": [{"claim": r["claim"], "flags": r["flags"]} for r in results if r["flags"]],
        "unqualified_factual_relay_blocked": blocked,
        "support_threshold": support_threshold,
        "contradict_threshold": contradict_threshold,
        "usage": {**scorer.usage.as_dict(), "budget_exhausted": bool(run and run.budget_exhausted)},
    }
