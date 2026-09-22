"""Score explicit requests for human input in chronological agent events."""

from __future__ import annotations

import json
from typing import Any

from .client import DEFAULT_MODEL, build_client
from .jev import JevScorer, QuestionSpec, ScoreItem
from .triage_candidates import REQUEST_THRESHOLD, dialogue_windows, enumerate_candidates, extract_acceptance

SPECS = {
    "request": QuestionSpec("items", "Does {ref} request a decision, information, or action from Derick?", "The span asks Derick to provide a decision, information, or action.", "The span does not ask Derick for input."),
    "answered": QuestionSpec("items", "Does the later human dialogue in {ref} supply the requested decision or information for this specific request?", "A later human turn answers this exact request.", "No later human turn answers this exact request."),
    "withdrawn": QuestionSpec("items", "Does a later assistant statement in {ref} withdraw or supersede this request?", "A later assistant statement removes or replaces this request.", "The request remains in force."),
    "blocking": QuestionSpec("items", "Does the latest assistant statement in {ref} say progress is waiting for Derick's response to this request?", "The latest assistant statement explicitly waits for this response.", "The latest assistant statement does not wait for this response."),
    "one_step_left": QuestionSpec("items", "Does the latest progress evidence in {ref} identify exactly one remaining action before the currently stated acceptance is met?", "Exactly one action remains before acceptance.", "Zero or multiple actions remain, or the progress is unclear."),
}


def _item(payload: dict, index: int) -> ScoreItem:
    return ScoreItem(str(index), json.dumps(payload, sort_keys=True, ensure_ascii=False), payload)


async def _judge(scorer: JevScorer, kind: str, payloads: list[dict]) -> tuple[list[float | None], bool]:
    if not payloads:
        return [], False
    # One state per request or window. Shared states blurred scores in the eval.
    run = await scorer.score("", [_item(p, i) for i, p in enumerate(payloads)], SPECS[kind], max_items_per_batch=1)
    return [run.scores.get(str(i)) for i in range(len(payloads))], run.budget_exhausted


def _truncated(events: list[dict]) -> bool:
    return any(e.get("id") == "TRUNCATION_MARKER" or e.get("coverage_truncated") for e in events)


async def _score_thread(events: list[dict], contract: dict, journal_priority: str | None, status: str | None,
                        scorer: JevScorer, request_threshold: float, thread_id: str | None = None) -> dict:
    start_usage = scorer.usage.as_dict()
    candidates = enumerate_candidates(events)
    candidate_scores, exhausted = await _judge(scorer, "request", [
        {"title": contract.get("title", ""), "span": c["span_text"], "context": c["context"]} for c in candidates
    ])
    requests = []
    coverage_truncated = _truncated(events)
    for candidate, probability in zip(candidates, candidate_scores):
        if probability is None or probability < request_threshold:
            continue
        position = next(i for i, e in enumerate(events) if e.get("id") == candidate["event_id"])
        target = {"span": candidate["span_text"], "context": candidate["context"], "event_id": candidate["event_id"], "ts": candidate["ts"]}
        later = events[position + 1:]
        human = [e for e in later if e.get("role") == "human" and e.get("kind") == "text"]
        assistant = [e for e in later if e.get("role") == "assistant" and e.get("kind") == "text"]
        human_windows = dialogue_windows(human)
        assistant_windows = dialogue_windows(assistant)
        answers, used = await _judge(scorer, "answered", [{"target_request": target, "later_dialogue": w} for w in human_windows])
        exhausted |= used
        withdrawals, used = await _judge(scorer, "withdrawn", [{"target_request": target, "later_assistant": w} for w in assistant_windows])
        exhausted |= used
        latest = next((e for e in reversed(events) if e.get("role") == "assistant" and e.get("kind") == "text"), None)
        blocking, used = await _judge(scorer, "blocking", [{"target_request": target, "latest_assistant_text": latest["text"]}]) if latest else ([None], False)
        exhausted |= used
        answered = max((p for p in answers if p is not None), default=None)
        withdrawn = max((p for p in withdrawals if p is not None), default=None)
        answer_index = answers.index(answered) if answered is not None else None
        withdrawal_index = withdrawals.index(withdrawn) if withdrawn is not None else None
        human_seen = {e["id"] for w, p in zip(human_windows, answers) if p is not None for e in w}
        resolution = ("answered" if answered is not None and answered >= request_threshold else
                      "withdrawn" if withdrawn is not None and withdrawn >= request_threshold else
                      "unknown" if (coverage_truncated or (human and answered is None)) else "unanswered")
        # Running status suppresses a blocker only when newer assistant progress follows the request.
        recent_progress = status == "running" and bool(assistant)
        blocked = (blocking[0] is not None and blocking[0] >= request_threshold
                   and resolution == "unanswered" and not recent_progress)
        requests.append({**candidate, "probabilities": {"request": probability, "answered": answered,
                         "withdrawn": withdrawn, "blocking": blocking[0]}, "resolution": resolution,
                         "blocked_on_derick": blocked, "optional": blocking[0] is not None and blocking[0] < request_threshold and answered is None,
                         "evidence_event_ids": {"answered": list(dict.fromkeys(e["id"] for e in human_windows[answer_index])) if answer_index is not None else [],
                                                "withdrawn": list(dict.fromkeys(e["id"] for e in assistant_windows[withdrawal_index])) if withdrawal_index is not None else [],
                                                "blocking": [latest["id"]] if latest else []},
                         "coverage": {"windows": len(human_windows), "human_turns_total": len(human), "human_turns_seen": len(human_seen),
                                      "assistant_windows": len(assistant_windows)}})
    acceptance = extract_acceptance(contract)
    one_step = None
    progress = [e for e in events if e.get("role") == "assistant" and e.get("kind") == "text"][-3:]
    if acceptance and progress:
        scores, used = await _judge(scorer, "one_step_left", [{"acceptance_text": acceptance, "latest_progress": progress}])
        exhausted |= used
        one_step = scores[0]
    blocked_requests = [r for r in requests if r["blocked_on_derick"]]
    one_step_yes = one_step is not None and one_step >= request_threshold
    unresolved = [r for r in requests if r["resolution"] in {"unknown", "unanswered"} and not r["blocked_on_derick"]]
    bucket = ("blocked_on_derick" if blocked_requests else "one_step_left" if one_step_yes else
              "unknown" if (coverage_truncated or exhausted or (acceptance and one_step is None) or unresolved) else "nothing")
    last = events[-1] if events else {}
    end_usage = scorer.usage.as_dict()
    thread_usage = {key: end_usage[key] - start_usage[key] for key in ("tokens", "input_tokens", "output_tokens", "requests", "cache_hits", "cost_usd")}
    return {"thread_id": thread_id, "bucket": bucket,
            "buckets": {"blocked_on_derick": [{"excerpt": r["span_text"], "event_id": r["event_id"], "probabilities": r["probabilities"], "evidence_event_ids": r["evidence_event_ids"]} for r in blocked_requests],
                        "one_step_left": {"probability": one_step, "acceptance_text": acceptance or None, "evidence_event_ids": [e["id"] for e in progress]} if acceptance else "unknown",
                        "unknown": [{"excerpt": r["span_text"], "event_id": r["event_id"], "probabilities": r["probabilities"], "evidence_event_ids": r["evidence_event_ids"]} for r in unresolved],
                        "nothing": []},
            "requests": requests, "coverage": {"truncated": coverage_truncated, "candidates": len(candidates), "requests_scored": len(requests)},
            "snapshot": {"last_event_id": last.get("id"), "last_ts": last.get("ts")},
            "journal_priority": journal_priority, "usage": {**thread_usage, "budget_exhausted": exhausted}}


async def score_thread(events: list[dict], contract: dict, journal_priority: str | None = None,
                       status: str | None = None, *, client, cache, budget: float,
                       request_threshold: float = REQUEST_THRESHOLD) -> dict:
    if budget <= 0 or not 0 <= request_threshold <= 1:
        raise ValueError("budget must be positive and request_threshold in [0, 1]")
    scorer = JevScorer(client, cache=cache, budget_usd=budget)
    return await _score_thread(events, contract, journal_priority, status, scorer, request_threshold)


async def jev_triage_threads(threads: list[dict], budget_usd: float = 0.50,
                             request_threshold: float = REQUEST_THRESHOLD, *, client=None, cache=None) -> dict:
    if budget_usd <= 0 or not 0 <= request_threshold <= 1:
        raise ValueError("budget must be positive and request_threshold in [0, 1]")
    owned = client is None
    if owned:
        client = build_client()
    scorer = JevScorer(client, cache=cache, budget_usd=budget_usd)
    try:
        results = []
        for thread in threads:
            results.append(await _score_thread(thread["events"], thread["contract"], thread.get("journal_priority"),
                                                thread.get("status"), scorer, request_threshold, thread["thread_id"]))
    finally:
        if owned:
            await client.aclose()
    priority = {"blocked_on_derick": 0, "one_step_left": 1, "unknown": 2, "nothing": 3}
    results.sort(key=lambda r: (priority[r["bucket"]], 0 if r["journal_priority"] else 1, r["journal_priority"] or ""))
    return {"threads": results, "usage": {**scorer.usage.as_dict(), "budget_exhausted": any(r["usage"]["budget_exhausted"] for r in results)}}
