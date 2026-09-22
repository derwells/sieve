"""Enumerate visible requests and bound later dialogue for triage."""

from __future__ import annotations

import re

from .jev import estimate_tokens

REQUEST_THRESHOLD = 0.5  # Provisional until fitted on the triage eval.
WINDOW_TOKENS = 5000
WINDOW_OVERLAP = 1000
VERBS = re.compile(r"^(?:choose|confirm|decide|let me know|should i|do you want|which|tell me|approve|pick)\b", re.I)
SENTENCES = re.compile(r"(?<=[.!?])\s+(?=[A-Z`*])|\n+")
ACCEPTANCE = re.compile(r"\b(?:acceptance|done when|gate|must)\b", re.I)

def enumerate_candidates(events: list[dict]) -> list[dict]:
    """Split visible assistant prose into request spans; ignore quotes and code."""
    found = []
    for event in events:
        if event.get("role") != "assistant" or event.get("kind") != "text":
            continue
        in_fence = False
        paragraphs: list[str] = []
        for line in event.get("text", "").splitlines():
            if line.lstrip().startswith("```"):
                in_fence = not in_fence
                continue
            if in_fence or line.lstrip().startswith(">"):
                continue
            paragraphs.append(line)
        for paragraph in re.split(r"\n\s*\n", "\n".join(paragraphs)):
            context = paragraph.strip()
            if not context:
                continue
            for line in paragraph.splitlines():
                line = re.sub(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)", "", line).strip()
                for sentence in SENTENCES.split(line):
                    span = sentence.strip()
                    if span and (span.endswith("?") or VERBS.match(span)):
                        found.append({"event_id": event["id"], "ts": event.get("ts"), "span_text": span, "context": context})
    return found


def extract_acceptance(contract: dict) -> str:
    texts = [contract.get("first_prompt") or "", *(contract.get("human_amendments") or [])]
    lines = []
    for value in texts:
        if isinstance(value, dict):
            value = value.get("text", "")
        lines.extend(line.strip() for line in str(value).splitlines() if ACCEPTANCE.search(line))
    return "\n".join(lines)


def dialogue_windows(turns: list[dict], limit: int = WINDOW_TOKENS, overlap: int = WINDOW_OVERLAP) -> list[list[dict]]:
    """Sliding windows with overlap, retaining event IDs and chronology."""
    if not turns:
        return []
    if not 0 <= overlap < limit:
        raise ValueError("overlap must be smaller than limit")
    # Split exceptionally long turns, preserving their original event IDs.
    pieces = []
    for turn in turns:
        text = turn.get("text", "")
        width = max(4, limit * 4)
        for start in range(0, len(text), width):
            pieces.append({**turn, "text": text[start:start + width]})
    windows = []
    start = 0
    while start < len(pieces):
        end = start
        size = 0
        while end < len(pieces) and (size + estimate_tokens(pieces[end]["text"]) <= limit or end == start):
            size += estimate_tokens(pieces[end]["text"])
            end += 1
        windows.append(pieces[start:end])
        if end == len(pieces):
            break
        back = end - 1
        shared = 0
        while back > start and shared < overlap:
            shared += estimate_tokens(pieces[back]["text"])
            back -= 1
        start = max(start + 1, back + 1)
    return windows


