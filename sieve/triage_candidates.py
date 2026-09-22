"""Enumerate visible requests and bound later dialogue for triage."""

from __future__ import annotations

import re

from .jev import estimate_tokens

REQUEST_THRESHOLD = 0.5  # Provisional until fitted on the triage eval.
WINDOW_TOKENS = 5000
WINDOW_OVERLAP = 1000
VERBS = re.compile(r"^(?:choose|confirm|decide|let me know|should i|do you want|which|tell me|approve|pick)\b", re.I)
SENTENCES = re.compile(r"(?<=[.!?])\s+(?=[A-Z`*])")
LIST_ITEM = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)")
READER = re.compile(r"\b(?:you|your)\b", re.I)
REQUEST_CUE = re.compile(r"\b(?:can|could|would|should|must|need\w*|want|approv\w*|confirm\w*)\b", re.I)
REQUEST_HEADING = re.compile(
    r"\b(?:needs?\s+you|(?:decisions?|actions?|confirmations?|approvals?|input)\b.*"
    r"\b(?:needed|required|from you)|(?:things?|decisions?|actions?)\s+from you)\b", re.I)
ACCEPTANCE = re.compile(r"\b(?:acceptance|done when|gate|must)\b", re.I)


def _plain(text: str) -> str:
    return re.sub(r"[*_`]+", "", text)


def _prose_units(paragraph: str) -> list[tuple[str, bool]]:
    """Join wrapped prose, keeping list items and table cells separate."""
    units: list[tuple[str, bool]] = []
    for line in paragraph.splitlines():
        item = LIST_ITEM.match(line)
        if line.strip().startswith("|"):
            units.extend((cell.strip(), False) for cell in line.strip().strip("|").split("|")
                         if cell.strip() and not re.fullmatch(r"[\s:|-]+", cell))
        elif item:
            units.append((line[item.end():].strip(), True))
        elif units and not units[-1][0].endswith(":"):
            text, is_item = units[-1]
            units[-1] = (text + " " + line.strip(), is_item)
        else:
            units.append((line.strip(), False))
    return units


def enumerate_candidates(events: list[dict]) -> list[dict]:
    """Split visible assistant prose into request spans; ignore quotes and code."""
    found = []
    for event in events:
        if event.get("role") != "assistant" or event.get("kind") != "text":
            continue
        fence = None
        paragraphs: list[str] = []
        for line in event.get("text", "").splitlines():
            marker = re.match(r"^\s*(`{3,}|~{3,})", line)
            if marker:
                if fence is None:
                    fence = marker[1]
                elif marker[1][0] == fence[0] and len(marker[1]) >= len(fence):
                    fence = None
                paragraphs.append("")
                continue
            if fence or line.lstrip().startswith(">"):
                paragraphs.append("")
                continue
            paragraphs.append(line)
        heading = ""
        for paragraph in re.split(r"\n\s*\n", "\n".join(paragraphs)):
            context = paragraph.strip()
            if not context:
                continue
            units = _prose_units(paragraph)
            if not units:
                continue
            if not units[0][1]:
                heading = context if len(units) == 1 and REQUEST_HEADING.search(_plain(context)) else ""
            directed = bool(REQUEST_HEADING.search(_plain(context)) or heading)
            if heading and context != heading:
                context = heading + "\n" + context
            for line, is_item in units:
                if not is_item and REQUEST_HEADING.fullmatch(_plain(line).rstrip(":").strip()):
                    continue
                # A reader-directed list can contain noun phrases, not just verbs.
                sentences = [line] if directed and is_item else SENTENCES.split(line)
                for sentence in sentences:
                    span = sentence.strip()
                    plain = _plain(span)
                    embedded = any(VERBS.match(part.strip()) for part in plain.split(":")[1:])
                    if span and ("?" in plain or VERBS.match(plain) or embedded
                                 or (READER.search(plain) and REQUEST_CUE.search(plain))
                                 or (directed and is_item)):
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
