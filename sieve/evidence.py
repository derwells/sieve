"""Deterministic quote matching and evidence window selection.

No model runs here. A quote is found by normalised substring match, and the
passages a claim is judged against are picked by lexical overlap, so the same
source and claim always produce the same windows and the same offsets.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

#: Largest evidence window handed to the model, in characters.
MAX_WINDOW_CHARS = 1500
#: How many lexical windows a claim gets when its quote did not match.
LEXICAL_WINDOWS = 3

#: Curly quotes and dashes folded to their ASCII forms before matching.
FOLD = str.maketrans(
    {
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "‚": "'",
        "„": '"',
        "′": "'",
        "″": '"',
        "–": "-",
        "—": "-",
        "−": "-",
        " ": " ",
    }
)

_TOKEN = re.compile(r"[A-Za-z0-9]+(?:[.\-_/][A-Za-z0-9]+)*")
#: Words too common to discriminate between passages of the same document.
STOPWORDS = frozenset(
    """
    a an the and or but if then than that this these those of in on at to for from by with
    as is are was were be been being it its it's they them their there here we you your our
    not no can could should would may might will shall do does did done has have had
    """.split()
)


def fold(text: str) -> str:
    """Unify quote and dash characters without changing length."""
    return text.translate(FOLD)


def normalise(text: str) -> str:
    """Folded text with every run of whitespace collapsed to one space."""
    return re.sub(r"\s+", " ", fold(text)).strip()


def _normalised_with_map(text: str) -> tuple[str, list[int]]:
    """Normalised text plus, per normalised character, its index in the original."""
    folded = fold(text)
    out: list[str] = []
    offsets: list[int] = []
    in_space = True  # leading whitespace is dropped, as in normalise()
    for index, char in enumerate(folded):
        if char.isspace():
            if in_space:
                continue
            out.append(" ")
            offsets.append(index)
            in_space = True
            continue
        out.append(char)
        offsets.append(index)
        in_space = False
    while out and out[-1] == " ":
        out.pop()
        offsets.pop()
    return "".join(out), offsets


@dataclass(frozen=True)
class QuoteMatch:
    """Where a quote was found in the source, and how literally."""

    kind: str
    """`exact`, `normalised`, or `not_found`."""
    start: int = -1
    end: int = -1

    @property
    def found(self) -> bool:
        return self.kind != "not_found"


def find_quote(text: str, quote: str) -> QuoteMatch:
    """Locate `quote` in `text`, exactly first, then normalised, then case-folded."""
    if not quote or not quote.strip():
        return QuoteMatch("not_found")
    literal = text.find(quote)
    if literal >= 0:
        return QuoteMatch("exact", literal, literal + len(quote))

    haystack, offsets = _normalised_with_map(text)
    needle = normalise(quote)
    if not needle:
        return QuoteMatch("not_found")
    position = haystack.find(needle)
    if position < 0:
        position = haystack.lower().find(needle.lower())
    if position < 0:
        return QuoteMatch("not_found")
    start = offsets[position]
    end = offsets[position + len(needle) - 1] + 1
    return QuoteMatch("normalised", start, end)


@dataclass(frozen=True)
class Paragraph:
    start: int
    end: int
    text: str


def split_paragraphs(text: str) -> list[Paragraph]:
    """Blank-line separated blocks, with their offsets in `text`."""
    paragraphs: list[Paragraph] = []
    for match in re.finditer(r"[^\n](?:.*?)(?=\n\s*\n|\Z)", text, re.DOTALL):
        block = match.group(0)
        stripped = block.strip()
        if not stripped:
            continue
        lead = len(block) - len(block.lstrip())
        start = match.start() + lead
        paragraphs.append(Paragraph(start=start, end=start + len(stripped), text=stripped))
    return paragraphs


def tokenise(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(fold(text))]


def content_tokens(text: str) -> list[str]:
    return [t for t in tokenise(text) if t not in STOPWORDS and len(t) > 1]


def bm25_scores(claim: str, paragraphs: list[Paragraph], k1: float = 1.5, b: float = 0.75) -> list[float]:
    """BM25 over the source's own paragraphs, so scores are comparable within a source."""
    if not paragraphs:
        return []
    docs = [content_tokens(p.text) for p in paragraphs]
    lengths = [len(d) or 1 for d in docs]
    average = sum(lengths) / len(lengths)
    total = len(docs)
    scores = [0.0] * total
    for term in set(content_tokens(claim)):
        containing = sum(1 for d in docs if term in d)
        if containing == 0:
            continue
        idf = math.log(1 + (total - containing + 0.5) / (containing + 0.5))
        for index, doc in enumerate(docs):
            frequency = doc.count(term)
            if not frequency:
                continue
            norm = frequency * (k1 + 1) / (frequency + k1 * (1 - b + b * lengths[index] / average))
            scores[index] += idf * norm
    return scores


@dataclass(frozen=True)
class Window:
    """A passage of a source, with the offsets it came from."""

    start: int
    end: int
    text: str
    reason: str
    """`quote` when the window holds a matched quote, `lexical` otherwise."""


def _clip(text: str, start: int, end: int, focus: tuple[int, int] | None, limit: int) -> Window:
    """Cut [start, end) down to `limit` characters, keeping `focus` inside it."""
    if end - start <= limit:
        return Window(start, end, text[start:end], "")
    if focus is None:
        return Window(start, start + limit, text[start : start + limit], "")
    centre = (focus[0] + focus[1]) // 2
    left = max(start, min(centre - limit // 2, end - limit))
    return Window(left, left + limit, text[left : left + limit], "")


def windows_for(
    text: str,
    claim: str,
    quote: QuoteMatch | None = None,
    limit: int = MAX_WINDOW_CHARS,
    count: int = LEXICAL_WINDOWS,
) -> list[Window]:
    """The passages a claim is judged against: the quote's neighbourhood, or the best matches."""
    paragraphs = split_paragraphs(text)
    if not paragraphs:
        return []

    if quote is not None and quote.found:
        index = next(
            (i for i, p in enumerate(paragraphs) if p.start <= quote.start < p.end),
            None,
        )
        if index is None:
            index = min(
                range(len(paragraphs)),
                key=lambda i: abs(paragraphs[i].start - quote.start),
            )
        start = paragraphs[max(0, index - 1)].start
        end = paragraphs[min(len(paragraphs) - 1, index + 1)].end
        clipped = _clip(text, start, end, (quote.start, quote.end), limit)
        return [Window(clipped.start, clipped.end, clipped.text, "quote")]

    scores = bm25_scores(claim, paragraphs)
    ranked = sorted(range(len(paragraphs)), key=lambda i: (-scores[i], i))
    chosen = ranked[:count]
    windows: list[Window] = []
    seen: set[tuple[int, int]] = set()
    for index in sorted(chosen):
        start = paragraphs[index].start
        end = paragraphs[min(len(paragraphs) - 1, index + 1)].end
        clipped = _clip(text, start, end, (paragraphs[index].start, paragraphs[index].end), limit)
        key = (clipped.start, clipped.end)
        if key in seen:
            continue
        seen.add(key)
        windows.append(Window(clipped.start, clipped.end, clipped.text, "lexical"))
    return windows
