"""Turn a Markdown report into claim records, in code and without a model.

The structure a writer already put in the document carries the context a claim
needs: the headings above it, the list item it hangs off, the header row of its
table. Every claim extracted here is marked `extraction_uncertain`, because the
split into claims is a guess about prose, not something the caller declared.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

#: Markdown link, ignoring an optional title.
LINK = re.compile(r"\[([^\]]*)\]\(\s*<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)")
#: A bare URL not already inside a markdown link.
BARE_URL = re.compile(r"(?<![(\]<])\bhttps?://[^\s<>()\[\]\"']+")
#: Backticked text, which may be a file path.
BACKTICKED = re.compile(r"`([^`\n]+)`")
#: A list item: indent, bullet or number, then the text.
LIST_ITEM = re.compile(r"^(\s*)(?:[-*+]|\d+[.)])\s+(.*)$")
#: An ATX heading.
HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*$")
#: A table separator row, which marks the line above it as the header.
TABLE_RULE = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")
#: A fenced code block boundary.
FENCE = re.compile(r"^\s*(```|~~~)")

#: File extensions a backticked token has to carry to read as a path.
PATH_SUFFIXES = frozenset(
    """
    .py .js .ts .tsx .jsx .rs .go .rb .java .kt .c .h .cc .cpp .sh .bash .zsh .sql
    .md .txt .rst .json .toml .yaml .yml .ini .cfg .csv .tsv .lock .html .css
    """.split()
)
#: Abbreviations that end in a full stop without ending a sentence.
ABBREVIATIONS = frozenset(
    """
    e.g. i.e. etc. vs. cf. al. fig. no. approx. dr. mr. mrs. ms. prof. st. jan. feb. mar.
    apr. jun. jul. aug. sep. sept. oct. nov. dec.
    """.split()
)

SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])[\"')\]]*\s+")


def looks_like_path(token: str) -> bool:
    """Whether a backticked token reads as a file path rather than an identifier."""
    token = token.strip()
    if not token or " " in token or token.startswith("$"):
        return False
    if token.startswith(("./", "../", "/", "~/")):
        return True
    suffix = token[token.rfind(".") :].lower() if "." in token else ""
    return "/" in token and suffix in PATH_SUFFIXES or (suffix in PATH_SUFFIXES and "/" not in token)


def strip_trailing_punctuation(url: str) -> str:
    return url.rstrip(".,;:!?)’'\"")


def find_citations(block: str) -> list[dict]:
    """Every locator in one block: markdown links, bare URLs, backticked paths."""
    found: list[str] = []
    for _, target in LINK.findall(block):
        target = target.strip()
        if target and not target.startswith("#"):
            found.append(strip_trailing_punctuation(target))
    for url in BARE_URL.findall(block):
        found.append(strip_trailing_punctuation(url))
    for token in BACKTICKED.findall(block):
        if looks_like_path(token):
            found.append(token.strip())
    seen: set[str] = set()
    citations: list[dict] = []
    for locator in found:
        if locator in seen:
            continue
        seen.add(locator)
        citations.append({"locator": locator})
    return citations


def clean(text: str) -> str:
    """The readable sentence: link text without its target, no emphasis markers."""
    text = LINK.sub(lambda m: m.group(1) or m.group(2), text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"(?<!\w)[*_]([^*_\n]+)[*_](?!\w)", r"\1", text)
    return " ".join(text.split())


def split_sentences(text: str) -> list[str]:
    """Sentences of a paragraph, keeping abbreviations and decimals intact."""
    text = text.strip()
    if not text:
        return []
    sentences: list[str] = []
    current = ""
    for piece in SENTENCE_BOUNDARY.split(text):
        current = f"{current} {piece}".strip() if current else piece
        tail = current.rsplit(" ", 1)[-1].lower()
        if tail in ABBREVIATIONS or re.search(r"\b[A-Za-z]\.$", current):
            continue
        sentences.append(current)
        current = ""
    if current:
        sentences.append(current)
    return [s for s in sentences if s]


@dataclass
class _Item:
    indent: int
    text: str
    citations: list[dict] = field(default_factory=list)


def _context(headings: list[str], parents: list[str]) -> str:
    return " > ".join([*headings, *parents])


def _table_cells(line: str) -> list[str]:
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [clean(cell.strip()) for cell in stripped.split("|")]


def extract_claims(report: str) -> list[dict]:
    """Claim records for every sentence, bullet, and table row of a Markdown report."""
    lines = report.replace("\r\n", "\n").split("\n")
    claims: list[dict] = []
    headings: list[str] = []
    stack: list[_Item] = []
    paragraph: list[str] = []
    table_header: list[str] | None = None
    table_rows: list[str] = []
    fenced = False

    def flush_paragraph() -> None:
        nonlocal paragraph
        if not paragraph:
            return
        block = "\n".join(paragraph)
        paragraph = []
        citations = find_citations(block) or _inherited(stack)
        parents = [item.text for item in stack]
        for sentence in split_sentences(clean(block)):
            claims.append(_record(sentence, _context(headings, parents), citations))

    def flush_table() -> None:
        nonlocal table_header, table_rows
        if table_header is not None:
            heading_line = " | ".join(table_header)
            for row in table_rows:
                cells = _table_cells(row)
                citations = find_citations(row) or _inherited(stack)
                pairs = [
                    f"{name}: {value}"
                    for name, value in zip(table_header, cells)
                    if value
                ]
                if pairs:
                    context = _context(headings, [*[i.text for i in stack], f"table columns: {heading_line}"])
                    claims.append(_record("; ".join(pairs), context, citations))
        table_header, table_rows = None, []

    index = 0
    while index < len(lines):
        line = lines[index]
        if FENCE.match(line):
            flush_paragraph()
            flush_table()
            fenced = not fenced
            index += 1
            continue
        if fenced:
            index += 1
            continue

        if not line.strip():
            flush_paragraph()
            flush_table()
            stack.clear()
            index += 1
            continue

        heading = HEADING.match(line)
        if heading:
            flush_paragraph()
            flush_table()
            stack.clear()
            level = len(heading.group(1))
            del headings[level - 1 :]
            headings.append(clean(heading.group(2)))
            index += 1
            continue

        if "|" in line and index + 1 < len(lines) and TABLE_RULE.match(lines[index + 1]):
            flush_paragraph()
            flush_table()
            table_header = _table_cells(line)
            index += 2
            continue
        if table_header is not None:
            if "|" in line:
                table_rows.append(line)
                index += 1
                continue
            flush_table()

        item = LIST_ITEM.match(line)
        if item:
            flush_paragraph()
            indent = len(item.group(1).expandtabs(4))
            body = [item.group(2)]
            index += 1
            while index < len(lines):
                nxt = lines[index]
                if not nxt.strip() or LIST_ITEM.match(nxt) or HEADING.match(nxt) or FENCE.match(nxt):
                    break
                if len(nxt) - len(nxt.lstrip()) <= indent:
                    break
                body.append(nxt.strip())
                index += 1
            block = "\n".join(body)
            while stack and stack[-1].indent >= indent:
                stack.pop()
            citations = find_citations(block) or _inherited(stack)
            text = clean(block)
            parents = [entry.text for entry in stack]
            for sentence in split_sentences(text) or [text]:
                if sentence:
                    claims.append(_record(sentence, _context(headings, parents), citations))
            stack.append(_Item(indent=indent, text=text, citations=find_citations(block)))
            continue

        paragraph.append(line)
        index += 1

    flush_paragraph()
    flush_table()
    return claims


def _inherited(stack: list[_Item]) -> list[dict]:
    """The nearest enclosing list item's citations, for a block that has none."""
    for item in reversed(stack):
        if item.citations:
            return item.citations
    return []


def _record(claim: str, claim_context: str, citations: list[dict]) -> dict:
    return {
        "claim": claim,
        "claim_context": claim_context,
        "kind": "fact",
        "citations": [dict(c) for c in citations],
        "extraction_uncertain": True,
    }
