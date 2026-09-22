"""Enumerate a repository into scoreable units: files, functions, or line chunks.

The gitignore-aware walk, the "nested functions stay inside their enclosing
function" rule, and the fixed-size chunking are adapted from jgrep:

    https://github.com/keltokhy/jgrep
    MIT License, Copyright (c) 2026 Khaled Eltokhy

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
    IN THE SOFTWARE.

This module never calls a model.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from pathspec import GitIgnoreSpec

from .errors import EnumerationError

#: Opening lines of a file shown to Jev when scoring the file itself.
FILE_PREVIEW_LINES = 40
#: A file no longer than this multiple of the preview is shown whole. Cutting a
#: short file a few lines from its end throws away signal for almost no saving.
PREVIEW_WHOLE_FILE_MULTIPLE = 2
#: Character budget for a whole file preview: the opening lines plus the outline of
#: the rest. The 40-line head alone runs ~1.5-2.5 kB, so this is roughly 2-3x the
#: old preview and still ~1.5k tokens, leaving 8 units plus their questions well
#: under the 30k state limit.
FILE_PREVIEW_CHARS = 6_000
#: Most entries an outline carries; longer files are subsampled evenly, never cut
#: off at the top, so the outline still spans the whole file.
MAX_OUTLINE_ENTRIES = 60
#: An outline entry is one line and never longer than this.
MAX_OUTLINE_ENTRY_CHARS = 160
#: Suffixes whose outline is the file's headings.
MARKDOWN_SUFFIXES = {".md", ".markdown", ".mdx"}
#: ATX heading, and a fenced code block delimiter that suppresses headings inside it.
MARKDOWN_HEADING = re.compile(r"^ {0,3}#{1,6}\s+\S")
MARKDOWN_FENCE = re.compile(r"^\s*(```|~~~)")
#: Lines per unit when a file has no tree-sitter grammar, or no functions in it.
CHUNK_LINES = 60
#: A single unit is never allowed to dominate a batch.
MAX_UNIT_CHARS = 12_000
#: Bytes inspected when deciding whether a file is binary.
BINARY_SNIFF_BYTES = 8192
#: Directories never worth walking even when a repo forgets to ignore them.
ALWAYS_SKIP = {".git", ".hg", ".svn", ".sieve-cache"}

#: Extension to tree-sitter-language-pack grammar name.
LANGUAGE_BY_SUFFIX = {
    ".py": "python", ".pyi": "python",
    ".js": "javascript", ".mjs": "javascript", ".cjs": "javascript", ".jsx": "javascript",
    ".ts": "typescript", ".mts": "typescript", ".cts": "typescript", ".tsx": "tsx",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".c": "c", ".h": "c",
    ".cc": "cpp", ".cpp": "cpp", ".cxx": "cpp", ".hpp": "cpp", ".hh": "cpp",
    ".cs": "csharp",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".kt": "kotlin", ".kts": "kotlin",
    ".scala": "scala",
    ".sh": "bash", ".bash": "bash", ".zsh": "bash",
    ".lua": "lua",
    ".ex": "elixir", ".exs": "elixir",
    ".zig": "zig",
    ".hs": "haskell",
    ".ml": "ocaml",
    ".pl": "perl", ".pm": "perl",
    ".r": "r", ".R": "r",
    ".jl": "julia",
    ".dart": "dart",
    ".sql": "sql",
}

#: Tree-sitter node types that name a function-like unit, across grammars. A match
#: is never descended into, so a nested function stays inside its enclosing one.
FUNCTION_NODE_TYPES = {
    "function_definition",
    "function_declaration",
    "function_item",
    "method_definition",
    "method_declaration",
    "constructor_declaration",
    "class_method",
    "singleton_method",
    "method",
    "func_literal",
    "subroutine",
    "fun",
}

#: Tree-sitter node types worth naming in a file's outline, beyond the function-like
#: ones above: types, classes, and top-level bindings, across grammars. Unknown types
#: are ignored, so a grammar contributing none falls back to sampled lines.
OUTLINE_NODE_TYPES = FUNCTION_NODE_TYPES | {
    "class_definition", "class_declaration", "abstract_class_declaration", "class",
    "class_specifier", "struct_specifier", "enum_specifier", "union_specifier",
    "interface_declaration", "type_alias_declaration", "enum_declaration",
    "record_declaration", "namespace_declaration", "namespace_definition",
    "type_definition", "type_declaration", "module", "internal_module",
    "struct_item", "enum_item", "trait_item", "impl_item", "type_item",
    "const_item", "static_item", "mod_item", "macro_definition",
    "lexical_declaration", "variable_declaration", "var_declaration",
    "const_declaration", "decorated_definition", "export_statement",
    "assignment", "data_type_declaration", "defmodule",
}


@dataclass(frozen=True)
class Unit:
    """One scoreable piece of a repository."""

    path: str
    """Repo-relative POSIX path."""
    line_start: int
    line_end: int
    kind: str
    """"file", "function", or "chunk"."""
    text: str
    """The unit's source text, exactly as Jev is shown it."""
    symbol: str | None = None
    size_lines: int | None = None
    """Lines in the whole file the unit came from, so Jev knows what it is not seeing."""


def _load_rules(directory: Path, errors: list[str]) -> list[tuple[Path, GitIgnoreSpec]]:
    rules = []
    for name in (".gitignore", ".ignore"):
        path = directory / name
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            continue
        except (OSError, UnicodeError) as e:
            errors.append(f"{path}: {e}")
            continue
        rules.append((directory, GitIgnoreSpec.from_lines(lines)))
    return rules


def _ignored(path: Path, is_dir: bool, rules) -> bool:
    """Last matching spec wins, so a later negation can re-include a path."""
    result = False
    for base, spec in rules:
        relative = path.relative_to(base).as_posix() + ("/" if is_dir else "")
        match = spec.check_file(relative).include
        if match is not None:
            result = match
    return result


def walk_repo(root: Path) -> tuple[list[str], list[str]]:
    """Repo-relative paths of every non-ignored, non-binary text file under `root`.

    Nested `.gitignore` and `.ignore` files apply from the directory they sit in
    down, and when `root` is inside a repository its ancestors' ignore files
    apply too. Symlinks are never followed.
    """
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise EnumerationError(f"{root}: not a directory")
    errors: list[str] = []

    ancestors: list[Path] = []
    if not (root / ".git").exists():
        for parent in root.parents:
            ancestors.append(parent)
            if (parent / ".git").exists():
                break
        else:
            ancestors = []
    rules = [rule for parent in reversed(ancestors) for rule in _load_rules(parent, errors)]

    found: list[str] = []

    def walk(directory: Path, inherited) -> None:
        active = inherited + _load_rules(directory, errors)
        try:
            with os.scandir(directory) as entries:
                children = sorted(entries, key=lambda entry: entry.name)
        except OSError as e:
            errors.append(f"{directory}: {e}")
            return
        for entry in children:
            path = directory / entry.name
            try:
                if entry.is_symlink() or entry.name in ALWAYS_SKIP:
                    continue
                is_dir = entry.is_dir(follow_symlinks=False)
                if _ignored(path, is_dir, active):
                    continue
                if is_dir:
                    walk(path, active)
                elif entry.is_file(follow_symlinks=False) and not _is_binary(path):
                    found.append(path.relative_to(root).as_posix())
            except OSError as e:
                errors.append(f"{path}: {e}")

    walk(root, rules)
    return found, errors


def _is_binary(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            return b"\0" in f.read(BINARY_SNIFF_BYTES)
    except OSError:
        return True


def read_text(path: Path) -> str | None:
    """File contents, or None when the file cannot be read as text."""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except (OSError, UnicodeError):
        return None


def _clip(text: str, limit: int = MAX_UNIT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n... [unit truncated by sieve]"


def _thin(entries: list[tuple[int, str]], limit: int) -> list[tuple[int, str]]:
    """At most `limit` entries, spaced evenly so they still span the whole file."""
    if len(entries) <= limit:
        return entries
    if limit <= 1:
        return entries[:limit]
    step = (len(entries) - 1) / (limit - 1)
    return [entries[round(i * step)] for i in range(limit)]


def _markdown_outline(lines: list[str]) -> list[tuple[int, str]]:
    entries: list[tuple[int, str]] = []
    fenced = False
    for number, line in enumerate(lines, start=1):
        if MARKDOWN_FENCE.match(line):
            fenced = not fenced
            continue
        if not fenced and MARKDOWN_HEADING.match(line):
            entries.append((number, line.strip()))
    return entries


def _code_outline(text: str, language: str, lines: list[str]) -> list[tuple[int, str]]:
    """One entry per top-level declaration: its first real line, as its signature."""
    try:
        from tree_sitter_language_pack import get_parser
    except ImportError:  # pragma: no cover
        return []
    try:
        parser = get_parser(language)
    except Exception:  # pragma: no cover
        return []
    root = parser.parse(text.encode("utf-8")).root_node
    entries: list[tuple[int, str]] = []
    for child in root.children:
        if child.type not in OUTLINE_NODE_TYPES:
            continue
        last = min(child.end_point[0], len(lines) - 1)
        for index in range(child.start_point[0], last + 1):
            stripped = lines[index].strip()
            if stripped and not stripped.startswith("@"):  # skip decorator lines
                entries.append((index + 1, stripped))
                break
    return entries


def _sampled_outline(lines: list[str], after: int, limit: int) -> list[tuple[int, str]]:
    """Evenly spaced non-blank lines from past the head, for text sieve cannot parse."""
    indexes = [i for i in range(after, len(lines)) if lines[i].strip()]
    return _thin([(i + 1, lines[i].strip()) for i in indexes], limit)


def file_outline(
    relative: str,
    text: str,
    lines: list[str],
    after: int = 0,
    limit: int = MAX_OUTLINE_ENTRIES,
) -> list[tuple[int, str]]:
    """(line number, text) of what the whole file contains, past its opening lines.

    Markdown gets its headings, a file with a tree-sitter grammar gets its
    top-level declarations, and anything else gets evenly spaced sample lines.
    """
    entries: list[tuple[int, str]] = []
    if Path(relative).suffix.lower() in MARKDOWN_SUFFIXES:
        entries = _markdown_outline(lines)
    else:
        language = language_for(relative)
        if language is not None:
            entries = _code_outline(text, language, lines)
    if not entries:
        return _sampled_outline(lines, after, limit)
    return _thin(entries, limit)


def render_outline(entries: list[tuple[int, str]], size_lines: int) -> str:
    header = f"... [sieve outline of the whole file, {size_lines} lines total]"
    rows = [f"L{number}: {body[:MAX_OUTLINE_ENTRY_CHARS]}" for number, body in entries]
    return "\n".join([header] + rows)


def file_unit(root: Path, relative: str, preview_lines: int = FILE_PREVIEW_LINES) -> Unit | None:
    """A whole file as one unit: its path, its opening lines, and an outline of the rest.

    A file within `PREVIEW_WHOLE_FILE_MULTIPLE` of the preview length is shown
    whole rather than cut short of its end, and needs no outline. Anything longer
    gets its head followed by a line-numbered outline of the whole file, so
    relevant content below the head is still visible. The two together are capped
    at `FILE_PREVIEW_CHARS`.
    """
    text = read_text(root / relative)
    if text is None:
        return None
    lines = text.splitlines()
    size_lines = max(len(lines), 1)
    if len(lines) <= preview_lines * PREVIEW_WHOLE_FILE_MULTIPLE:
        body = "\n".join(lines)
    else:
        body = "\n".join(lines[:preview_lines])
        entries = file_outline(relative, text, lines, after=preview_lines)
        budget = FILE_PREVIEW_CHARS - len(body) - 2
        if entries and budget > 0:
            body += "\n\n" + _clip(render_outline(entries, size_lines), budget)
    return Unit(
        path=relative,
        line_start=1,
        line_end=size_lines,
        kind="file",
        text=_clip(body, FILE_PREVIEW_CHARS),
        size_lines=size_lines,
    )


def language_for(relative: str) -> str | None:
    """Grammar name for a path, or None when sieve has no grammar for it."""
    name = LANGUAGE_BY_SUFFIX.get(Path(relative).suffix)
    if name is None:
        return None
    try:
        from tree_sitter_language_pack import has_language
    except ImportError:  # pragma: no cover - the grammar pack is a hard dependency
        return None
    try:
        return name if has_language(name) else None
    except Exception:  # pragma: no cover - a broken grammar is not sieve's problem
        return None


def function_spans(text: str, language: str) -> list[tuple[int, int, str | None]]:
    """(first line, last line, symbol) for each top-level function-like node.

    One-based and inclusive. A function nested inside another is not emitted
    separately; it stays part of the text of the function that contains it.
    """
    try:
        from tree_sitter_language_pack import get_parser
    except ImportError:  # pragma: no cover
        return []
    try:
        parser = get_parser(language)
    except Exception:  # pragma: no cover
        return []
    encoded = text.encode("utf-8")
    root = parser.parse(encoded).root_node

    spans: list[tuple[int, int, str | None]] = []

    def symbol_of(node) -> str | None:
        name = node.child_by_field_name("name")
        if name is None:
            return None
        return encoded[name.start_byte : name.end_byte].decode("utf-8", "replace")

    def visit(node) -> None:
        for child in node.children:
            if child.type in FUNCTION_NODE_TYPES:
                spans.append((child.start_point[0] + 1, child.end_point[0] + 1, symbol_of(child)))
                continue  # nested functions stay with their enclosing function
            visit(child)

    visit(root)
    spans.sort()
    return spans


def chunk_units(relative: str, text: str, size: int = CHUNK_LINES) -> list[Unit]:
    """Fixed-size line chunks, used when a file has no grammar or no functions."""
    lines = text.splitlines()
    if not lines:
        return []
    units = []
    for start in range(0, len(lines), size):
        window = lines[start : start + size]
        if not "".join(window).strip():
            continue
        units.append(
            Unit(
                path=relative,
                line_start=start + 1,
                line_end=start + len(window),
                kind="chunk",
                text=_clip("\n".join(window)),
                size_lines=len(lines),
            )
        )
    return units


def split_file(root: Path, relative: str, chunk_size: int = CHUNK_LINES) -> list[Unit]:
    """Split one file into function units, falling back to fixed-line chunks.

    Chunks are used when the file's extension has no tree-sitter grammar, and
    also when a grammar exists but finds no function in the file — a module of
    top-level statements would otherwise disappear from the results.
    """
    text = read_text(root / relative)
    if text is None:
        return []
    language = language_for(relative)
    if language is not None:
        lines = text.splitlines()
        spans = function_spans(text, language)
        units = []
        for first, last, symbol in spans:
            units.append(
                Unit(
                    path=relative,
                    line_start=first,
                    line_end=last,
                    kind="function",
                    text=_clip("\n".join(lines[first - 1 : last])),
                    symbol=symbol,
                    size_lines=len(lines),
                )
            )
        if units:
            return units
    return chunk_units(relative, text, chunk_size)
