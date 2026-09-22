"""Enumeration: gitignore-aware walk, function splitting, chunk fallback."""

from pathlib import Path

from sieve.enumerate import (
    CHUNK_LINES,
    FILE_PREVIEW_CHARS,
    MAX_OUTLINE_ENTRIES,
    MAX_UNIT_CHARS,
    chunk_units,
    file_outline,
    file_unit,
    function_spans,
    language_for,
    split_file,
    walk_repo,
)


def build(tmp_path: Path, files: dict[str, str]) -> Path:
    for name, body in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")
    return tmp_path


def test_walk_honours_gitignore_and_skips_git(tmp_path):
    root = build(
        tmp_path,
        {
            ".gitignore": "build/\n*.log\n",
            "a.py": "x = 1\n",
            "build/out.py": "y = 2\n",
            "noisy.log": "junk\n",
            "pkg/b.py": "z = 3\n",
            ".git/config": "[core]\n",
        },
    )
    files, errors = walk_repo(root)
    assert errors == []
    assert files == [".gitignore", "a.py", "pkg/b.py"]


def test_nested_gitignore_applies_from_its_own_directory(tmp_path):
    root = build(
        tmp_path,
        {".gitignore": "", "pkg/.gitignore": "skip.py\n", "pkg/skip.py": "1\n", "skip.py": "1\n"},
    )
    files, _ = walk_repo(root)
    assert "skip.py" in files
    assert "pkg/skip.py" not in files


def test_negation_can_re_include(tmp_path):
    root = build(tmp_path, {".gitignore": "*.log\n!keep.log\n", "drop.log": "a\n", "keep.log": "b\n"})
    files, _ = walk_repo(root)
    assert "keep.log" in files
    assert "drop.log" not in files


def test_binary_files_are_skipped(tmp_path):
    root = build(tmp_path, {"a.py": "x = 1\n"})
    (root / "blob.bin").write_bytes(b"\x00\x01\x02binary")
    files, _ = walk_repo(root)
    assert files == ["a.py"]


def test_file_unit_previews_the_head_and_reports_the_whole_span(tmp_path):
    root = build(tmp_path, {"long.py": "".join(f"line {i}\n" for i in range(1, 101))})
    unit = file_unit(root, "long.py", preview_lines=5)
    assert unit.kind == "file"
    assert unit.line_start == 1
    assert unit.line_end == 100
    assert unit.size_lines == 100
    assert unit.text.splitlines()[:5] == [f"line {i}" for i in range(1, 6)]


def test_a_short_file_is_shown_whole_with_no_outline(tmp_path):
    root = build(tmp_path, {"short.md": "# Title\n\nbody\n"})
    unit = file_unit(root, "short.md")
    assert unit.text == "# Title\n\nbody"
    assert "sieve outline" not in unit.text
    assert unit.size_lines == 3


def test_a_long_file_gets_its_head_then_an_outline_of_the_rest(tmp_path):
    body = "".join(f"line {i}\n" for i in range(1, 101))
    root = build(tmp_path, {"long.txt": body})
    unit = file_unit(root, "long.txt", preview_lines=5)
    head, _, outline = unit.text.partition("... [sieve outline")
    assert head.split() == [w for i in range(1, 6) for w in ("line", str(i))]
    assert "100 lines total" in outline
    # Sampled lines come from past the head and span to the end of the file.
    assert "L6: line 6" in outline
    assert "L100: line 100" in outline


def test_a_markdown_outline_is_its_headings_and_skips_fenced_ones(tmp_path):
    lines = ["intro"] * 90
    lines[10] = "## Real heading"
    lines[20] = "```"
    lines[21] = "# not a heading"
    lines[22] = "```"
    lines[30] = "### Another"
    entries = file_outline("doc.md", "\n".join(lines), lines)
    assert entries == [(11, "## Real heading"), (31, "### Another")]


def test_a_code_outline_names_top_level_declarations_with_line_numbers(tmp_path):
    source = (
        "import os\n"
        "\n"
        "CONST = 1\n"
        "\n"
        "class Thing:\n"
        "    def method(self):\n"
        "        return 2\n"
        "\n"
        "@decorator\n"
        "def top(a, b):\n"
        "    return a\n"
    )
    entries = file_outline("m.py", source, source.splitlines())
    assert (5, "class Thing:") in entries
    assert (10, "def top(a, b):") in entries
    assert all("@decorator" != text for _, text in entries)


def test_an_outline_is_thinned_but_still_spans_the_file(tmp_path):
    lines = [f"## H{i}" if i % 2 == 0 else "text" for i in range(400)]
    entries = file_outline("doc.md", "\n".join(lines), lines)
    assert len(entries) == MAX_OUTLINE_ENTRIES
    assert entries[0] == (1, "## H0")
    assert entries[-1] == (399, "## H398")


def test_the_whole_preview_stays_inside_the_character_budget(tmp_path):
    body = "".join(f"def f{i}():\n    return {i}\n" for i in range(4000))
    root = build(tmp_path, {"many.py": body})
    unit = file_unit(root, "many.py")
    assert len(unit.text) <= FILE_PREVIEW_CHARS + 100


def test_python_functions_split_with_nested_ones_kept_inside(tmp_path):
    source = (
        "import os\n"
        "\n"
        "def outer():\n"
        "    def inner():\n"
        "        return 1\n"
        "    return inner()\n"
        "\n"
        "class Thing:\n"
        "    def method(self):\n"
        "        return 2\n"
    )
    root = build(tmp_path, {"m.py": source})
    units = split_file(root, "m.py")
    assert [u.kind for u in units] == ["function", "function"]
    assert [u.symbol for u in units] == ["outer", "method"]
    assert [(u.line_start, u.line_end) for u in units] == [(3, 6), (9, 10)]
    assert "def inner" in units[0].text


def test_a_grammarless_file_falls_back_to_fixed_chunks(tmp_path):
    body = "".join(f"line {i}\n" for i in range(1, 131))
    root = build(tmp_path, {"notes.md": body})
    assert language_for("notes.md") is None
    units = split_file(root, "notes.md")
    assert [u.kind for u in units] == ["chunk", "chunk", "chunk"]
    assert units[0].line_start == 1
    assert units[0].line_end == CHUNK_LINES
    assert units[2].line_end == 130


def test_a_source_file_with_no_functions_falls_back_to_chunks(tmp_path):
    root = build(tmp_path, {"script.py": "import os\nprint(os.getcwd())\n"})
    units = split_file(root, "script.py")
    assert [u.kind for u in units] == ["chunk"]


def test_chunks_skip_blank_windows():
    units = chunk_units("f.txt", "\n" * 200)
    assert units == []


def test_oversized_units_are_clipped(tmp_path):
    root = build(tmp_path, {"big.md": "x" * (MAX_UNIT_CHARS * 2)})
    unit = file_unit(root, "big.md")
    assert len(unit.text) < MAX_UNIT_CHARS + 100
    assert unit.text.endswith("[unit truncated by sieve]")


def test_go_functions_are_found():
    source = "package main\n\nfunc Add(a, b int) int {\n\treturn a + b\n}\n"
    spans = function_spans(source, "go")
    assert spans == [(3, 5, "Add")]
