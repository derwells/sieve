"""PDF sources: bounded text extraction, honest failures, and quote matching."""

import httpx2
import pytest

from sieve import fetch as fetch_module
from sieve import pdf_text
from sieve.cache import AnswerCache
from sieve.evidence import find_quote
from sieve.fetch import fetch_source
from sieve.pdf_text import looks_like_pdf, pdf_to_text
from sieve.verify import jev_verify

from .conftest import FakeClient

PHRASE = "The pool reached fifty candidates in seven requests, per the trial."


def make_pdf(lines: list[str]) -> bytes:
    """A minimal valid PDF with one page of Helvetica text, one line per entry."""
    ops = ["BT", "/F1 11 Tf", "72 720 Td", "14 TL"]
    for line in lines:
        escaped = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        ops.append(f"({escaped}) Tj T*")
    ops.append("ET")
    stream = "\n".join(ops).encode("latin-1")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    for offset in offsets:
        out += b"%010d 00000 n \n" % offset
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objects) + 1, xref)
    return bytes(out)


PDF = make_pdf(["Results", PHRASE])
EMPTY_PDF = make_pdf([])


@pytest.fixture(params=["pdftotext", "pypdf"])
def extractor(request, monkeypatch):
    """Run each extraction test through both paths; pypdf alone when pdftotext is hidden."""
    if request.param == "pypdf":
        monkeypatch.setattr(pdf_text.shutil, "which", lambda name: None)
    elif pdf_text.shutil.which("pdftotext") is None:
        pytest.skip("pdftotext is not installed")
    return request.param


def test_the_pdf_header_is_recognised():
    assert looks_like_pdf(PDF)
    assert looks_like_pdf(b"\n  %PDF-1.7 ...")
    assert not looks_like_pdf(b"<html>%PDF-</html>")


async def test_text_comes_out_of_a_pdf(extractor):
    result = await pdf_to_text(PDF)
    assert result["extractor"] == extractor
    assert PHRASE in " ".join(result["text"].split())


async def test_a_pdf_without_a_text_layer_is_an_error(extractor):
    result = await pdf_to_text(EMPTY_PDF)
    assert "no extractable text" in result["error"]
    assert "text" not in result


async def test_a_corrupt_pdf_is_an_error_not_bytes(extractor):
    result = await pdf_to_text(b"%PDF-1.4\n" + b"\x00garbage" * 50)
    assert "error" in result
    assert "text" not in result


async def test_a_hung_extractor_is_killed():
    outcome = await pdf_text._run(["sleep", "5"], None, 0.2)
    assert outcome == "timed out after 0s"


def test_ligatures_are_normalised():
    assert pdf_text._normalised({"text": "overﬂowing ﬁle"}, "x")["text"] == "overflowing file"


def pdf_client(payload, content_type="application/pdf", path="/paper"):
    def handler(request):
        return httpx2.Response(200, content=payload, headers={"content-type": content_type})

    return httpx2.AsyncClient(transport=httpx2.MockTransport(handler))


async def test_a_fetched_pdf_is_text_with_the_hash_of_its_bytes():
    source = await fetch_source("https://example.org/paper", client=pdf_client(PDF))
    assert source.ok, source.error
    assert source.format == "pdf"
    assert not source.text.startswith("%PDF")
    assert PHRASE in " ".join(source.text.split())
    assert source.source_version == fetch_module.version_of(PDF)
    assert source.byte_length == len(PDF)


async def test_an_undeclared_pdf_is_recognised_by_its_header():
    source = await fetch_source("https://example.org/download", client=pdf_client(PDF, "application/octet-stream"))
    assert source.format == "pdf"
    assert PHRASE in " ".join(source.text.split())


async def test_a_declared_pdf_gets_the_larger_byte_cap(monkeypatch):
    monkeypatch.setattr(fetch_module, "MAX_SOURCE_BYTES", 100)
    source = await fetch_source("https://example.org/paper", client=pdf_client(PDF))
    assert source.ok, source.error
    monkeypatch.setattr(fetch_module, "MAX_PDF_BYTES", 100)
    capped = await fetch_source("https://example.org/paper", client=pdf_client(PDF))
    assert capped.error == "response exceeds the 100 byte cap"


async def test_an_html_page_served_as_pdf_is_an_honest_error():
    source = await fetch_source("https://example.org/paper.pdf", client=pdf_client(b"<html>Please log in</html>"))
    assert not source.ok
    assert "not a PDF" in source.error
    assert source.text == ""


async def test_an_unreadable_pdf_fails_with_a_reason_and_no_text():
    source = await fetch_source("https://example.org/paper", client=pdf_client(EMPTY_PDF))
    assert not source.ok
    assert "no extractable text" in source.error
    assert source.text == ""


async def test_a_local_pdf_is_read_as_text(tmp_path):
    path = tmp_path / "paper.pdf"
    path.write_bytes(PDF)
    source = await fetch_source(str(path))
    assert source.format == "pdf"
    assert PHRASE in " ".join(source.text.split())


async def test_verify_matches_a_quote_inside_a_pdf(tmp_path):
    path = tmp_path / "paper.pdf"
    path.write_bytes(PDF)
    out = await jev_verify(
        records=[{"claim": "The pool reached fifty candidates.", "citations": [{"locator": str(path), "quote": "reached fifty candidates in seven requests"}]}],
        client=FakeClient(scorer=lambda state, i: {"supports_fully": 0.9, "partially_supports": 0.05, "contradicts": 0.02, "does_not_address": 0.03}),
        cache=AnswerCache(tmp_path / "a.sqlite3"),
    )
    assert out["results"][0]["quote_match"] in {"exact", "normalised"}
    assert not any(f["flag"] == "quote_not_found" for f in out["results"][0]["flags"])


# --- quote matching around PDF extraction artefacts ---------------------------


def test_a_space_before_punctuation_does_not_block_a_match():
    text = "Its protocol is to let different experts,\np1 , ..., pN , take turns in sequence."
    assert find_quote(text, "experts, p1, ..., pN, take turns in sequence.").found


def test_a_closing_period_the_source_lacks_does_not_block_a_match():
    text = "78% of news articles were derived from news agency\nsources (46% partially, 32% wholly) and introduced"
    match = find_quote(text, "78% of news articles were derived from news agency sources (46% partially, 32% wholly).")
    assert match.kind == "normalised"
    assert text[match.start:match.end].endswith("wholly)")


def test_a_closing_period_after_a_digit_is_kept():
    assert not find_quote("the cost was 3.55 dollars", "the cost was 3.5.").found


def test_the_memory_limit_reaches_the_program_the_child_execs():
    import subprocess
    import sys

    out = subprocess.run(
        [sys.executable, "-m", "sieve.pdf_text", "--exec", "/bin/sh", "-c", "ulimit -v"],
        capture_output=True, text=True, timeout=10,
    )
    assert out.stdout.strip() == str(pdf_text.MEMORY_LIMIT_BYTES // 1024)
