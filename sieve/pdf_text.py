"""Text from a PDF, extracted in a child process with hard limits.

PDF parsing runs untrusted input through a large parser, so it never runs in
the server process. `pdftotext` (poppler) is used when it is installed, since
it keeps word spacing and column order better; otherwise, or if it fails,
`pypdf` runs in a child Python. Either child runs under a memory limit and is
killed if it outlives the timeout. At most `MAX_PAGES` pages are read and
`MAX_CHARS` characters kept, and the text is NFKC-normalised so ligatures such
as "ﬂ" match typed quotes. Every failure is a reason string, never the raw
bytes: an encrypted PDF, a PDF without a text layer, or a parse error.

    python -m sieve.pdf_text < file.pdf          # pypdf: {"text": ...} or {"error": ...}
    python -m sieve.pdf_text --exec pdftotext ... # set the limit, then exec pdftotext

Both children are this module's `main`, which sets its own memory limit first,
so the server never needs `preexec_fn` (unsafe in a process with threads).
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import tempfile
import unicodedata

#: Pages read from one PDF, at most.
MAX_PAGES = 200
#: Characters kept from one PDF, at most.
MAX_CHARS = 1_000_000
#: Wall time for one extraction, in seconds.
TIMEOUT_SECONDS = 30.0
#: Address-space limit for the child, in bytes.
MEMORY_LIMIT_BYTES = 1_000_000_000

PDF_MAGIC = b"%PDF-"


def looks_like_pdf(payload: bytes) -> bool:
    """True for bytes that open with the PDF header, allowing leading whitespace."""
    return payload[:1024].lstrip().startswith(PDF_MAGIC)


def _extract(payload: bytes, max_pages: int, max_chars: int) -> dict:
    import io

    from pypdf import PdfReader

    try:
        reader = PdfReader(io.BytesIO(payload))
        if reader.is_encrypted:
            try:
                if not reader.decrypt(""):
                    return {"error": "PDF is encrypted"}
            except Exception:
                return {"error": "PDF is encrypted"}
        total = len(reader.pages)
        parts: list[str] = []
        size = 0
        read = 0
        for page in reader.pages[:max_pages]:
            text = page.extract_text() or ""
            read += 1
            parts.append(text)
            size += len(text)
            if size >= max_chars:
                break
    except Exception as e:
        return {"error": f"PDF could not be parsed: {type(e).__name__}: {e}"[:300]}
    text = "\n\n".join(parts)[:max_chars]
    if not text.strip():
        return {"error": "PDF has no extractable text layer (it may be scanned images)"}
    return {"text": text, "pages": total, "pages_read": read, "truncated": read < total or size > max_chars}


def _limit_memory() -> None:
    try:
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (MEMORY_LIMIT_BYTES, MEMORY_LIMIT_BYTES))
    except (ImportError, ValueError, OSError):
        pass


async def _run(argv: list[str], payload: bytes | None, timeout: float) -> tuple[int, bytes, bytes] | str:
    """Run a child to completion; a string is the reason it did not finish."""
    try:
        process = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.PIPE if payload is not None else asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except OSError as e:
        return f"could not start: {e}"
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(payload), timeout=timeout)
    except (TimeoutError, asyncio.TimeoutError):
        process.kill()
        await process.wait()
        return f"timed out after {timeout:.0f}s"
    return process.returncode or 0, stdout, stderr


async def _pdftotext(executable: str, payload: bytes, timeout: float) -> dict:
    try:
        with tempfile.TemporaryDirectory(prefix="sieve-pdf-") as scratch:
            path = os.path.join(scratch, "source.pdf")
            with open(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as f:
                f.write(payload)
            argv = [sys.executable, "-m", "sieve.pdf_text", "--exec", executable,
                    "-q", "-enc", "UTF-8", "-l", str(MAX_PAGES), path, "-"]
            outcome = await _run(argv, None, timeout)
    except OSError as e:
        return {"error": f"pdftotext scratch file failed: {e}"}
    if isinstance(outcome, str):
        return {"error": f"pdftotext {outcome}"}
    code, stdout, stderr = outcome
    if code != 0:
        # pdftotext exits 1 on a file it cannot open, 3 on permissions (encryption).
        reason = "PDF is encrypted" if code == 3 else f"pdftotext exit {code}"
        return {"error": reason}
    text = stdout.decode("utf-8", "replace")
    if not text.strip():
        return {"error": "PDF has no extractable text layer (it may be scanned images)"}
    return {"text": text[:MAX_CHARS], "truncated": len(text) > MAX_CHARS}


def _normalised(result: dict, extractor: str) -> dict:
    if "text" in result:
        result = {**result, "text": unicodedata.normalize("NFKC", result["text"])}
    return {**result, "extractor": extractor}


async def pdf_to_text(payload: bytes, *, timeout: float = TIMEOUT_SECONDS) -> dict:
    """`{"text", "truncated", "extractor", ...}` or `{"error", "extractor"}`. Never raises.

    pdftotext first when installed; pypdf if it is absent or fails. A PDF that
    neither can read reports the last reason.
    """
    executable = shutil.which("pdftotext")
    if executable:
        first = await _pdftotext(executable, payload, timeout)
        if "text" in first:
            return _normalised(first, "pdftotext")
    result = _normalised(await _pypdf(payload, timeout), "pypdf")
    if "error" in result and executable:
        result["error"] = f"{result['error']} (pdftotext: {first['error']})"
    return result


async def _pypdf(payload: bytes, timeout: float) -> dict:
    try:
        outcome = await _run([sys.executable, "-m", "sieve.pdf_text"], payload, timeout)
    except Exception as e:  # never raise out of extraction
        return {"error": f"PDF extraction failed: {type(e).__name__}: {e}"}
    if isinstance(outcome, str):
        return {"error": f"PDF extraction {outcome}"}
    code, stdout, stderr = outcome
    try:
        result = json.loads(stdout)
    except ValueError:
        tail = stderr.decode("utf-8", "replace").strip()[-200:]
        return {"error": f"PDF extraction failed (exit {code}){': ' + tail if tail else ''}"}
    return result if isinstance(result, dict) and ("text" in result or "error" in result) else {"error": "PDF extraction returned nothing"}


def main() -> None:
    _limit_memory()
    if len(sys.argv) > 2 and sys.argv[1] == "--exec":
        os.execv(sys.argv[2], sys.argv[2:])  # the limit carries over to pdftotext
    payload = sys.stdin.buffer.read()
    sys.stdout.write(json.dumps(_extract(payload, MAX_PAGES, MAX_CHARS)))


if __name__ == "__main__":
    main()
