"""Fetch a cited source once and turn it into plain text with a version hash.

A locator is a URL or a file path. Every fetch is capped in size and time, and a
failure is reported as a reason on the citation rather than folded into any
support score: sieve never lets a 404 look like weak evidence.

A PDF, recognised by its content type, `.pdf` suffix or `%PDF-` header, is
turned into text by `sieve.pdf_text` in a bounded child process. A PDF that
yields no text is a fetch failure with the reason, never its raw bytes.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from .pdf_text import looks_like_pdf, pdf_to_text

#: Largest source sieve will read, in bytes.
MAX_SOURCE_BYTES = 2_000_000
#: Largest PDF sieve will read, in bytes. Papers with figures run past 2 MB.
MAX_PDF_BYTES = 25_000_000
#: Per-URL timeout, in seconds.
FETCH_TIMEOUT_SECONDS = 15.0
#: Mintlify serves any docs page as Markdown when `.md` is appended to its path.
MINTLIFY_HOSTS = frozenset({"docs.typesafe.ai"})

#: Suffixes that are read as markup rather than as plain text.
HTML_SUFFIXES = frozenset({".html", ".htm", ".xhtml"})

#: Elements whose text is layout or code, never prose.
SKIP_TAGS = frozenset({"script", "style", "noscript", "head", "svg", "template", "iframe"})
#: Elements that end a paragraph of text.
BLOCK_TAGS = frozenset(
    """
    address article aside blockquote body caption div dd dl dt fieldset figcaption figure
    footer form h1 h2 h3 h4 h5 h6 header hr li main nav ol p pre section table tbody td
    tfoot th thead tr ul
    """.split()
)


@dataclass(frozen=True)
class Source:
    """One fetched locator: its text, what it hashed to, and why it failed."""

    locator: str
    text: str = ""
    source_version: str = ""
    byte_length: int = 0
    error: str | None = None
    #: "pdf" when the text was extracted from a PDF.
    format: str = ""

    @property
    def ok(self) -> bool:
        return self.error is None


class _TextExtractor(HTMLParser):
    """HTML to text, keeping paragraph breaks and dropping scripts and styles."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in SKIP_TAGS:
            self._skip_depth += 1
        elif tag == "br":
            self.parts.append("\n")
        elif tag in BLOCK_TAGS:
            self.parts.append("\n\n")

    def handle_startendtag(self, tag: str, attrs) -> None:
        if tag == "br":
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
        elif tag in BLOCK_TAGS:
            self.parts.append("\n\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            self.parts.append(data)

    def text(self) -> str:
        return "".join(self.parts)


def html_to_text(markup: str) -> str:
    """Plain text with paragraph breaks preserved. Standard library only."""
    parser = _TextExtractor()
    try:
        parser.feed(markup)
        parser.close()
    except Exception:  # a malformed page still yields whatever was parsed
        pass
    return tidy_text(parser.text())


def tidy_text(text: str) -> str:
    """Trim line-trailing space and collapse runs of blank lines into one."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def version_of(payload: bytes) -> str:
    """The identity of the bytes a verdict was computed against."""
    return hashlib.sha256(payload).hexdigest()


def is_url(locator: str) -> bool:
    return urlsplit(locator.strip()).scheme in {"http", "https"}


def markdown_url(url: str) -> str:
    """The Markdown twin of a Mintlify docs page, or the url unchanged."""
    parts = urlsplit(url)
    host = (parts.hostname or "").lower()
    if host not in MINTLIFY_HOSTS:
        return url
    path = parts.path
    if not path or path.endswith("/") or "." in path.rsplit("/", 1)[-1]:
        return url
    return urlunsplit((parts.scheme, parts.netloc, path + ".md", parts.query, ""))


def resolve_path(locator: str, base_path: str | None = None) -> Path:
    """A local locator as an absolute path, relative locators taken from `base_path`."""
    path = Path(locator).expanduser()
    if not path.is_absolute() and base_path:
        path = Path(base_path).expanduser() / path
    return path


def is_pdf_name(name: str) -> bool:
    return urlsplit(name).path.lower().endswith(".pdf")


async def read_local(locator: str, base_path: str | None = None) -> Source:
    """Read a file from disk, capped at `MAX_SOURCE_BYTES` (`MAX_PDF_BYTES` for a PDF)."""
    path = resolve_path(locator, base_path)
    try:
        if not path.is_file():
            return Source(locator=locator, error=f"not a readable file: {path}")
        cap = MAX_PDF_BYTES if path.suffix.lower() == ".pdf" else MAX_SOURCE_BYTES
        size = path.stat().st_size
        if size > cap:
            return Source(locator=locator, error=f"file is {size} bytes, over the {cap} byte cap")
        payload = path.read_bytes()
    except OSError as e:
        return Source(locator=locator, error=f"{type(e).__name__}: {e}")
    if looks_like_pdf(payload):
        return await _source_from_pdf(locator, payload)
    markup = path.suffix.lower() in HTML_SUFFIXES
    return _source_from_bytes(locator, payload, html=markup)


async def _source_from_pdf(locator: str, payload: bytes) -> Source:
    version = version_of(payload)
    if not looks_like_pdf(payload):
        return Source(locator=locator, source_version=version, byte_length=len(payload),
                      error="served as a PDF but the bytes are not a PDF")
    result = await pdf_to_text(payload)
    if "error" in result:
        return Source(locator=locator, source_version=version, byte_length=len(payload), error=result["error"], format="pdf")
    return Source(locator=locator, text=tidy_text(result["text"]), source_version=version, byte_length=len(payload), format="pdf")


def _source_from_bytes(locator: str, payload: bytes, *, html: bool) -> Source:
    decoded = payload.decode("utf-8", errors="replace")
    text = html_to_text(decoded) if html else tidy_text(decoded)
    return Source(
        locator=locator,
        text=text,
        source_version=version_of(payload),
        byte_length=len(payload),
    )


async def fetch_url(locator: str, *, client=None) -> Source:
    """GET a URL with a timeout and a size cap, extracting text from HTML."""
    import httpx2

    url = markdown_url(locator)
    owned = client is None
    client = client or httpx2.AsyncClient(follow_redirects=True, timeout=FETCH_TIMEOUT_SECONDS)
    try:
        async with client.stream("GET", url, timeout=FETCH_TIMEOUT_SECONDS) as response:
            if response.status_code >= 400:
                return Source(locator=locator, error=f"HTTP {response.status_code}")
            content_type = (response.headers.get("content-type") or "").lower()
            declared_pdf = "application/pdf" in content_type or is_pdf_name(str(response.url))
            cap = MAX_PDF_BYTES if declared_pdf else MAX_SOURCE_BYTES
            chunks: list[bytes] = []
            total = 0
            async for chunk in response.aiter_bytes():
                total += len(chunk)
                if total > cap:
                    return Source(
                        locator=locator,
                        error=f"response exceeds the {cap} byte cap",
                    )
                chunks.append(chunk)
    except Exception as e:
        return Source(locator=locator, error=f"{type(e).__name__}: {e}")
    finally:
        if owned:
            await client.aclose()
    payload = b"".join(chunks)
    if "application/pdf" in content_type or looks_like_pdf(payload):
        return await _source_from_pdf(locator, payload)
    html = "html" in content_type or (not content_type and payload.lstrip()[:1] == b"<")
    return _source_from_bytes(locator, payload, html=html)


async def fetch_source(locator: str, *, base_path: str | None = None, client=None) -> Source:
    """Fetch one locator, URL or file path, and never raise for a fetch failure."""
    locator = (locator or "").strip()
    if not locator:
        return Source(locator=locator, error="empty locator")
    if is_url(locator):
        return await fetch_url(locator, client=client)
    return await read_local(locator, base_path)
