"""Offline checks for citation extraction, evidence, scoring, and aggregation."""

import hashlib

import httpx2
import pytest

from sieve.cache import AnswerCache
from sieve.errors import InvalidAnswerError
from sieve.evidence import find_quote, windows_for
from sieve.fetch import fetch_source
from sieve.report import extract_claims
from sieve.verify import jev_verify

from .conftest import FakeChoiceAnswer, FakeClient, FakeResponse, FakeUsage


SUPPORT = {"supports_fully": 0.8, "partially_supports": 0.1, "contradicts": 0.05, "does_not_address": 0.05}
CONTRADICT = {"supports_fully": 0.1, "partially_supports": 0.05, "contradicts": 0.8, "does_not_address": 0.05}


def test_markdown_claim_extraction_keeps_structure_and_citations():
    report = """# Summary
- The release changed.
  - The CLI reads `sieve/server.py`.

| Product | Count | Source |
| --- | --- | --- |
| sieve | three | [docs](https://example.org/docs) |
"""
    claims = extract_claims(report)
    child = next(c for c in claims if "CLI reads" in c["claim"])
    assert child["claim_context"] == "Summary > The release changed."
    assert child["citations"] == [{"locator": "sieve/server.py"}]
    table = next(c for c in claims if "Product: sieve" in c["claim"])
    assert "table columns: Product | Count | Source" in table["claim_context"]
    assert table["citations"] == [{"locator": "https://example.org/docs"}]
    assert all(c["extraction_uncertain"] for c in claims)


def test_quote_normalisation_and_window_offsets():
    text = 'First paragraph.\n\nThe “quoted”   phrase has 3 entries.\n\nLast paragraph.'
    match = find_quote(text, 'the "quoted" phrase has 3 entries.')
    assert match.kind == "normalised"
    windows = windows_for(text, "has 3 entries", match)
    assert len(windows) == 1
    assert windows[0].text == text[windows[0].start:windows[0].end]
    assert "Last paragraph" in windows[0].text
    assert find_quote(text, "missing").kind == "not_found"
    lexical = windows_for("cats nap.\n\nDogs run.\n\nBirds fly.\n\nFish swim.", "Birds fly")
    assert any("Birds fly" in w.text for w in lexical)
    assert all(w.text == "cats nap.\n\nDogs run.\n\nBirds fly.\n\nFish swim."[w.start:w.end] for w in lexical)


async def test_fetches_url_once_and_reports_failure(tmp_path):
    calls = []

    def handler(request):
        calls.append(str(request.url))
        if request.url.path == "/bad":
            return httpx2.Response(404)
        return httpx2.Response(200, text="<p>There are three entries.</p>", headers={"content-type": "text/html"})

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as http_client:
        source = await fetch_source("https://example.org/good", client=http_client)
        out = await jev_verify(
            records=[
                {"claim": "There are three entries.", "citations": [{"locator": "https://example.org/good"}]},
                {"claim": "There are three entries.", "citations": [{"locator": "https://example.org/good"}, {"locator": "https://example.org/bad"}]},
            ],
            client=FakeClient(scorer=lambda state, index: SUPPORT), http_client=http_client,
        )
    assert source.source_version == hashlib.sha256(b"<p>There are three entries.</p>").hexdigest()
    assert calls.count("https://example.org/good") == 2  # one explicit fetch, one verify fetch
    assert calls.count("https://example.org/bad") == 1
    assert out["results"][1]["flags"][0]["flag"] == "fetch_failed"
    assert out["results"][1]["verdict"] == SUPPORT


async def test_typesafe_docs_use_markdown_variant_and_size_cap():
    paths = []

    def handler(request):
        paths.append(request.url.path)
        if request.url.path == "/large":
            return httpx2.Response(200, content=b"x" * 2_000_001)
        return httpx2.Response(200, text="A short source.")

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as http_client:
        source = await fetch_source("https://docs.typesafe.ai/primitives/choice", client=http_client)
        large = await fetch_source("https://example.org/large", client=http_client)
    assert paths == ["/primitives/choice.md", "/large"]
    assert source.text == "A short source."
    assert "byte cap" in large.error


async def test_quote_not_found_still_scores_and_keeps_contradiction(tmp_path):
    path = tmp_path / "source.txt"
    path.write_text("There are three entries.\n\nThe system starts locally.")
    out = await jev_verify(
        records=[{"claim": "There are five entries.", "citations": [{"locator": str(path), "quote": "There are five entries."}]}],
        client=FakeClient(scorer=lambda state, index: CONTRADICT),
    )
    row = out["results"][0]
    assert row["quote_match"] == "not_found"
    assert "quote_not_found" in [flag["flag"] for flag in row["flags"]]
    assert row["evidence"] and row["max_contradicts"] == 0.8
    assert row["best_evidence"]["excerpt"] == path.read_text()[row["best_evidence"]["start"]:row["best_evidence"]["end"]]
    assert out["unqualified_factual_relay_blocked"]


async def test_best_support_and_max_contradiction_are_separate(tmp_path):
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("The count is three.")
    second.write_text("The count is five.")
    client = FakeClient(scorer=lambda state, index: (
        SUPPORT if state["pairs"][index]["source_locator"] == str(first) else CONTRADICT
    ))
    out = await jev_verify(records=[{
        "claim": "The count is three.",
        "citations": [{"locator": str(first)}, {"locator": str(second)}],
    }], client=client)
    row = out["results"][0]
    assert row["verdict"] == SUPPORT
    assert row["best_evidence"]["locator"] == str(first)
    assert row["max_contradicts"] == 0.8
    assert len(row["evidence"]) == 2
    assert out["unqualified_factual_relay_blocked"]


async def test_aggregation_recommendation_premises_and_block(tmp_path):
    path = tmp_path / "source.txt"
    path.write_text("There are three entries.\n\nThe system starts locally.")
    client = FakeClient(scorer=lambda state, index: CONTRADICT if "five" in state["pairs"][index]["claim"] else SUPPORT)
    out = await jev_verify(records=[
        {"claim": "Use the system.", "kind": "recommendation", "premises": ["There are five entries."], "citations": [{"locator": "source.txt", "quote": "There are three entries."}]},
        {"claim": "No source available.", "citations": [{"locator": "missing.txt"}]},
    ], base_path=str(tmp_path), client=client)
    recommendation = out["results"][0]
    assert recommendation["exempt"] and recommendation["verdict"] is None
    assert recommendation["premises"][0]["max_contradicts"] == 0.8
    assert recommendation["premises"][0]["verdict"] == CONTRADICT
    assert recommendation["quote_match"] == "exact"
    assert out["unqualified_factual_relay_blocked"]
    assert out["counts"]["contradicts"] == 1
    assert out["results"][1]["flags"][0]["flag"] == "fetch_failed"
    assert set(client.calls[0].state["pairs"][0]) == {"claim", "claim_context", "evidence_text", "source_locator", "source_version"}


async def test_choice_validation_rejects_invalid_answer(tmp_path):
    path = tmp_path / "source.txt"
    path.write_text("There are three entries.")

    class BrokenClient(FakeClient):
        async def system_one(self, state, questions, **kwargs):
            return FakeResponse({"q0": FakeChoiceAnswer("supports_fully", {"supports_fully": 1.0})}, FakeUsage(10, 1))

    with pytest.raises(InvalidAnswerError, match="omit offered options"):
        await jev_verify(records=[{"claim": "There are three entries.", "citations": [{"locator": str(path)}]}], client=BrokenClient())

    class NonfiniteClient(FakeClient):
        async def system_one(self, state, questions, **kwargs):
            probabilities = {"supports_fully": float("nan"), "partially_supports": 0.0,
                             "contradicts": 0.0, "does_not_address": 0.0}
            return FakeResponse({"q0": FakeChoiceAnswer("supports_fully", probabilities)}, FakeUsage(10, 1))

    with pytest.raises(InvalidAnswerError, match="outside"):
        await jev_verify(records=[{"claim": "There are three entries.", "citations": [{"locator": str(path)}]}], client=NonfiniteClient())


async def test_choice_cache_reuses_distribution(tmp_path):
    path = tmp_path / "source.txt"
    path.write_text("There are three entries.")
    client = FakeClient(scorer=lambda state, index: SUPPORT)
    cache = AnswerCache(tmp_path / "answers.sqlite3")
    records = [{"claim": "There are three entries.", "citations": [{"locator": str(path)}]}]
    try:
        await jev_verify(records=records, client=client, cache=cache)
        out = await jev_verify(records=records, client=client, cache=cache)
    finally:
        cache.close()
    assert len(client.calls) == 1
    assert out["usage"]["cache_hits"] == 1
