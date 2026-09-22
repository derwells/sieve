"""jev_search: deterministic variants, url canonicalisation, dedupe, rerank."""

import pytest

from sieve.backends import SearchBackendError, SearchHit
from sieve.search import (
    candidate_text,
    canonical_url,
    dedupe_hits,
    jev_search,
    names_software,
    propose_variants,
    rephrase,
    strip_filler,
)

from .conftest import FakeClient, FakeSearchBackend


def test_the_original_query_is_always_the_first_variant():
    assert propose_variants("why is my build slow")[0] == "why is my build slow"


def test_filler_words_are_stripped_in_the_second_variant():
    assert strip_filler("how do I please fix the a build") == "fix build"


def test_stripping_everything_falls_back_to_the_query():
    assert strip_filler("how do I") == "how do I"


def test_a_rephrase_reorders_the_key_terms_and_drops_quotes():
    assert rephrase('the "tree sitter" python grammar') == "grammar tree sitter python"


def test_a_short_query_is_not_rotated():
    assert rephrase("postgres bloat") == "postgres bloat"


@pytest.mark.parametrize(
    "query",
    ["the TypeSafe SDK", "install tree-sitter-language-pack", "what does FastMCP do", "reading a foo.py file"],
)
def test_software_queries_are_recognised(query):
    assert names_software(query)


@pytest.mark.parametrize("query", ["best sourdough starter recipe", "when to plant tulip bulbs"])
def test_plain_queries_are_not_taken_for_software(query):
    assert not names_software(query)


def test_software_queries_get_docs_and_github_variants():
    variants = propose_variants("TypeSafe Jev rerank cookbook", 4)
    assert variants[0] == "TypeSafe Jev rerank cookbook"
    assert variants[-2].endswith(" docs")
    assert variants[-1].endswith(" github")
    assert len(variants) == 4


def test_variants_are_capped_and_floored():
    assert len(propose_variants("the TypeSafe SDK reference guide", 9)) == 4
    assert len(propose_variants("the TypeSafe SDK reference guide", 1)) == 2


def test_variants_are_unique_and_deterministic():
    first = propose_variants("TypeSafe Jev rerank cookbook", 4)
    assert first == propose_variants("TypeSafe Jev rerank cookbook", 4)
    assert len(set(first)) == len(first)


@pytest.mark.parametrize(
    "url,expected",
    [
        ("HTTPS://WWW.Example.COM/a/b/", "https://example.com/a/b"),
        ("https://example.com:443/a", "https://example.com/a"),
        ("http://example.com:80/a", "http://example.com/a"),
        ("http://example.com:8080/a", "http://example.com:8080/a"),
        ("https://example.com/a#section", "https://example.com/a"),
        ("https://example.com/a?utm_source=x&utm_medium=y&q=1", "https://example.com/a?q=1"),
        ("https://example.com/a?fbclid=z&gclid=w&ref=hn", "https://example.com/a"),
        ("https://example.com/", "https://example.com"),
    ],
)
def test_canonical_url(url, expected):
    assert canonical_url(url) == expected


def test_dedupe_keeps_the_first_title_and_prefers_a_non_empty_snippet():
    hits = [
        SearchHit(url="https://www.Example.com/a/", title="First", snippet=""),
        SearchHit(url="https://example.com/a?utm_source=x", title="Second", snippet="real snippet"),
        SearchHit(url="https://other.test/b", title="Other", snippet="b"),
    ]
    deduped = dedupe_hits(hits)
    assert [h.url for h in deduped] == ["https://www.Example.com/a/", "https://other.test/b"]
    assert deduped[0].title == "First"
    assert deduped[0].snippet == "real snippet"


def test_dedupe_drops_empty_urls():
    assert dedupe_hits([SearchHit(url="  ", title="x")]) == []


def test_candidate_text_omits_an_absent_snippet():
    from sieve.search import DedupedHit

    with_snippet = DedupedHit(key="k", url="https://a.test", title="T", snippet="S")
    without = DedupedHit(key="k", url="https://a.test", title="T", snippet="")
    assert candidate_text(with_snippet) == "T — https://a.test — S"
    assert candidate_text(without) == "T — https://a.test"


def hits(*pairs):
    return [SearchHit(url=u, title=t, snippet=s) for u, t, s in pairs]


async def test_end_to_end_ranks_deduped_hits_by_probability():
    backend = FakeSearchBackend(
        default=hits(
            ("https://docs.typesafe.ai/cookbooks/rerank_typesafe", "Rerank cookbook", "how to rerank"),
            ("https://example.com/unrelated/", "Unrelated", ""),
            ("https://www.example.com/unrelated?utm_source=x", "Unrelated dupe", "a snippet"),
        )
    )

    def scorer(state, index):
        text = state["candidates"][index]["text"]
        return 0.97 if "typesafe" in text else 0.11

    client = FakeClient(scorer=scorer)
    out = await jev_search("TypeSafe Jev rerank cookbook", top_k=5, backend=backend, client=client)

    assert out["backend"] == "fake"
    assert out["variants"] == propose_variants("TypeSafe Jev rerank cookbook", 3)
    assert len(out["variants"]) == 3
    assert backend.queries == out["variants"]
    assert backend.max_concurrent > 1
    assert [r["url"] for r in out["results"]] == [
        "https://docs.typesafe.ai/cookbooks/rerank_typesafe",
        "https://example.com/unrelated/",
    ]
    assert out["results"][0]["probability"] == 0.97
    assert out["results"][1]["snippet"] == "a snippet"
    assert out["hits_found"] == 9
    assert out["hits_deduped"] == 2
    assert out["usage"]["requests"] == 1
    assert out["usage"]["cost_usd"] >= 0
    assert [c["query"] for c in out["backend_calls"]] == out["variants"]
    assert out["wall_seconds"] >= 0


async def test_top_k_truncates_the_ranked_list():
    backend = FakeSearchBackend(default=hits(*[(f"https://a{i}.test/", f"t{i}", "") for i in range(8)]))
    client = FakeClient(scorer=lambda state, i: 0.9 - i / 100)
    out = await jev_search("plain english question", top_k=3, backend=backend, client=client)
    assert len(out["results"]) == 3


async def test_one_failing_variant_does_not_lose_the_others():
    variants = propose_variants("TypeSafe Jev rerank cookbook", 3)
    backend = FakeSearchBackend(
        default=hits(("https://a.test/", "A", "")),
        error_on=[variants[1]],
    )
    client = FakeClient()
    out = await jev_search("TypeSafe Jev rerank cookbook", backend=backend, client=client)
    assert len(out["backend_calls"]) == 2
    assert out["backend_errors"][0]["query"] == variants[1]
    assert [r["url"] for r in out["results"]] == ["https://a.test/"]


async def test_every_variant_failing_raises():
    variants = propose_variants("TypeSafe Jev rerank cookbook", 3)
    backend = FakeSearchBackend(error_on=variants)
    with pytest.raises(SearchBackendError, match="every fake search variant failed"):
        await jev_search("TypeSafe Jev rerank cookbook", backend=backend, client=FakeClient())
