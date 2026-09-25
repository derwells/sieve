"""SearXNG backend, retrieval depth and the retrieval cache. Loopback only, no engines."""

import json
import os
import pathlib
import stat
import time

import httpx2
import pytest

from sieve.backends import (
    BACKEND_ENV,
    BRAVE_API_KEY_ENV,
    SEARXNG_URL_ENV,
    BackendResult,
    BraveBackend,
    SearchBackendError,
    SearchHit,
    SearxngBackend,
    backend_name,
    select_backend,
)
from sieve.backends.searxng import parse_searxng_payload, unresponsive_engines
from sieve.search import MAX_DEPTH, dedupe_hits, interleave, jev_search, resolve_depth, variant_counts
from sieve.search_cache import DB_NAME, SearchCache, _make_private, result_key

from .conftest import FakeClient, FakeSearchBackend

PAGE = json.loads((pathlib.Path(__file__).parent / "fixtures" / "searxng_page.json").read_text())


def page_of(n, start=0, **extra):
    """A SearXNG-shaped page of `n` distinct results."""
    return {
        "query": "q",
        "results": [
            {"url": f"https://site{i}.test/p", "title": f"T{i}", "content": f"snippet {i}", "engines": ["google"]}
            for i in range(start, start + n)
        ],
        "unresponsive_engines": [],
        **extra,
    }


def backend_for(handler, **kwargs):
    return SearxngBackend("http://127.0.0.1:8888", transport=httpx2.MockTransport(handler), **kwargs)


# --- parsing -------------------------------------------------------------------


def test_parsing_a_recorded_page_keeps_real_urls_titles_snippets_and_engines():
    hits = parse_searxng_payload(PAGE)
    assert [h.url for h in hits][:1] == ["https://docs.python.org/3/library/asyncio-task.html"]
    assert len(hits) == 4  # the magnet link and the url-less entry are dropped
    assert hits[0].title == "Coroutines and tasks — Python 3.14.7 documentation"
    assert hits[0].snippet.startswith("TaskGroup provides stronger safety guarantees")
    assert hits[0].engines == ("brave", "google cse")
    assert hits[1].published == "2023-10-19T11:04:15"
    assert hits[0].published == ""


def test_unresponsive_engines_are_read_as_engine_and_reason():
    assert unresponsive_engines(PAGE) == [
        {"engine": "duckduckgo", "reason": "CAPTCHA"},
        {"engine": "wikidata", "reason": "Suspended: timeout"},
    ]


def test_parsing_survives_an_empty_or_odd_payload():
    assert parse_searxng_payload({}) == []
    assert parse_searxng_payload({"results": ["junk", None]}) == []
    assert unresponsive_engines({"unresponsive_engines": [[], "x"]}) == []


def test_a_result_without_an_engines_list_falls_back_to_engine():
    hits = parse_searxng_payload({"results": [{"url": "https://a.test", "engine": "yahoo"}]})
    assert hits[0].engines == ("yahoo",)


# --- paging and failure --------------------------------------------------------


async def test_requests_the_json_format_and_stops_once_it_has_enough():
    seen = []

    def handler(request):
        seen.append(dict(request.url.params))
        page = int(request.url.params["pageno"])
        return httpx2.Response(200, json=page_of(20, start=(page - 1) * 20))

    result = await backend_for(handler, max_pages=5).search("jev rerank", 30)
    assert [p["pageno"] for p in seen] == ["1", "2"]
    assert seen[0]["format"] == "json"
    assert seen[0]["q"] == "jev rerank"
    assert len(result.hits) == 40  # the whole of page 2, never padded or cut here
    assert result.usage["pages"] == 2
    assert result.usage["requests"] == 2
    assert result.usage["stopped"] == "count"


async def test_the_page_cap_bounds_a_deep_request():
    def handler(request):
        page = int(request.url.params["pageno"])
        return httpx2.Response(200, json=page_of(10, start=page * 10))

    result = await backend_for(handler, max_pages=2).search("q", 50)
    assert len(result.hits) == 20
    assert result.usage["stopped"] == "page_cap"


def test_the_page_cap_has_a_hard_ceiling():
    assert SearxngBackend(env={"SIEVE_SEARXNG_MAX_PAGES": "999"}).max_pages == 5
    assert SearxngBackend(env={"SIEVE_SEARXNG_MAX_PAGES": "nope"}).max_pages == 3


async def test_a_page_with_nothing_new_ends_the_search_short():
    result = await backend_for(lambda r: httpx2.Response(200, json=page_of(7)), max_pages=5).search("q", 50)
    assert len(result.hits) == 7
    assert result.usage["stopped"] == "exhausted"
    assert result.usage["pages"] == 2


async def test_the_deadline_stops_further_pages():
    result = await backend_for(lambda r: httpx2.Response(200, json=page_of(5)), deadline=1e-9).search("q", 50)
    assert result.usage["pages"] == 1
    assert result.usage["stopped"] == "deadline"


async def test_a_later_page_failing_keeps_the_earlier_pages():
    def handler(request):
        if request.url.params["pageno"] == "2":
            return httpx2.Response(502, text="bad gateway")
        return httpx2.Response(200, json=page_of(10))

    result = await backend_for(handler).search("q", 50)
    assert len(result.hits) == 10
    assert result.usage["stopped"] == "page_error"
    assert "HTTP 502" in result.usage["page_errors"][0]
    assert result.usage["requests"] == 2  # later pages are not retried


async def test_the_first_page_is_retried_once_on_a_server_error():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx2.Response(503, text="busy") if len(calls) == 1 else httpx2.Response(200, json=page_of(12))

    result = await backend_for(handler, max_pages=1).search("q", 10)
    assert len(calls) == 2
    assert result.usage["requests"] == 2
    assert len(result.hits) == 12


async def test_a_timeout_is_retried_once_then_reported():
    calls = []

    def handler(request):
        calls.append(1)
        raise httpx2.ReadTimeout("timed out", request=request)

    with pytest.raises(SearchBackendError, match="unreachable: ReadTimeout"):
        await backend_for(handler).search("q", 10)
    assert len(calls) == 2


async def test_a_client_error_is_not_retried():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx2.Response(429, text="slow down")

    with pytest.raises(SearchBackendError, match="HTTP 429"):
        await backend_for(handler).search("q", 10)
    assert len(calls) == 1


async def test_a_disabled_json_format_names_the_setting():
    with pytest.raises(SearchBackendError, match="search.formats"):
        await backend_for(lambda r: httpx2.Response(403, text="Forbidden")).search("q", 10)


async def test_no_results_with_every_engine_down_is_an_error_naming_them():
    payload = {"results": [], "unresponsive_engines": [["google", "CAPTCHA"], ["yahoo", "timeout"]]}
    with pytest.raises(SearchBackendError, match=r"google \(CAPTCHA\), yahoo \(timeout\)"):
        await backend_for(lambda r: httpx2.Response(200, json=payload)).search("q", 10)


async def test_partial_engine_failures_are_reported_alongside_results():
    result = await backend_for(lambda r: httpx2.Response(200, json=PAGE), max_pages=1).search("q", 10)
    assert len(result.hits) == 4
    assert {e["engine"] for e in result.usage["unresponsive_engines"]} == {"duckduckgo", "wikidata"}


async def test_an_unreachable_instance_is_a_clear_error():
    backend = SearxngBackend("http://127.0.0.1:9", timeout=2)
    with pytest.raises(SearchBackendError, match="unreachable"):
        await backend.search("q", 10)


# --- selection -----------------------------------------------------------------


def test_the_searxng_url_selects_searxng_even_with_a_brave_key():
    assert backend_name({SEARXNG_URL_ENV: "http://127.0.0.1:8888", BRAVE_API_KEY_ENV: "x"}) == "searxng"


def test_the_explicit_backend_beats_a_brave_key():
    env = {BACKEND_ENV: "searxng", BRAVE_API_KEY_ENV: "x"}
    assert backend_name(env) == "searxng"
    backend = select_backend(env={**env, SEARXNG_URL_ENV: "http://127.0.0.1:9999/"})
    assert backend.name == "searxng"
    assert backend.base_url == "http://127.0.0.1:9999"


def test_brave_is_still_chosen_without_a_searxng_url():
    assert backend_name({BRAVE_API_KEY_ENV: "x"}) == "brave"


async def test_a_failing_searxng_never_falls_back_to_brave(monkeypatch):
    async def brave_called(self, query, count):
        raise AssertionError("brave must not be called")

    monkeypatch.setattr(BraveBackend, "search", brave_called)
    monkeypatch.delenv(BACKEND_ENV, raising=False)
    monkeypatch.setenv(BRAVE_API_KEY_ENV, "x")
    monkeypatch.setenv(SEARXNG_URL_ENV, "http://127.0.0.1:9")
    monkeypatch.setenv("SIEVE_SEARXNG_TIMEOUT", "2")
    with pytest.raises(SearchBackendError, match="every searxng search variant failed"):
        await jev_search("postgres bloat vacuum", client=FakeClient())


# --- depth, dedupe, rerank -----------------------------------------------------


class PagingBackend(FakeSearchBackend):
    paginates = True


def test_depth_is_capped_and_never_below_top_k():
    assert resolve_depth(None, 10) == 30
    assert resolve_depth(500, 10) == MAX_DEPTH == 50
    assert resolve_depth(5, 20) == 20


def test_a_paging_backend_asks_the_original_query_for_the_whole_pool():
    assert variant_counts(PagingBackend(), 50, 3) == [50, 17, 17]
    assert variant_counts(FakeSearchBackend(), 50, 3) == [17, 17, 17]
    assert variant_counts(FakeSearchBackend(), 30, 3) == [10, 10, 10]  # the old 3 x 10


def test_interleave_takes_each_variants_best_hits_first():
    a = BackendResult(query="a", hits=[SearchHit(url="https://a1"), SearchHit(url="https://a2")])
    b = BackendResult(query="b", hits=[SearchHit(url="https://b1")])
    assert [h.url for h in interleave([a, b])] == ["https://a1", "https://b1", "https://a2"]


def test_dedupe_merges_engines_and_keeps_a_published_date():
    merged = dedupe_hits(
        [
            SearchHit(url="https://a.test/x", title="A", engines=("google",)),
            SearchHit(url="https://www.a.test/x/", snippet="s", engines=("yahoo", "google"), published="2026-01-01"),
        ]
    )
    assert len(merged) == 1
    assert merged[0].engines == ["google", "yahoo"]
    assert merged[0].published == "2026-01-01"
    assert merged[0].snippet == "s"


def many(n, prefix="u"):
    return [SearchHit(url=f"https://{prefix}{i}.test/", title=f"t{i}", snippet="s", engines=("google",)) for i in range(n)]


async def test_depth_and_top_k_are_separate():
    backend = PagingBackend(default=many(80))
    client = FakeClient(scorer=lambda state, i: 0.5)
    out = await jev_search("how do I plan a vegetable garden", top_k=5, depth=50, backend=backend, client=client)
    assert out["depth"] == 50
    assert out["hits_requested"] == [50, 17, 17]
    assert out["hits_found"] == 84
    assert out["candidates_scored"] == 50  # reranked pool
    assert len(out["results"]) == 5  # returned
    assert "pool_short" not in out
    assert out["results"][0]["engines"] == ["google"]


async def test_a_sparse_query_reports_a_short_pool_instead_of_padding():
    backend = PagingBackend(default=many(12))
    out = await jev_search("how do I plan a vegetable garden", top_k=10, depth=50, backend=backend, client=FakeClient())
    assert out["candidates_scored"] == 12
    assert out["pool_short"] is True


async def test_the_default_depth_keeps_the_old_request_size_for_other_backends():
    backend = FakeSearchBackend(default=many(40))
    out = await jev_search("how do I plan a vegetable garden", backend=backend, client=FakeClient())
    assert out["hits_requested"] == [10, 10, 10]
    assert out["hits_found"] == 30


# --- retrieval cache -----------------------------------------------------------


def test_the_cache_round_trips_and_expires(tmp_path):
    cache = SearchCache(tmp_path / "search", ttl=60)
    backend = FakeSearchBackend()
    key = result_key(backend, "q", 10)
    result = BackendResult(query="q", hits=[SearchHit(url="https://a.test", engines=("google",))], usage={"requests": 2})
    cache.put(key, result)
    got = cache.get(key)
    assert got.hits == result.hits and got.usage == {"requests": 2}
    cache.ttl = 1e-9
    time.sleep(0.01)
    assert cache.get(key) is None
    cache.close()


def test_a_zero_ttl_disables_the_cache(tmp_path):
    cache = SearchCache(tmp_path / "search", ttl=0)
    key = result_key(FakeSearchBackend(), "q", 10)
    cache.put(key, BackendResult(query="q", hits=[SearchHit(url="https://a.test")]))
    assert cache.get(key) is None
    cache.close()


def test_the_key_separates_backends_configs_and_counts():
    one = SearxngBackend("http://127.0.0.1:1")
    two = SearxngBackend("http://127.0.0.1:2")
    assert result_key(one, "q", 10) != result_key(two, "q", 10)
    assert result_key(one, "q", 10) != result_key(one, "q", 30)


async def test_a_repeated_search_is_served_from_the_cache(tmp_path):
    cache = SearchCache(tmp_path / "search", ttl=3600)
    backend = FakeSearchBackend(default=many(10))
    first = await jev_search("how do I plan a vegetable garden", backend=backend, client=FakeClient(), search_cache=cache)
    asked = len(backend.queries)
    second = await jev_search("how do I plan a vegetable garden", backend=backend, client=FakeClient(), search_cache=cache)
    assert len(backend.queries) == asked
    assert first["backend_requests"] == 3
    assert second["backend_requests"] == 0
    assert all(call["cached"] for call in second["backend_calls"])
    assert [r["url"] for r in second["results"]] == [r["url"] for r in first["results"]]
    cache.close()


async def test_a_failed_variant_is_not_cached(tmp_path):
    cache = SearchCache(tmp_path / "search", ttl=3600)
    query = "how do I plan a vegetable garden"
    from sieve.search import propose_variants

    failing = propose_variants(query)[1]
    backend = FakeSearchBackend(default=many(5), error_on=[failing])
    await jev_search(query, backend=backend, client=FakeClient(), search_cache=cache)
    backend.error_on = set()
    out = await jev_search(query, backend=backend, client=FakeClient(), search_cache=cache)
    assert backend.queries.count(failing) == 2
    assert out["backend_requests"] == 1
    cache.close()


def _mode(path):
    return stat.S_IMODE(os.stat(path).st_mode)


@pytest.fixture
def permissive_umask():
    previous = os.umask(0)
    yield
    os.umask(previous)


def test_a_new_cache_is_private_under_a_permissive_umask(tmp_path, permissive_umask):
    directory = tmp_path / "search"
    cache = SearchCache(directory, ttl=60)
    cache.put(result_key(FakeSearchBackend(), "private query", 10), BackendResult(query="private query", hits=[SearchHit(url="https://a.test", snippet="s")]))
    assert _mode(directory) == 0o700
    assert _mode(directory / DB_NAME) == 0o600
    cache.close()


def _permissive_cache_dir(tmp_path):
    directory = tmp_path / "search"
    directory.mkdir()
    os.chmod(directory, 0o777)
    names = (DB_NAME, f"{DB_NAME}-journal", f"{DB_NAME}-wal", f"{DB_NAME}-shm")
    for name in names:
        (directory / name).touch()
        os.chmod(directory / name, 0o666)
    os.chmod(tmp_path, 0o755)
    return directory, names


def test_existing_permissive_files_and_sidecars_are_tightened(tmp_path, permissive_umask):
    directory, names = _permissive_cache_dir(tmp_path)
    _make_private(directory)
    assert _mode(directory) == 0o700
    for name in names:
        assert _mode(directory / name) == 0o600, name
    assert _mode(tmp_path) == 0o755  # parents are left alone


def test_opening_an_existing_permissive_cache_leaves_it_private(tmp_path, permissive_umask):
    directory, _ = _permissive_cache_dir(tmp_path)
    cache = SearchCache(directory, ttl=60)
    cache.put(result_key(FakeSearchBackend(), "q", 10), BackendResult(query="q", hits=[SearchHit(url="https://a.test")]))
    assert _mode(directory) == 0o700
    leftovers = [p for p in directory.iterdir()]
    assert directory / DB_NAME in leftovers
    assert all(_mode(p) == 0o600 for p in leftovers), [(p.name, oct(_mode(p))) for p in leftovers]
    assert _mode(tmp_path) == 0o755
    cache.close()
