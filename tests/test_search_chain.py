"""The search chain: route order, cache, provenance, environment isolation, CLI. Offline."""

import json
import pathlib

import pytest

from sieve import search_chain
from sieve.backends import BackendResult, ClaudeSearchBackend, CodexSearchBackend, SearchBackendError, SearchHit, SearxngBackend
from sieve.backends.base import cli_environment
from sieve.search_chain import apreflight, asearch, cache_key, parse_routes

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


class FakeRoute:
    """A backend that answers from a list, fails on demand, and counts its calls."""

    def __init__(self, name, hits=None, error=None, provenance="observed", model="m"):
        self.name = name
        self.hits = hits if hits is not None else [SearchHit(url=f"https://{name}.test/{i}", title=f"{name} {i}", snippet="s") for i in range(5)]
        self.error = error
        self.provenance = provenance
        self.model = model
        self.calls = 0

    def config(self):
        return {"route": self.name, "model": self.model, "effort": None, "cmd_fingerprint": "f", "url": None}

    async def search(self, query, count):
        self.calls += 1
        if self.error:
            raise SearchBackendError(self.error)
        return BackendResult(query=query, hits=self.hits[:count], usage={"provenance": self.provenance})


def fakes(**overrides):
    routes = {name: FakeRoute(name) for name in ("codex", "claude", "searxng")}
    routes.update(overrides)
    return routes


async def test_the_first_healthy_route_answers_and_later_routes_never_run(tmp_path):
    backends = fakes()
    out = await asearch("q", 3, ["codex", "claude", "searxng"], tmp_path, backends=backends)
    assert out["route"] == "codex"
    assert out["cache"] == "miss"
    assert out["degraded"] is False and out["fallback_from"] is None and out["errors"] == {}
    assert out["observed"] is True and out["transcribed"] is False
    assert [h["rank"] for h in out["hits"]] == [1, 2, 3]
    assert set(out["hits"][0]) == {"rank", "title", "url", "snippet"}
    assert out["config"]["route"] == "codex"
    assert backends["claude"].calls == backends["searxng"].calls == 0


async def test_a_repeat_is_a_cache_hit_on_the_same_route(tmp_path):
    backends = fakes()
    await asearch("q", 3, None, tmp_path, backends=backends)
    out = await asearch("  Q ", 3, None, tmp_path, backends=backends)
    assert out["cache"] == "hit" and out["route"] == "codex"
    assert backends["codex"].calls == 1


async def test_a_failed_hosted_route_falls_back_and_says_so(tmp_path):
    backends = fakes(codex=FakeRoute("codex", error="codex search exited 1: quota"),
                     claude=FakeRoute("claude", error="claude search timed out after 90s"))
    out = await asearch("q", 3, None, tmp_path, backends=backends)
    assert out["route"] == "searxng"
    assert out["degraded"] is True
    assert out["fallback_from"] == "codex"
    assert out["errors"] == {"codex": "codex search exited 1: quota", "claude": "claude search timed out after 90s"}


async def test_a_later_routes_cache_is_never_served_while_an_earlier_route_works(tmp_path):
    backends = fakes()
    await asearch("q", 3, ["searxng"], tmp_path, backends=backends)  # searxng result now cached
    out = await asearch("q", 3, ["codex", "searxng"], tmp_path, backends=backends)
    assert out["route"] == "codex" and out["cache"] == "miss"


async def test_a_later_routes_cache_serves_once_the_earlier_route_fails(tmp_path):
    backends = fakes()
    await asearch("q", 3, ["searxng"], tmp_path, backends=backends)
    backends["codex"].error = "down"
    out = await asearch("q", 3, ["codex", "searxng"], tmp_path, backends=backends)
    assert out["route"] == "searxng" and out["cache"] == "hit" and out["degraded"] is True
    assert backends["searxng"].calls == 1


async def test_an_explicit_route_order_is_honoured(tmp_path):
    backends = fakes()
    out = await asearch("q", 3, "searxng,codex", tmp_path, backends=backends)
    assert out["route"] == "searxng"
    assert out["degraded"] is False  # the first listed route answered
    assert backends["codex"].calls == 0


async def test_empty_results_count_as_a_failure(tmp_path):
    backends = fakes(codex=FakeRoute("codex", hits=[SearchHit(url="magnet:x", title="t")]))
    out = await asearch("q", 3, ["codex", "claude"], tmp_path, backends=backends)
    assert out["route"] == "claude"
    assert out["errors"]["codex"] == "no http(s) results"


async def test_every_route_failing_returns_no_hits_and_every_reason(tmp_path):
    backends = {n: FakeRoute(n, error=f"{n} broke") for n in ("codex", "claude", "searxng")}
    out = await asearch("q", 3, None, tmp_path, backends=backends)
    assert out["hits"] == [] and out["route"] is None and out["config"] is None
    assert out["errors"] == {"codex": "codex broke", "claude": "claude broke", "searxng": "searxng broke"}
    assert out["degraded"] is True


async def test_transcribed_hits_are_flagged(tmp_path):
    backends = fakes(codex=FakeRoute("codex", provenance="transcribed"))
    out = await asearch("q", 3, ["codex"], tmp_path, backends=backends)
    assert out["observed"] is False and out["transcribed"] is True


async def test_engines_are_kept_as_provenance(tmp_path):
    hit = SearchHit(url="https://a.test", title="A", snippet="s", engines=("google", "yahoo"))
    out = await asearch("q", 3, ["searxng"], tmp_path, backends={"searxng": FakeRoute("searxng", hits=[hit])})
    assert out["hits"][0]["engines"] == ["google", "yahoo"]


def test_the_cache_key_covers_route_config_query_and_count():
    base = {"route": "codex", "model": "gpt-6-luna", "effort": "low", "cmd_fingerprint": "abc", "url": None}
    key = cache_key(base, "Jev  rerank", 10)
    assert key == cache_key(base, "jev rerank", 10)
    assert key != cache_key(base, "jev rerank", 20)
    for field, value in (("route", "claude"), ("model", "gpt-6-astra"), ("effort", "high"), ("cmd_fingerprint", "zzz"), ("url", "http://x")):
        assert key != cache_key({**base, field: value}, "jev rerank", 10), field


def test_routes_are_validated():
    assert parse_routes(None) == ["codex", "claude", "searxng"]
    assert parse_routes(" claude , searxng ") == ["claude", "searxng"]
    with pytest.raises(ValueError, match="unknown"):
        parse_routes("codex,bing")
    with pytest.raises(ValueError, match="twice"):
        parse_routes("codex,codex")


# --- preflight -----------------------------------------------------------------


async def test_preflight_passes_only_observed_titled_http_results(tmp_path):
    backends = {
        "codex": FakeRoute("codex", provenance="transcribed"),
        "claude": FakeRoute("claude", hits=[SearchHit(url="https://a.test", title="  ")]),
        "searxng": FakeRoute("searxng"),
    }
    out = await apreflight("q", None, tmp_path, backends=backends)
    routes = out["routes"]
    assert routes["codex"]["ok"] is False and "transcribed" in routes["codex"]["error"]
    assert routes["claude"]["ok"] is False and routes["claude"]["error"] == "no http(s) result with a title"
    assert routes["searxng"]["ok"] is True and routes["searxng"]["results"] == 5 and routes["searxng"]["observed"]
    assert set(routes["searxng"]) == {"ok", "results", "observed", "error", "seconds", "config"}


async def test_preflight_reports_a_failure_and_never_reads_the_cache(tmp_path):
    backends = fakes()
    await asearch("q", 10, ["codex"], tmp_path, backends=backends)
    backends["codex"].error = "exited 2"
    out = await apreflight("q", ["codex"], tmp_path, backends=backends)
    assert out["routes"]["codex"] == {**out["routes"]["codex"], "ok": False, "error": "exited 2"}


async def test_a_passing_preflight_warms_the_cache(tmp_path):
    backends = fakes()
    await apreflight("q", ["claude"], tmp_path, backends=backends, count=10)
    out = await asearch("q", 10, ["claude"], tmp_path, backends=backends)
    assert out["cache"] == "hit" and backends["claude"].calls == 1


# --- real backends behind the chain --------------------------------------------


def runner_returning(stdout, record=None):
    async def run(argv, *, cwd, env, timeout):
        if record is not None:
            record.append({"argv": argv, "env": env})
        return 0, stdout, ""

    return run


async def test_codex_0_156_events_are_observed_through_the_chain(tmp_path):
    backend = CodexSearchBackend(runner=runner_returning((FIXTURES / "codex_stream_0.156.1.jsonl").read_text()), env={})
    out = await asearch("jev rerank", 5, ["codex"], tmp_path, backends={"codex": backend})
    assert out["observed"] is True and out["transcribed"] is False
    assert all(h["snippet"] for h in out["hits"])


async def test_an_older_codex_stream_is_transcribed_and_fails_preflight(tmp_path):
    backend = CodexSearchBackend(runner=runner_returning((FIXTURES / "codex_stream.jsonl").read_text()), env={})
    out = await asearch("jev rerank", 5, ["codex"], tmp_path, backends={"codex": backend})
    assert out["transcribed"] is True and out["observed"] is False
    check = await apreflight("jev rerank", ["codex"], tmp_path / "p", backends={"codex": backend})
    assert check["routes"]["codex"]["ok"] is False


# --- configuration and environment --------------------------------------------

ROUTED_ENV = {
    "PATH": "/usr/bin:/bin",
    "HOME": "/home/someone",
    "CLAUDE_CODE_OAUTH_TOKEN": "login",
    "ANTHROPIC_BASE_URL": "https://third-party.example/anthropic",
    "ANTHROPIC_AUTH_TOKEN": "tp-x",
    "ANTHROPIC_API_KEY": "sk-x",
    "ANTHROPIC_MODEL": "other-flash",
    "ANTHROPIC_DEFAULT_OPUS_MODEL": "other-pro",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "other-flash",
    "ANTHROPIC_SMALL_FAST_MODEL": "other-flash",
    "CLAUDE_CODE_SUBAGENT_MODEL": "other-pro",
    "CLAUDE_CODE_MAX_CONTEXT_TOKENS": "1000000",
    "CLAUDE_CODE_USE_BEDROCK": "0",
    "CLAUDE_CODE_USE_VERTEX": "0",
    "API_TIMEOUT_MS": "3000000",
    "WRAPPER_PLAN_URL": "https://third-party.example",
    "WRAPPER_MODEL": "other-pro",
    "PROVIDER_API_KEY": "tp-x",
    "SIEVE_STRIP_ENV_PREFIXES": "WRAPPER_, PROVIDER_",
    "OPENAI_BASE_URL": "https://third-party.example/v1",
    "OPENAI_API_KEY": "sk-y",
    "CLAUDECODE": "1",
}


def test_routing_overrides_are_stripped_and_the_rest_kept():
    child = cli_environment(ROUTED_ENV)
    assert child == {
        "PATH": "/usr/bin:/bin",
        "HOME": "/home/someone",
        "CLAUDE_CODE_OAUTH_TOKEN": "login",
        "SIEVE_STRIP_ENV_PREFIXES": "WRAPPER_, PROVIDER_",
    }


def test_extra_prefixes_are_stripped_only_when_configured():
    assert "WRAPPER_MODEL" in cli_environment({"WRAPPER_MODEL": "x"})


@pytest.mark.parametrize("backend_class", [CodexSearchBackend, ClaudeSearchBackend])
async def test_cli_children_get_the_stripped_environment(backend_class):
    record = []
    stream = (FIXTURES / ("codex_stream_0.156.1.jsonl" if backend_class is CodexSearchBackend else "claude_stream.jsonl")).read_text()
    await backend_class(runner=runner_returning(stream, record), env=ROUTED_ENV).search("q", 3)
    assert set(record[0]["env"]) == {"PATH", "HOME", "CLAUDE_CODE_OAUTH_TOKEN", "SIEVE_STRIP_ENV_PREFIXES"}


def test_cheap_defaults_and_their_overrides():
    codex = search_chain.build_backend("codex", {})
    argv = codex.argv_template()
    assert argv[argv.index("-m") + 1] == "gpt-6-luna"
    assert "model_reasoning_effort=low" in argv
    assert argv[argv.index("-s") + 1] == "read-only"
    assert {"--ephemeral", "--json", "tools.web_search=true"} <= set(argv)
    assert codex.config() | {"cmd_fingerprint": None} == {"route": "codex", "model": "gpt-6-luna", "effort": "low", "cmd_fingerprint": None, "url": None}
    claude = search_chain.build_backend("claude", {})
    assert claude.config()["model"] == "claude-haiku-4-5-20251001"
    assert "claude-haiku-4-5-20251001" in claude.argv_template()

    chosen = search_chain.build_backend("codex", {"SIEVE_CODEX_SEARCH_MODEL": "gpt-6-astra", "SIEVE_CODEX_SEARCH_EFFORT": "medium"})
    assert chosen.config()["model"] == "gpt-6-astra" and chosen.config()["effort"] == "medium"
    assert chosen.config()["cmd_fingerprint"] != codex.config()["cmd_fingerprint"]
    assert search_chain.build_backend("claude", {"SIEVE_CLAUDE_SEARCH_MODEL": "claude-sonnet-5"}).config()["model"] == "claude-sonnet-5"
    assert search_chain.build_backend("searxng", {"SIEVE_SEARXNG_URL": "http://127.0.0.1:9999"}).config()["url"] == "http://127.0.0.1:9999"


def test_a_per_route_command_replaces_the_line_and_the_generic_one_is_ignored():
    custom = search_chain.build_backend("codex", {"SIEVE_SEARCH_CMD_CODEX": "codex exec -m my-model --json"})
    assert custom.argv_template() == ["codex", "exec", "-m", "my-model", "--json"]
    assert custom.config()["model"] is None
    generic = search_chain.build_backend("claude", {"SIEVE_SEARCH_CMD": "codex exec --json"})
    assert generic.argv_template()[0] == "claude"  # a single-backend override must not leak into the chain


# --- CLI -----------------------------------------------------------------------


def test_cli_search_prints_the_json_result(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(search_chain, "build_backend", lambda route, env=None: FakeRoute(route, error="down") if route == "codex" else FakeRoute(route))
    code = search_chain.main(["search", "--routes", "codex,searxng", "--count", "2", "--cache-dir", str(tmp_path), "--json", "jev", "rerank"])
    out = json.loads(capsys.readouterr().out)
    assert code == 0
    assert out["route"] == "searxng" and out["degraded"] is True and len(out["hits"]) == 2
    assert set(out) == {"hits", "route", "fallback_from", "errors", "degraded", "observed", "transcribed", "config", "cache", "seconds"}


def test_cli_exits_non_zero_when_nothing_answers(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(search_chain, "build_backend", lambda route, env=None: FakeRoute(route, error="down"))
    assert search_chain.main(["preflight", "--routes", "searxng", "--cache-dir", str(tmp_path), "--json", "q"]) == 1
    assert json.loads(capsys.readouterr().out)["routes"]["searxng"]["ok"] is False


def test_cli_rejects_an_unknown_route(capsys):
    with pytest.raises(SystemExit):
        search_chain.main(["search", "--routes", "bing", "q"])
