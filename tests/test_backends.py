"""Search backends: parsing, selection, and failure modes. No network, no CLI."""

import json
import pathlib

import httpx2
import pytest

from sieve.backends import (
    BACKEND_ENV,
    BraveBackend,
    ClaudeSearchBackend,
    CodexSearchBackend,
    SearchBackendError,
    backend_name,
    run_cli,
    select_backend,
)
from sieve.backends.base import TIMEOUT_ENV, command_override, timeout_seconds
from sieve.backends.brave import API_KEY_ENV as BRAVE_KEY, parse_brave_payload
from sieve.backends.claude_cli import BASE_COMMAND as CLAUDE_COMMAND
from sieve.backends.claude_cli import build_command, parse_claude_stream
from sieve.backends.codex_cli import BASE_COMMAND as CODEX_COMMAND
from sieve.backends.codex_cli import parse_codex_stream, parse_markdown_links

FIXTURES = pathlib.Path(__file__).parent / "fixtures"
CLAUDE_STREAM = (FIXTURES / "claude_stream.jsonl").read_text()
CODEX_STREAM = (FIXTURES / "codex_stream.jsonl").read_text()


def fake_runner(stdout="", stderr="", code=0, record=None):
    async def run(argv, *, cwd, env, timeout):
        if record is not None:
            record.append({"argv": argv, "cwd": cwd, "env": env, "timeout": timeout})
        return code, stdout, stderr

    return run


# --- brave -------------------------------------------------------------------

BRAVE_PAYLOAD = {
    "web": {
        "results": [
            {"url": "https://docs.typesafe.ai/cookbooks/rerank_typesafe", "title": "Rerank", "description": "how to"},
            {"url": "https://example.com/b", "title": "B"},
            {"title": "no url at all"},
        ]
    }
}


def test_brave_parsing_maps_description_to_snippet():
    hits = parse_brave_payload(BRAVE_PAYLOAD)
    assert [h.url for h in hits] == ["https://docs.typesafe.ai/cookbooks/rerank_typesafe", "https://example.com/b"]
    assert hits[0].snippet == "how to"
    assert hits[1].snippet == ""


def test_brave_parsing_survives_an_empty_payload():
    assert parse_brave_payload({}) == []


async def test_brave_sends_the_key_and_count_and_parses_the_response():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["token"] = request.headers.get("X-Subscription-Token")
        seen["accept"] = request.headers.get("Accept")
        return httpx2.Response(200, json=BRAVE_PAYLOAD)

    backend = BraveBackend("test-key", transport=httpx2.MockTransport(handler))
    result = await backend.search("jev rerank", 50)
    assert seen["token"] == "test-key"
    assert seen["accept"] == "application/json"
    assert "q=jev+rerank" in seen["url"]
    assert "count=20" in seen["url"]  # capped at the Brave maximum
    assert len(result.hits) == 2
    assert result.usage["results"] == 2
    assert result.wall_seconds >= 0


async def test_brave_raises_on_a_non_200():
    backend = BraveBackend("k", transport=httpx2.MockTransport(lambda r: httpx2.Response(429, text="slow down")))
    with pytest.raises(SearchBackendError, match="HTTP 429"):
        await backend.search("q", 5)


async def test_brave_without_a_key_is_a_clear_error(monkeypatch):
    monkeypatch.delenv(BRAVE_KEY, raising=False)
    with pytest.raises(SearchBackendError, match=BRAVE_KEY):
        await BraveBackend().search("q", 5)


# --- claude ------------------------------------------------------------------


def test_claude_stream_parsing_reads_the_tool_use_result():
    hits, usage = parse_claude_stream(CLAUDE_STREAM)
    assert len(hits) == 5
    assert hits[0].url.startswith("https://")
    assert hits[0].title
    assert all(h.snippet == "" for h in hits)  # WebSearch has no snippet field
    assert usage["searches"] == 1
    assert usage["total_cost_usd"] == pytest.approx(0.0541333)
    assert usage["duration_ms"] == 13124


def test_claude_reports_zero_web_search_requests_despite_results():
    """Recorded so nobody counts searches off server_tool_use again."""
    result = [json.loads(line) for line in CLAUDE_STREAM.splitlines() if line][-1]
    assert result["usage"]["server_tool_use"]["web_search_requests"] == 0


def test_claude_stream_falls_back_to_the_links_text():
    stripped = []
    for line in CLAUDE_STREAM.splitlines():
        event = json.loads(line)
        if event.get("type") == "user":
            event.pop("tool_use_result", None)
        stripped.append(json.dumps(event))
    hits, usage = parse_claude_stream("\n".join(stripped))
    assert len(hits) == 5
    assert hits[0].url.startswith("https://")
    assert usage["searches"] == 1


def test_claude_stream_ignores_junk_lines():
    assert parse_claude_stream("not json\n\n[]\n")[0] == []


def test_claude_command_carries_the_stripping_flags():
    argv = build_command("do a search")
    assert argv[0] == "claude"
    assert argv[-1] == "do a search"
    for flag in ("--verbose", "--strict-mcp-config", "--setting-sources", "--max-turns"):
        assert flag in argv
    assert argv[argv.index("--output-format") + 1] == "stream-json"
    assert argv[argv.index("--allowedTools") + 1] == "WebSearch"
    assert argv[argv.index("--mcp-config") + 1] == '{"mcpServers":{}}'
    assert CLAUDE_COMMAND[0] == "claude"


def test_an_override_replaces_the_whole_command_line():
    assert build_command("p", ["my-cli", "--flag"]) == ["my-cli", "--flag", "p"]


async def test_claude_backend_runs_in_an_empty_dir_without_claudecode(monkeypatch):
    monkeypatch.setenv("CLAUDECODE", "1")
    record = []
    backend = ClaudeSearchBackend(runner=fake_runner(CLAUDE_STREAM, record=record))
    result = await backend.search("jev rerank", 3)
    assert len(result.hits) == 3
    assert "CLAUDECODE" not in record[0]["env"]
    assert record[0]["cwd"] != str(pathlib.Path.cwd())
    assert not list(pathlib.Path(record[0]["cwd"]).iterdir()) if pathlib.Path(record[0]["cwd"]).exists() else True
    assert "jev rerank" in record[0]["argv"][-1]


async def test_claude_backend_reports_a_non_zero_exit_with_the_stderr_tail():
    backend = ClaudeSearchBackend(runner=fake_runner("", "boom: credit balance too low", code=2))
    with pytest.raises(SearchBackendError, match="credit balance too low"):
        await backend.search("q", 5)


async def test_claude_backend_errors_when_no_results_came_back():
    backend = ClaudeSearchBackend(runner=fake_runner("{}\n"))
    with pytest.raises(SearchBackendError, match="no WebSearch results"):
        await backend.search("q", 5)


# --- codex -------------------------------------------------------------------


def test_codex_markdown_parsing():
    hits = parse_markdown_links(
        "- [First](https://a.test/one)\n"
        "* [Second](https://b.test/two)\n"
        "- https://c.test/three\n"
        "not a bullet https://d.test/four\n"
    )
    assert [h.url for h in hits] == ["https://a.test/one", "https://b.test/two", "https://c.test/three"]
    assert hits[0].title == "First"
    assert hits[2].title == ""


def test_codex_stream_parsing_uses_the_agent_message_and_turn_usage():
    hits, usage = parse_codex_stream(CODEX_STREAM)
    assert len(hits) == 6
    assert hits[0].url == "https://frutik.github.io/awesome-search/Articles/TypeSafe-Cookbook---Re-ranking"
    assert usage["input_tokens"] == 45846
    assert usage["searches"] == 1


def test_codex_web_search_event_carries_no_results():
    """The reason this backend has to read the model's prose."""
    events = [json.loads(line) for line in CODEX_STREAM.splitlines() if line]
    search = next(e["item"] for e in events if e.get("item", {}).get("type") == "web_search")
    assert set(search) == {"id", "type", "query", "action"}


def test_codex_command_enables_web_search():
    assert CODEX_COMMAND[:2] == ["codex", "exec"]
    assert "tools.web_search=true" in CODEX_COMMAND
    assert "--search" not in CODEX_COMMAND  # codex exec has no such flag
    assert "--json" in CODEX_COMMAND


async def test_codex_backend_parses_a_recorded_stream():
    backend = CodexSearchBackend(runner=fake_runner(CODEX_STREAM))
    result = await backend.search("jev rerank", 2)
    assert len(result.hits) == 2
    assert all(h.snippet == "" for h in result.hits)


async def test_codex_backend_errors_when_nothing_parses():
    backend = CodexSearchBackend(runner=fake_runner('{"type":"turn.completed","usage":{}}\n'))
    with pytest.raises(SearchBackendError, match="no parseable links"):
        await backend.search("q", 5)


# --- selection and shared config ---------------------------------------------


def test_the_env_var_picks_the_backend():
    assert backend_name({BACKEND_ENV: "codex", BRAVE_KEY: "x"}) == "codex"
    assert select_backend(env={BACKEND_ENV: "claude"}).name == "claude"


def test_brave_is_automatic_when_its_key_is_present():
    assert backend_name({BRAVE_KEY: "x"}) == "brave"


def test_claude_is_the_fallback():
    assert backend_name({}) == "claude"


def test_an_unknown_backend_is_rejected():
    with pytest.raises(SearchBackendError, match="not a backend"):
        backend_name({BACKEND_ENV: "bing"})


def test_the_timeout_comes_from_the_env_with_a_sane_default():
    assert timeout_seconds({}) == 90.0
    assert timeout_seconds({TIMEOUT_ENV: "15"}) == 15.0
    assert timeout_seconds({TIMEOUT_ENV: "nonsense"}) == 90.0
    assert timeout_seconds({TIMEOUT_ENV: "-3"}) == 90.0


def test_the_command_override_is_split_with_shlex():
    assert command_override({"SIEVE_SEARCH_CMD": "my-cli --flag 'two words'"}) == [
        "my-cli",
        "--flag",
        "two words",
    ]
    assert command_override({}) is None


async def test_a_hanging_cli_is_killed_and_reported(tmp_path):
    """The real run_cli path: a process that outlives the timeout is killed."""
    with pytest.raises(SearchBackendError, match="timed out after"):
        await run_cli(["sleep", "30"], cwd=str(tmp_path), env={}, timeout=0.2)


async def test_run_cli_returns_the_exit_code_and_streams(tmp_path):
    code, stdout, stderr = await run_cli(
        ["sh", "-c", "echo out; echo err >&2; exit 3"], cwd=str(tmp_path), env={}, timeout=10
    )
    assert (code, stdout.strip(), stderr.strip()) == (3, "out", "err")
