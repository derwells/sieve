"""The served transport: one loopback HTTP server shared by many clients."""

import asyncio
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx2
import pytest
from mcp import Client
from mcp.server.mcpserver.exceptions import ToolError

from sieve import fetch, server as server_module

ROOT = Path(__file__).resolve().parent.parent


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def served():
    """A real `sieve --http` process with no API key, torn down after the test."""
    port = _free_port()
    env = {k: v for k, v in os.environ.items() if k != "TYPESAFE_API_KEY"}
    proc = subprocess.Popen(
        [sys.executable, "-m", "sieve", "--http", "--port", str(port)],
        cwd=ROOT,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                break
        except OSError:
            time.sleep(0.1)
    else:
        proc.kill()
        pytest.fail("sieve --http did not start listening")
    try:
        yield f"http://127.0.0.1:{port}/mcp"
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def test_main_defaults_to_stdio(monkeypatch):
    calls = []
    monkeypatch.setattr(server_module.server, "run", lambda **kw: calls.append(kw))
    server_module.main([])
    assert calls == [{"transport": "stdio"}]


def test_http_binds_loopback_statelessly(monkeypatch):
    calls = []
    monkeypatch.setattr(server_module.server, "run", lambda **kw: calls.append(kw))
    monkeypatch.setattr(fetch, "CWD_RELATIVE_PATHS", True)
    server_module.main(["--http", "--port", "9999"])
    assert calls == [{
        "transport": "streamable-http",
        "host": "127.0.0.1",
        "port": 9999,
        "streamable_http_path": "/mcp",
        "stateless_http": True,
    }]
    assert fetch.CWD_RELATIVE_PATHS is False


def test_port_comes_from_the_environment(monkeypatch):
    calls = []
    monkeypatch.setattr(server_module.server, "run", lambda **kw: calls.append(kw))
    monkeypatch.setattr(fetch, "CWD_RELATIVE_PATHS", True)
    monkeypatch.setenv("SIEVE_HTTP_PORT", "8124")
    server_module.main(["--http"])
    assert calls[0]["port"] == 8124


async def test_served_mode_rejects_relative_paths(monkeypatch, tmp_path):
    monkeypatch.setattr(fetch, "CWD_RELATIVE_PATHS", False)
    with pytest.raises(ToolError, match="absolute"):
        await server_module.jev_grep(question="q", path="some/repo")
    source = await fetch.read_local("notes.md")
    assert source.error and "base_path" in source.error
    (tmp_path / "notes.md").write_text("hello", encoding="utf-8")
    source = await fetch.read_local("notes.md", base_path=str(tmp_path))
    assert source.error is None


async def test_many_clients_share_one_server(served):
    async def one(i: int) -> tuple[list[str], str]:
        async with Client(served) as client:
            tools = sorted(tool.name for tool in (await client.list_tools()).tools)
            result = await client.call_tool("jev_ask", {"question": f"q{i}", "items": ["one"], "kind": "judge"})
            assert result.is_error
            return tools, result.content[0].text

    outcomes = await asyncio.gather(*(one(i) for i in range(24)))
    for tools, error in outcomes:
        assert "jev_ask" in tools and "jev_grep" in tools
        assert "yes and no" in error


async def test_foreign_host_header_is_refused(served):
    async with httpx2.AsyncClient() as http:
        response = await http.post(
            served,
            headers={"Host": "evil.example", "Content-Type": "application/json", "Accept": "application/json, text/event-stream"},
            content=b"{}",
        )
    assert response.status_code in (400, 403, 421)
