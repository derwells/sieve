"""The MCP server exposes its tools with the contracted schemas."""

import pytest

from sieve.client import API_KEY_ENV, build_client
from sieve.errors import MissingAPIKeyError
from sieve.server import server


async def test_tools_are_listed():
    tools = await server.list_tools()
    assert sorted(tool.name for tool in tools) == ["jev_ask", "jev_grep", "jev_rank", "jev_route", "jev_search", "jev_triage_paseo", "jev_triage_threads", "jev_verify"]
    assert tools[0].name == "jev_ask", "the general tool is listed before the shortcuts"


async def test_jev_ask_schema_matches_the_contract():
    tool = next(t for t in await server.list_tools() if t.name == "jev_ask")
    schema = tool.input_schema
    assert set(schema["required"]) == {"question", "items"}
    properties = schema["properties"]
    assert properties["kind"]["default"] == "judge"
    assert properties["max_chars"]["default"] == 4000
    assert properties["budget_usd"]["default"] == 0.50
    assert properties["items"]["type"] == "array"
    assert {"yes", "no", "options", "levels"} <= set(properties)


async def test_jev_ask_offers_every_primitive():
    tool = next(t for t in await server.list_tools() if t.name == "jev_ask")
    kind = tool.input_schema["properties"]["kind"]
    enum = kind.get("enum") or kind["anyOf"][0]["enum"]
    assert sorted(enum) == ["choose", "judge", "score"]


async def test_jev_ask_reports_a_bad_answer_space_as_a_tool_error():
    from sieve.server import jev_ask

    with pytest.raises(ValueError, match="yes and no"):
        await jev_ask(question="q", items=["one"], kind="judge")


async def test_jev_route_schema_matches_the_contract():
    tool = next(t for t in await server.list_tools() if t.name == "jev_route")
    schema = tool.input_schema
    assert set(schema["required"]) == {"ask", "routes"}
    assert schema["properties"]["budget_usd"]["default"] == 0.10
    assert schema["properties"]["routes"]["type"] == "array"


async def test_triage_schemas():
    tools = {tool.name: tool for tool in await server.list_tools()}
    threads = tools["jev_triage_threads"].input_schema
    paseo = tools["jev_triage_paseo"].input_schema
    assert set(threads["required"]) == {"threads"}
    assert set(paseo["required"]) == {"agent_ids"}
    assert threads["properties"]["budget_usd"]["default"] == 0.50
    assert paseo["properties"]["tail"]["default"] == 400
    assert "logs" not in paseo["properties"]


async def test_jev_verify_schema_matches_the_contract():
    tool = next(t for t in await server.list_tools() if t.name == "jev_verify")
    schema = tool.input_schema
    assert not schema.get("required")
    properties = schema["properties"]
    assert properties["budget_usd"]["default"] == 0.50
    assert properties["support_threshold"]["default"] == 0.6
    assert properties["contradict_threshold"]["default"] == 0.5
    assert "records" in properties and "report" in properties and "base_path" in properties


async def test_jev_grep_schema_matches_the_contract():
    tool = next(t for t in await server.list_tools() if t.name == "jev_grep")
    schema = tool.input_schema
    assert set(schema["required"]) == {"question", "path"}
    properties = schema["properties"]
    assert properties["mode"]["default"] == "files"
    assert properties["top_k"]["default"] == 20
    assert properties["threshold"]["default"] == 0.5
    assert properties["budget_usd"]["default"] == 0.50


async def test_jev_rank_schema_matches_the_contract():
    tool = next(t for t in await server.list_tools() if t.name == "jev_rank")
    schema = tool.input_schema
    assert set(schema["required"]) == {"question", "candidates"}
    assert schema["properties"]["threshold"]["default"] == 0.0
    assert schema["properties"]["top_k"]["default"] is None


async def test_jev_search_schema_matches_the_contract():
    tool = next(t for t in await server.list_tools() if t.name == "jev_search")
    schema = tool.input_schema
    assert set(schema["required"]) == {"query"}
    properties = schema["properties"]
    assert properties["top_k"]["default"] == 10
    assert properties["variants"]["default"] == 3
    assert properties["variants"]["minimum"] == 2
    assert properties["variants"]["maximum"] == 4


def test_a_missing_api_key_is_a_clear_error(monkeypatch):
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    with pytest.raises(MissingAPIKeyError, match=API_KEY_ENV):
        build_client()


def test_the_retry_policy_covers_the_documented_transient_statuses():
    from sieve.client import RETRY_POLICY

    assert {408, 429, 500, 502, 503, 529} <= RETRY_POLICY.http_statuses
    assert RETRY_POLICY.max_retries >= 1


async def test_jev_search_separates_depth_from_top_k():
    tool = next(t for t in await server.list_tools() if t.name == "jev_search")
    properties = tool.input_schema["properties"]
    assert properties["top_k"]["default"] == 10
    assert properties["depth"]["default"] == 30
    assert properties["depth"]["maximum"] == 50
