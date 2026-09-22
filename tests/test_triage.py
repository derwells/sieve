"""Offline thread triage with synthetic messages."""

import json
from types import SimpleNamespace

import sieve.paseo_adapter as adapter

from sieve.paseo_adapter import parse_claude, parse_codex, parse_paseo_logs, resolve_agent
from sieve.triage import dialogue_windows, enumerate_candidates, extract_acceptance, jev_triage_threads
from .conftest import FakeClient


def ev(n, role, text, kind="text"):
    return {"id": str(n), "ts": f"2026-01-01T00:00:{n:02d}Z", "role": role, "kind": kind, "text": text}


def fake(state, index):
    item = state["items"][index]
    if "later_dialogue" in item:
        return 0.95 if any("Postgres" in t["text"] for t in item["later_dialogue"]) else 0.05
    if "later_assistant" in item:
        return 0.9 if any("withdraw" in t["text"] for t in item["later_assistant"]) else 0.05
    if "latest_assistant_text" in item:
        return 0.9 if "waiting" in item["latest_assistant_text"] or "Should I" in item["latest_assistant_text"] else 0.05
    if "acceptance_text" in item:
        return 0.9 if "one action remains" in str(item["latest_progress"]).lower() else 0.05
    return 0.9


def test_enumerates_questions_imperatives_lists_but_not_quotes_or_code():
    text = """Choose a database. Should I use Postgres or SQLite?
- Confirm the region.
- Which tier?
> Approve the quoted plan?
```text
Tell me a secret?
```
The migration is complete."""
    spans = enumerate_candidates([ev(1, "assistant", text), ev(2, "human", "Which tier?"), ev(3, "assistant", "hidden?", "thinking")])
    assert [s["span_text"] for s in spans] == ["Choose a database.", "Should I use Postgres or SQLite?", "Confirm the region.", "Which tier?"]
    assert all(s["event_id"] == "1" and s["context"] for s in spans)


def test_windows_overlap_and_cover_every_turn():
    turns = [ev(i, "human", f"turn {i} " + "x" * 40) for i in range(10)]
    windows = dialogue_windows(turns, limit=35, overlap=12)
    assert len(windows) > 1
    assert {e["id"] for w in windows for e in w} == {e["id"] for e in turns}
    assert set(e["id"] for e in windows[0]) & set(e["id"] for e in windows[1])


def test_acceptance_lines():
    assert extract_acceptance({"first_prompt": "Do work.\nAcceptance: green tests", "human_amendments": ["Must run locally", "Thanks"]}) == "Acceptance: green tests\nMust run locally"
    assert extract_acceptance({"first_prompt": "Do work", "human_amendments": []}) == ""


async def test_answered_window_reconciles_max_and_keeps_evidence():
    events = [ev(1, "assistant", "Should I use Postgres or SQLite?")]
    events += [ev(i + 2, "human", ("Postgres" if i == 8 else "noise") + " x" * 1000) for i in range(10)]
    out = await jev_triage_threads([{"thread_id": "a", "events": events, "contract": {"title": "DB", "first_prompt": "Set up a database", "human_amendments": []}}], client=FakeClient(fake))
    req = out["threads"][0]["requests"][0]
    assert req["probabilities"]["answered"] == 0.95
    assert "10" in req["evidence_event_ids"]["answered"]
    assert req["coverage"]["windows"] > 1
    assert req["coverage"]["human_turns_seen"] == 10
    assert req["resolution"] == "answered"


async def test_unknown_vs_unanswered_and_block_rule():
    thread = {"thread_id": "a", "events": [ev(1, "assistant", "Should I use Postgres or SQLite? Waiting for your choice.")], "contract": {"title": "DB", "first_prompt": "Set up a database", "human_amendments": []}}
    out = await jev_triage_threads([thread], client=FakeClient(fake))
    row = out["threads"][0]
    assert row["bucket"] == "blocked_on_derick"
    assert row["requests"][0]["resolution"] == "unanswered"
    assert row["requests"][0]["probabilities"]["answered"] is None
    thread["events"].insert(0, {**ev(0, "tool", "omitted", "system"), "id": "TRUNCATION_MARKER"})
    row = (await jev_triage_threads([thread], client=FakeClient(fake)))["threads"][0]
    assert row["requests"][0]["resolution"] == "unknown"
    assert row["bucket"] == "unknown"


async def test_withdrawn_running_and_one_step_buckets():
    base = {"thread_id": "a", "contract": {"title": "DB", "first_prompt": "Acceptance: database ready", "human_amendments": []}}
    events = [ev(1, "assistant", "Should I use Postgres or SQLite?"), ev(2, "assistant", "I withdraw that request. One action remains.")]
    out = await jev_triage_threads([{**base, "events": events}], client=FakeClient(fake))
    row = out["threads"][0]
    assert row["requests"][0]["resolution"] == "withdrawn"
    assert row["bucket"] == "one_step_left"
    events = [ev(1, "assistant", "Should I use Postgres or SQLite?"), ev(2, "assistant", "Continuing work. Waiting for your choice.")]
    out = await jev_triage_threads([{**base, "events": events, "status": "running"}], client=FakeClient(fake))
    assert not out["threads"][0]["requests"][0]["blocked_on_derick"]


def test_native_parsers_and_text_fallback(tmp_path):
    claude = tmp_path / "claude.jsonl"
    claude.write_text("\n".join(json.dumps(x) for x in [
        {"type": "user", "uuid": "u", "timestamp": "t1", "message": {"role": "user", "content": [{"type": "text", "text": "Start"}]}},
        {"type": "assistant", "uuid": "a", "timestamp": "t2", "message": {"role": "assistant", "content": [{"type": "text", "text": "Choose one?"}, {"type": "tool_use", "name": "Read", "input": {}}]}},
        {"type": "user", "uuid": "r", "timestamp": "t3", "message": {"role": "user", "content": [{"type": "tool_result", "content": "noise"}]}},
    ]))
    assert [(e["role"], e["kind"]) for e in parse_claude(claude)] == [("human", "text"), ("assistant", "text"), ("assistant", "tool_use"), ("tool", "tool_result")]
    codex = tmp_path / "codex.jsonl"
    codex.write_text("\n".join(json.dumps(x) for x in [
        {"type": "response_item", "ordinal": 1, "timestamp": "t1", "payload": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "<environment_context>hidden"}, {"type": "input_text", "text": "Start"}]}},
        {"type": "response_item", "ordinal": 2, "timestamp": "t2", "payload": {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Confirm?"}]}},
        {"type": "response_item", "ordinal": 3, "timestamp": "t3", "payload": {"type": "custom_tool_call_output", "output": "noise"}},
    ]))
    assert [(e["role"], e["kind"]) for e in parse_codex(codex)] == [("tool", "system"), ("human", "text"), ("assistant", "text"), ("tool", "tool_result")]
    assert [(e["role"], e["text"]) for e in parse_paseo_logs("[t1] user: Start\n[t2] assistant: Confirm?\n  more\n[t3] tool: noise")] == [("human", "Start"), ("assistant", "Confirm?\n  more"), ("tool", "noise")]


def test_adapter_resolves_native_and_lossy_fallback(tmp_path, monkeypatch):
    agents = tmp_path / "agents" / "workspace"
    agents.mkdir(parents=True)
    (agents / "agent1.json").write_text(json.dumps({"provider": "claude", "cwd": "/project", "title": "Test", "lastStatus": "stopped", "persistence": {"sessionId": "session1"}}))
    projects = tmp_path / "projects" / "-project"
    projects.mkdir(parents=True)
    (projects / "session1.jsonl").write_text("\n".join(json.dumps(x) for x in [
        {"type": "user", "uuid": "u", "timestamp": "t1", "message": {"role": "user", "content": "Start"}},
        {"type": "assistant", "uuid": "a", "timestamp": "t2", "message": {"role": "assistant", "content": "Choose?"}},
    ]))
    monkeypatch.setattr(adapter, "PASEO_AGENTS_DIR", tmp_path / "agents")
    monkeypatch.setattr(adapter, "CLAUDE_PROJECTS_DIR", tmp_path / "projects")
    native = resolve_agent("agent1")
    assert native["contract"]["first_prompt"] == "Start"
    assert not any(e["id"] == "TRUNCATION_MARKER" for e in native["events"])
    (projects / "session1.jsonl").unlink()
    calls = []
    def run(args, **kwargs):
        calls.append(args)
        return SimpleNamespace(stdout="[t1] user: Start\n[t2] assistant: Choose?\n")
    monkeypatch.setattr(adapter.subprocess, "run", run)
    fallback = resolve_agent("agent1", tail=10)
    assert calls == [["paseo", "logs", "agent1", "--tail", "10"]]
    assert fallback["events"][0]["id"] == "TRUNCATION_MARKER"
