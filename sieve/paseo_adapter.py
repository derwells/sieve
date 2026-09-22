"""Read local Paseo agent indexes and native transcripts for thread triage."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from .triage import jev_triage_threads

PASEO_AGENTS_DIR = Path.home() / ".paseo" / "agents"
CLAUDE_PROJECTS_DIR = Path.home() / ".claude" / "projects"
CODEX_ARCHIVED_DIR = Path.home() / ".codex" / "archived_sessions"
CODEX_SESSIONS_DIR = Path.home() / ".codex" / "sessions"
INJECTED = ("<recommended_plugins>", "# AGENTS.md instructions", "<environment_context>", "<user_instructions>", "<paseo-system>", "<skill>", "[Request interrupted by user]")
SYNTHETIC = ("<task-notification>", "<paseo-system>", "Another Claude session sent a message:", "[Request interrupted by user]", "<local-command-stdout>")
LOG_LINE = re.compile(r"^\s*(?:\[([^]]+)\]\s*)?(human|user|assistant|tool)\s*:\s*(.*)$", re.I)


def _blocks(blocks) -> str:
    return "\n".join(str(b.get("text", "")) for b in blocks or [] if isinstance(b, dict) and b.get("type") in {"input_text", "output_text", "text"})


def _event(eid, ts, role, kind, text):
    return {"id": str(eid), "ts": ts, "role": role, "kind": kind, "text": str(text or "")}


def parse_claude(path: Path) -> list[dict]:
    events = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        try:
            data = json.loads(line)
        except ValueError:
            continue
        if data.get("type") not in {"user", "assistant"} or data.get("isSidechain"):
            continue
        msg = data.get("message") or {}
        blocks = msg.get("content") or []
        if isinstance(blocks, str):
            blocks = [{"type": "text", "text": blocks}]
        for i, block in enumerate(blocks):
            if not isinstance(block, dict):
                continue
            kind = block.get("type")
            eid = f"{data.get('uuid', f'line{number}')}:{i}"
            ts = data.get("timestamp")
            if kind == "text":
                value = block.get("text", "")
                synthetic = msg.get("role") == "user" and (data.get("isMeta") or data.get("turnCompanion") or value.lstrip().startswith(SYNTHETIC))
                events.append(_event(eid, ts, "tool" if synthetic else "human" if msg.get("role") == "user" else "assistant", "system" if synthetic else "text", value))
            elif kind == "thinking":
                events.append(_event(eid, ts, "assistant", "thinking", block.get("thinking")))
            elif kind == "tool_use":
                events.append(_event(eid, ts, "assistant", "tool_use", f"{block.get('name')}({json.dumps(block.get('input', {}))[:500]})"))
            elif kind == "tool_result":
                value = block.get("content")
                events.append(_event(eid, ts, "tool", "tool_result", _blocks(value) if isinstance(value, list) else value if isinstance(value, str) else json.dumps(value)))
    return events


def parse_codex(path: Path) -> list[dict]:
    events = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        try:
            data = json.loads(line)
        except ValueError:
            continue
        if data.get("type") != "response_item":
            continue
        p = data.get("payload") or {}
        kind = p.get("type")
        eid = f"ord{data['ordinal']}" if data.get("ordinal") is not None else p.get("id", f"line{number}")
        ts = data.get("timestamp")
        if kind == "message":
            role = p.get("role")
            blocks = p.get("content") or []
            if role == "user":
                for i, block in enumerate(blocks):
                    value = block.get("text", "") if isinstance(block, dict) else str(block)
                    system = value.lstrip().startswith(INJECTED)
                    events.append(_event(f"{eid}:{i}", ts, "tool" if system else "human", "system" if system else "text", value))
            else:
                events.append(_event(eid, ts, "assistant" if role == "assistant" else "tool", "text" if role == "assistant" else "system", _blocks(blocks)))
        elif kind == "reasoning":
            events.append(_event(eid, ts, "assistant", "thinking", _blocks(p.get("summary")) or "[reasoning, no summary available]"))
        elif kind in {"function_call", "custom_tool_call"}:
            arg = p.get("arguments") if kind == "function_call" else p.get("input")
            events.append(_event(eid, ts, "assistant", "tool_use", f"{p.get('name')}({str(arg)[:500]})"))
        elif kind in {"function_call_output", "custom_tool_call_output"}:
            value = p.get("output")
            events.append(_event(eid, ts, "tool", "tool_result", _blocks(value) if isinstance(value, list) else value if isinstance(value, str) else json.dumps(value)))
        elif kind == "agent_message":
            events.append(_event(eid, ts, "tool", "tool_result", _blocks(p.get("content"))))
    return events


def parse_paseo_logs(output: str) -> list[dict]:
    """Conservative text fallback; every such snapshot is marked truncated."""
    events = []
    current = None
    for line in output.splitlines():
        match = LOG_LINE.match(line)
        if match:
            ts, role, value = match.groups()
            current = _event(f"log{len(events) + 1}", ts, "human" if role.lower() in {"human", "user"} else role.lower(), "tool_result" if role.lower() == "tool" else "text", value)
            events.append(current)
        elif current is not None and line.strip():
            current["text"] += "\n" + line
    return events


def _record(agent_id: str) -> dict:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", agent_id):
        raise ValueError("invalid agent id")
    matches = list(PASEO_AGENTS_DIR.glob(f"*/{agent_id}.json"))
    if not matches:
        raise FileNotFoundError(f"no Paseo index record for {agent_id}")
    return json.loads(matches[0].read_text())


def _native(record: dict) -> tuple[list[dict], bool]:
    sid = (record.get("persistence") or {}).get("sessionId")
    if not sid or not re.fullmatch(r"[A-Za-z0-9_-]+", sid):
        raise FileNotFoundError("no valid native session ID")
    provider = record.get("provider")
    if provider == "claude":
        cwd = record.get("cwd") or ""
        direct = CLAUDE_PROJECTS_DIR / cwd.replace("/", "-") / f"{sid}.jsonl"
        paths = [direct] if direct.is_file() else list(CLAUDE_PROJECTS_DIR.glob(f"*/{sid}.jsonl"))
        return parse_claude(paths[0]), False
    if provider == "codex":
        paths = list(CODEX_ARCHIVED_DIR.glob(f"rollout-*-{sid}.jsonl")) or list(CODEX_SESSIONS_DIR.glob(f"**/*{sid}*.jsonl"))
        return parse_codex(paths[0]), False
    raise FileNotFoundError("unsupported provider")


def resolve_agent(agent_id: str, tail: int = 400) -> dict:
    if tail < 3:
        raise ValueError("tail must be at least 3")
    record = _record(agent_id)
    try:
        events, truncated = _native(record)
    except (FileNotFoundError, OSError, IndexError):
        result = subprocess.run(["paseo", "logs", agent_id, "--tail", str(tail)], capture_output=True, text=True, check=True)
        events, truncated = parse_paseo_logs(result.stdout), True
    events.sort(key=lambda e: (e["ts"] is None, e["ts"] or ""))
    all_humans = [e["text"] for e in events if e["role"] == "human" and e["kind"] == "text"]
    if len(events) > tail:
        head = min(20, (tail - 1) // 2)
        events = events[:head] + events[-(tail - head - 1):]
        truncated = True
    if truncated:
        events.insert(0, _event("TRUNCATION_MARKER", None, "tool", "system", "Transcript coverage is incomplete."))
    humans = [e["text"] for e in events if e["role"] == "human" and e["kind"] == "text"]
    return {"thread_id": agent_id, "events": events,
            "contract": {"title": record.get("title") or "", "first_prompt": all_humans[0] if all_humans else "", "human_amendments": humans[1:]},
            "status": record.get("lastStatus"), "journal_priority": None}


async def jev_triage_paseo(agent_ids: list[str], tail: int = 400, budget_usd: float = 0.50,
                           request_threshold: float = 0.5, *, client=None, cache=None) -> dict:
    threads = [resolve_agent(agent_id, tail) for agent_id in agent_ids]
    return await jev_triage_threads(threads, budget_usd, request_threshold, client=client, cache=cache)
