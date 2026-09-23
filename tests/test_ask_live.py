"""Live smoke for jev_ask: a real stdio client, a real key, a question written here.

Run with:  export TYPESAFE_API_KEY=...; uv run pytest -m live tests/test_ask_live.py
"""

import json
import os
import sys

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from sieve.client import API_KEY_ENV

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not os.environ.get(API_KEY_ENV), reason=f"{API_KEY_ENV} is not set"),
]

#: Nothing about these five needs a repository; the point is that the question is ad hoc.
ITEMS = [
    {"id": "crash", "text": "The importer segfaults when the CSV has a BOM."},
    {"id": "dark-mode", "text": "Please add a dark theme to the settings page."},
    {"id": "wrong-total", "text": "The invoice total is off by one cent on refunds."},
    {"id": "docs", "text": "The README still points at the old install command."},
    {"id": "slow", "text": "Search takes 8 seconds on a 200-row table."},
]


def _payload(result):
    """The tool's structured result, however this client version surfaces it."""
    if getattr(result, "structured_content", None):
        return result.structured_content
    return json.loads(result.content[0].text)


async def test_jev_ask_over_stdio():
    params = StdioServerParameters(command=sys.executable, args=["-m", "sieve.server"], env=dict(os.environ))
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        names = [tool.name for tool in (await session.list_tools()).tools]
        assert names[0] == "jev_ask"

        judged = _payload(await session.call_tool("jev_ask", {
            "question": "Does the item report something behaving incorrectly, rather than asking for new behaviour?",
            "items": ITEMS,
            "kind": "judge",
            "yes": "The item describes existing behaviour that is broken, wrong, or unacceptably slow.",
            "no": "The item asks for a feature that does not exist yet, or is not about behaviour at all.",
        }))
        print("\njudge:", judged["results"], f"${judged['cost_usd']:.5f}")
        probabilities = {row["id"]: row["probability"] for row in judged["results"]}
        assert probabilities["crash"] > 0.5
        assert probabilities["dark-mode"] < 0.5
        assert judged["kind"] == "judge" and judged["items_scored"] == len(ITEMS)

        chosen = _payload(await session.call_tool("jev_ask", {
            "question": "Which team should pick this up first?",
            "items": ITEMS,
            "kind": "choose",
            "options": {
                "backend": "data handling, parsing, money arithmetic, query performance",
                "frontend": "themes, page layout, anything the user sees in the browser",
                "docs": "written material about the product, not the product itself",
            },
        }))
        print("choose:", [(row["id"], row["choice"]) for row in chosen["results"]])
        by_id = {row["id"]: row["choice"] for row in chosen["results"]}
        assert by_id["dark-mode"] == "frontend"
        assert by_id["docs"] == "docs"

        scored = _payload(await session.call_tool("jev_ask", {
            "question": "How badly does the item hurt someone trying to use the product today?",
            "items": ITEMS,
            "kind": "score",
            "levels": [
                "a blemish nobody is blocked by",
                "an annoyance with a workaround",
                "work cannot be completed or the result is wrong",
            ],
        }))
        print("score:", [(row["id"], row["level"], row["expected"]) for row in scored["results"]])
        levels = {row["id"]: row["level"] for row in scored["results"]}
        assert levels["crash"] > levels["docs"]
        assert scored["levels"][0].startswith("a blemish")
