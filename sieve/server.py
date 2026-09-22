"""The sieve MCP server: two Jev-backed tools over stdio.

`mcp` 2.x renamed FastMCP to MCPServer; this is the same server, current name.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from mcp.server.mcpserver import MCPServer
from pydantic import Field

from . import grep as grep_module
from . import rank as rank_module
from . import search as search_module
from . import verify as verify_module
from .errors import SieveError

server = MCPServer(
    name="sieve",
    version="0.1.0",
    instructions=(
        "Jev-backed filters so you read less. Use jev_grep to find the files or "
        "functions in a repository that bear on a plain-language question, "
        "jev_rank to rerank candidates you already hold, and jev_search to search "
        "the web and get back only the pages worth opening. jev_verify checks "
        "cited claims against source passages. Read ranked survivors yourself."
    ),
)


@server.tool(
    name="jev_grep",
    title="Rank a repository against a question",
    description=(
        "Rank the files, or the functions inside the strongest files, of a repository "
        "by how relevant they are to a plain-language question. Returns "
        "{path, line_start, line_end, kind, probability} sorted by probability, plus "
        "token and cost usage. Enumeration is gitignore-aware; answers are cached per "
        "(model, question, unit)."
    ),
)
async def jev_grep(
    question: Annotated[str, Field(description="What you are trying to find out, in plain language.")],
    path: Annotated[str, Field(description="Absolute path to the repository or directory to search.")],
    mode: Annotated[
        Literal["files", "functions"],
        Field(description="'files' scores whole files; 'functions' then splits the best files."),
    ] = "files",
    top_k: Annotated[int, Field(ge=1, le=200, description="How many results to return.")] = 20,
    threshold: Annotated[
        float, Field(ge=0.0, le=1.0, description="Drop units scoring below this probability.")
    ] = 0.5,
    budget_usd: Annotated[
        float, Field(gt=0.0, description="Stop and return partial results before spending more than this.")
    ] = 0.50,
) -> dict[str, Any]:
    try:
        return await grep_module.jev_grep(
            question=question,
            path=path,
            mode=mode,
            top_k=top_k,
            threshold=threshold,
            budget_usd=budget_usd,
        )
    except SieveError as e:
        raise ValueError(str(e)) from e


@server.tool(
    name="jev_rank",
    title="Rerank candidates against a question",
    description=(
        "Score each candidate for relevance to a question and return "
        "[{id, probability}] sorted by probability, plus token and cost usage. "
        "Candidates are truncated to 2000 characters and sent 40 per request."
    ),
)
async def jev_rank(
    question: Annotated[str, Field(description="What the candidates are being ranked against.")],
    candidates: Annotated[
        list[dict[str, Any]],
        Field(description="Candidates as [{id, text}]. Missing ids fall back to list position."),
    ],
    top_k: Annotated[
        int | None, Field(ge=1, description="Return at most this many results; null returns all.")
    ] = None,
    threshold: Annotated[
        float, Field(ge=0.0, le=1.0, description="Drop candidates scoring below this probability.")
    ] = 0.0,
) -> dict[str, Any]:
    try:
        return await rank_module.jev_rank(
            question=question,
            candidates=candidates,
            top_k=top_k,
            threshold=threshold,
        )
    except SieveError as e:
        raise ValueError(str(e)) from e


@server.tool(
    name="jev_search",
    title="Search the web and keep only the relevant hits",
    description=(
        "Search the web for a plain-language query and return "
        "[{url, title, snippet, probability}] sorted by probability. sieve proposes "
        "2-4 query variants in code, runs them concurrently through the configured "
        "backend (brave, headless claude, or headless codex), dedupes by canonical "
        "url, and reranks everything against your original query with Jev. Snippets "
        "are empty on the CLI backends; codex titles are model-transcribed."
    ),
)
async def jev_search(
    query: Annotated[str, Field(description="What you are trying to find on the web, in plain language.")],
    top_k: Annotated[int, Field(ge=1, le=50, description="How many results to return.")] = 10,
    variants: Annotated[
        int, Field(ge=2, le=4, description="How many query variants to run concurrently.")
    ] = 3,
) -> dict[str, Any]:
    try:
        return await search_module.jev_search(query=query, top_k=top_k, variants=variants)
    except SieveError as e:
        raise ValueError(str(e)) from e


@server.tool(
    name="jev_verify",
    title="Check cited claims against their evidence",
    description=(
        "Check structured claim records or a Markdown report against cited URLs and files. "
        "Returns a Choice distribution per evidence window, quote and fetch flags, "
        "per-claim verdicts, and usage. Thresholds are provisional until fitted on the eval."
    ),
)
async def jev_verify(
    records: Annotated[
        list[dict[str, Any]] | None,
        Field(description="Claim records with claim, optional claim_context, kind, premises, and citations [{locator, quote?}]."),
    ] = None,
    report: Annotated[str | None, Field(description="Markdown report to extract claims and citations from in code.")] = None,
    base_path: Annotated[str | None, Field(description="Base directory for relative citation file paths.")] = None,
    budget_usd: Annotated[float, Field(gt=0.0, description="Approximate Jev scoring budget in USD.")] = 0.50,
    support_threshold: Annotated[float, Field(ge=0.0, le=1.0, description="Provisional full-support threshold.")] = 0.6,
    contradict_threshold: Annotated[float, Field(ge=0.0, le=1.0, description="Provisional contradiction threshold.")] = 0.5,
) -> dict[str, Any]:
    try:
        return await verify_module.jev_verify(
            records=records, report=report, base_path=base_path, budget_usd=budget_usd,
            support_threshold=support_threshold, contradict_threshold=contradict_threshold,
        )
    except SieveError as e:
        raise ValueError(str(e)) from e


def main() -> None:
    """Entry point for `uv run sieve`: serve the tools over stdio."""
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
