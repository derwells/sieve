"""The sieve MCP server: Jev-backed judgment over stdio.

`mcp` 2.x renamed FastMCP to MCPServer; this is the same server, current name.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from mcp.server.mcpserver import MCPServer
from pydantic import Field

from . import ask as ask_module
from . import grep as grep_module
from . import rank as rank_module
from . import route as route_module
from . import search as search_module
from . import verify as verify_module
from . import triage as triage_module
from . import paseo_adapter
from .cache import ask_cache
from .search_cache import SearchCache
from .errors import SieveError

server = MCPServer(
    name="sieve",
    version="0.1.0",
    instructions=(
        "Jev judges many items at once against a question you write. jev_ask is the "
        "general tool: write the question, enumerate the answer space (yes/no criteria, "
        "named options, or an ordered scale), pass the items, get a probability per item. "
        "Anything you were about to eyeball item by item — failures, comments, hits, "
        "threads, options — is a jev_ask. The other tools are shortcuts for recurring "
        "shapes: jev_grep ranks a repository's files or functions, jev_rank reranks "
        "candidates you hold, jev_search reranks web results, jev_verify checks cited "
        "claims, jev_route picks one destination. Jev never generates text; you read "
        "the survivors yourself."
    ),
)


@server.tool(
    name="jev_ask",
    title="Ask your own question of many items",
    description=(
        "Run one question you wrote over items you already hold. Write the question, "
        "enumerate the answer space, and Jev answers it for every item in parallel. "
        "kind='judge' takes yes/no criteria and returns a probability per item; "
        "kind='choose' takes named options and returns the pick plus the full "
        "distribution; kind='score' takes an ordered list of levels and returns each "
        "item's level, the expected score, and the distribution. Items are strings or "
        "{id, text}; each is cut to max_chars, 40 go per request, requests run "
        "concurrently, answers are cached on (model, question, item, criteria), and the "
        "result carries token and cost usage. Put shared context (the traceback, the "
        "spec, the goal) in the question, not in every item; write criteria that are "
        "concrete and mutually exclusive, since Jev reads them literally."
    ),
)
async def jev_ask(
    question: Annotated[str, Field(description="Your question, written to refer to one item at a time.")],
    items: Annotated[
        list[str | dict[str, Any]],
        Field(description="Items as strings or [{id, text}]. Missing ids fall back to list position."),
    ],
    kind: Annotated[
        Literal["judge", "choose", "score"],
        Field(description="'judge' for yes/no, 'choose' for named options, 'score' for an ordered scale."),
    ] = "judge",
    yes: Annotated[str | None, Field(description="kind='judge': what makes the answer true.")] = None,
    no: Annotated[str | None, Field(description="kind='judge': what makes the answer false.")] = None,
    options: Annotated[
        dict[str, str] | None,
        Field(description="kind='choose': {label: when this label applies}. Add your own 'none' if one is needed."),
    ] = None,
    levels: Annotated[
        list[str] | None,
        Field(description="kind='score': level descriptions in order, lowest first."),
    ] = None,
    top_k: Annotated[int | None, Field(ge=1, description="kind='judge': return at most this many items.")] = None,
    threshold: Annotated[
        float, Field(ge=0.0, le=1.0, description="kind='judge': drop items below this probability.")
    ] = 0.0,
    max_chars: Annotated[int, Field(ge=100, description="Cut each item's text to this many characters.")] = 4000,
    budget_usd: Annotated[
        float, Field(gt=0.0, description="Stop and return partial results before spending more than this.")
    ] = 0.50,
) -> dict[str, Any]:
    cache = ask_cache()
    try:
        return await ask_module.ask(
            question=question,
            items=items,
            kind=kind,
            yes=yes,
            no=no,
            options=options,
            levels=levels,
            top_k=top_k,
            threshold=threshold,
            max_chars=max_chars,
            budget_usd=budget_usd,
            cache=cache,
        )
    except SieveError as e:
        raise ValueError(str(e)) from e
    finally:
        cache.close()


@server.tool(
    name="jev_grep",
    title="Rank a repository against a question",
    description=(
        "Shortcut over jev_ask for repositories: it enumerates the candidates for you. "
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
        "Shortcut over jev_ask with relevance criteria already written. Score each "
        "candidate for relevance to a question and return "
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
    name="jev_route",
    title="Choose a route for a request",
    description=(
        "Choose one caller supplied route or none. Returns the choice, confidence, "
        "probabilities for every final option, and usage. More than ten routes "
        "are shortlisted in concurrent batches before a final Choice."
    ),
)
async def jev_route(
    ask: Annotated[str, Field(description="The request to route.")],
    routes: Annotated[
        list[dict[str, Any]],
        Field(description="Routes as [{id, description, aliases}]. IDs must be unique; none is reserved."),
    ],
    budget_usd: Annotated[float, Field(gt=0.0, description="Approximate Jev budget in USD.")] = 0.10,
) -> dict[str, Any]:
    try:
        return await route_module.jev_route(ask=ask, routes=routes, budget_usd=budget_usd)
    except SieveError as e:
        raise ValueError(str(e)) from e


@server.tool(
    name="jev_search",
    title="Search the web and keep only the relevant hits",
    description=(
        "Search the web for a plain-language query and return "
        "[{url, title, snippet, probability}] sorted by probability. sieve proposes "
        "2-4 query variants in code, runs them concurrently through the configured "
        "backend (searxng, brave, headless claude, or headless codex), dedupes by "
        "canonical url, keeps up to `depth` candidates, and reranks them against your "
        "original query with Jev; only the top_k come back. A sparse query yields a "
        "smaller pool (pool_short), never padding. Backend results are cached for an "
        "hour. Snippets are empty on the claude backend."
    ),
)
async def jev_search(
    query: Annotated[str, Field(description="What you are trying to find on the web, in plain language.")],
    top_k: Annotated[int, Field(ge=1, le=50, description="How many results to return.")] = 10,
    variants: Annotated[
        int, Field(ge=2, le=4, description="How many query variants to run concurrently.")
    ] = 3,
    depth: Annotated[
        int,
        Field(ge=1, le=50, description="Deduped candidates to gather and rerank before the top_k cut; raised to top_k if lower."),
    ] = search_module.DEFAULT_DEPTH,
) -> dict[str, Any]:
    search_cache = SearchCache()
    try:
        return await search_module.jev_search(
            query=query, top_k=top_k, variants=variants, depth=depth, search_cache=search_cache
        )
    except SieveError as e:
        raise ValueError(str(e)) from e
    finally:
        search_cache.close()


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


@server.tool(name="jev_triage_threads", title="Triage thread requests",
             description="Score human input requests in normalized chronological thread events, with evidence and usage.")
async def jev_triage_threads(
    threads: Annotated[list[dict[str, Any]], Field(description="Threads as [{thread_id, events, contract, status?, journal_priority?}].")],
    budget_usd: Annotated[float, Field(gt=0.0, description="Approximate Jev budget in USD.")] = 0.50,
    request_threshold: Annotated[float, Field(ge=0.0, le=1.0, description="Provisional request and decision threshold.")] = 0.5,
) -> dict[str, Any]:
    return await triage_module.jev_triage_threads(threads, budget_usd, request_threshold)


@server.tool(name="jev_triage_paseo", title="Triage local Paseo agents",
             description="Resolve local Paseo agent IDs from stored transcripts, then score human input requests. Accepts IDs only, never raw logs.")
async def jev_triage_paseo(
    agent_ids: Annotated[list[str], Field(description="Local Paseo agent IDs.")],
    tail: Annotated[int, Field(ge=3, description="Maximum normalized events per agent.")] = 400,
    budget_usd: Annotated[float, Field(gt=0.0, description="Approximate Jev budget in USD.")] = 0.50,
    request_threshold: Annotated[float, Field(ge=0.0, le=1.0, description="Provisional request and decision threshold.")] = 0.5,
) -> dict[str, Any]:
    return await paseo_adapter.jev_triage_paseo(agent_ids, tail, budget_usd, request_threshold)


def main() -> None:
    """Entry point for `uv run sieve`: serve the tools over stdio."""
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
