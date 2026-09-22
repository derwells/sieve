# sieve

sieve is a local MCP server that ranks repository files, functions, and web
search results for coding agents. Jev, TypeSafe's System One model, returns
probabilities over a fixed set of candidates and never generates text. sieve
enumerates candidates in code, asks Jev to score them, and returns locations
and probabilities. The agent still opens and reads the survivors itself.

The server runs locally but sends file previews and candidate text to the
TypeSafe API, so a TypeSafe API key is required. Get one at
https://docs.typesafe.ai/introduction/quickstart.

See [Cost notes](#cost-notes) for pricing and [Status](#status) for the failed
recall gate.

## Install

Requires Python 3.12 or later and [uv](https://docs.astral.sh/uv/).

```sh
git clone https://github.com/derwells/sieve.git
cd sieve
uv sync
```

sieve reads the TypeSafe API key from `TYPESAFE_API_KEY` in its environment. The
`bin/sieve-mcp` launcher also sources an env file if one exists, so the key never
has to sit in a harness config file. The default path is `~/.config/sieve/env`;
`SIEVE_ENV_FILE` overrides it. The launcher sources this file as shell code and
exports its assignments; values in the file override existing environment values.

```sh
mkdir -p ~/.config/sieve
cat > ~/.config/sieve/env <<'EOF'
TYPESAFE_API_KEY=...
BRAVE_API_KEY=...
EOF
chmod 600 ~/.config/sieve/env
```

Replace `...` with your API key. `BRAVE_API_KEY` is optional and only affects
`jev_search`; remove that line if you do not use Brave, since even the placeholder
selects the Brave backend. If no env file exists,
the launcher uses whatever is already in the environment.

Register the server with a harness. The launcher resolves the repository from its
own location, so any clone path works. Replace `/path/to/sieve/bin/sieve-mcp`
with the absolute launcher path in your clone. The launcher requires Bash,
`readlink -f`, and `uv` on the MCP client's executable search path.

Claude Code, user scope:

```sh
claude mcp add --scope user sieve -- /path/to/sieve/bin/sieve-mcp
```

Codex, in `~/.codex/config.toml`:

```toml
[mcp_servers.sieve]
command = "/path/to/sieve/bin/sieve-mcp"
```

OpenCode, in `~/.config/opencode/opencode.json`:

```json
{
  "mcp": {
    "sieve": {
      "type": "local",
      "command": ["/path/to/sieve/bin/sieve-mcp"],
      "enabled": true
    }
  }
}
```

To run the server directly from the clone: `uv run sieve`. It speaks MCP over
stdio. This command does not source the env file; export `TYPESAFE_API_KEY` first
or use the launcher.

## Tools

### `jev_grep(question, path, mode="files", top_k=20, threshold=0.5, budget_usd=0.50)`

Ranks a repository against a plain-language question. The walk is
gitignore-aware and never follows symlinks. `mode="files"` scores every file using
its path and preview. Files longer than 80 lines use their first 40 lines plus
a line-numbered outline; shorter files use their full text. Previews have a
6_000-character cap.
`mode="functions"` runs the files pass first, then splits the strongest surviving
files into functions with tree-sitter (fixed 60-line chunks where no grammar
applies) and scores those.

Returns:

```json
{
  "results": [
    {"path": "sieve/validate.py", "line_start": 1, "line_end": 91, "kind": "file", "probability": 0.91}
  ],
  "units_scored": 137,
  "tokens": 204233,
  "input_tokens": 201411,
  "output_tokens": 2822,
  "requests": 18,
  "cache_hits": 0,
  "cost_usd": 0.008459,
  "budget_exhausted": false
}
```

Results are sorted by probability. Units below `threshold` are dropped. Answers
are cached in sqlite, keyed on (model, question, unit text, prompt), so a repeated
question incurs no Jev charge when all units are cached and those inputs are
unchanged. The cache lives in `.sieve-cache/` inside the searched
repository when sieve can add that pattern to an existing `.gitignore` (which
it modifies if necessary), and under `~/.cache/sieve/<repo-hash>/` otherwise, so it never leaves an untracked directory
in someone else's working tree.

### `jev_rank(question, candidates, top_k=None, threshold=0.0)`

Reranks candidates the agent already holds. `candidates` is `[{id, text}]`; a
missing `id` falls back to the list position. Returned IDs are strings.
`top_k=None` returns all candidates that meet `threshold`. Each candidate is cut to 2000
characters, 40 go in one request, and requests run concurrently.

Returns:

```json
{
  "results": [{"id": "postgres", "probability": 0.94}],
  "candidates_scored": 5,
  "tokens": 1204,
  "input_tokens": 1180,
  "output_tokens": 24,
  "requests": 1,
  "cache_hits": 0,
  "cost_usd": 0.00005,
  "budget_exhausted": false
}
```

### `jev_route(ask, routes, budget_usd=0.10)`

Chooses one route for `ask` from caller supplied `routes`, each with `id`,
`description`, and `aliases`, or chooses `none` when no route fits. It asks one
Choice over the routes and `none`. With more than 10 routes, it scores groups of
at most 10, keeps the two strongest routes from each group, then asks a final
Choice over the survivors and `none`. Usage reports the number of stages and
groups. For more than five groups, the final Choice uses the 10 strongest
survivors from the first stage.

Returns:

```json
{
  "choice": "billing",
  "confidence": 0.92,
  "confidence_source": "sdk",
  "probabilities": {"billing": 0.94, "docs": 0.04, "none": 0.02},
  "usage": {"tokens": 410, "input_tokens": 380, "output_tokens": 30, "requests": 1, "cache_hits": 0, "cost_usd": 0.000016, "budget_exhausted": false, "stages": 1, "batches": 1}
}
```

### `jev_search(query, top_k=10, variants=3)`

Searches the web and reranks the results. Code requests 2 to 4 query variants: the original, one with
filler words stripped, a reordered rephrase, and `docs` and `github` suffixes when
the query names software. Duplicate variants are removed, so fewer may run.
The variants run concurrently through one backend. Hits are deduped by canonical URL (lowercase host, no `www.`, no default port, no
fragment, no tracking parameters), and the survivors are reranked against the
*original* query through the `jev_rank` path. Only the top k reach the agent. The JSON below is abbreviated: `usage` also
contains `input_tokens`, `output_tokens`, and `budget_exhausted`. Partial backend
failures add `backend_errors`; if every variant fails, the tool raises an error.

Returns:

```json
{
  "results": [{"url": "https://docs.typesafe.ai/...", "title": "Re-ranking", "snippet": "...", "probability": 0.93}],
  "variants": ["typesafe jev rerank cookbook", "..."],
  "backend": "brave",
  "usage": {"tokens": 3120, "requests": 1, "cost_usd": 0.00013, "cache_hits": 0},
  "hits_found": 30,
  "hits_deduped": 21,
  "backend_calls": [{"query": "...", "hits": 10, "wall_seconds": 0.7, "usage": {}}],
  "wall_seconds": 2.1
}
```

String rules produce query variants. The backend supplies each result's url,
title and snippet. Jev assigns a relevance probability to each deduped hit.

### `jev_verify(records=None, report=None, base_path=None, budget_usd=0.50, support_threshold=0.6, contradict_threshold=0.5)`

Checks claims against cited files or web pages. Provide either `records` or a
Markdown `report`. A record has a `claim`, optional `claim_context`, optional
`kind` (`fact` or `recommendation`), optional `premises` as strings, and
`citations` as `[{"locator": "...", "quote": "..."}]`. A locator is an HTTP URL
or a file path. Relative paths resolve against `base_path`. Recommendations are
exempt from a verdict; their premises are checked as facts using the same citations.

For a report, sieve extracts sentences, bullets, and table rows in code. Headings,
parent list items, and table headers supply context. These claims carry
`extraction_uncertain: true`, so review their wording before relying on a verdict.

sieve fetches each cited source once, up to 2 MB, and records the SHA256 hash of
its bytes as `source_version`. It finds supplied quotes by normalised text match,
then selects passages near the quote or by lexical overlap. Jev compares each
passage with the entire claim and returns probabilities for `supports_fully`,
`partially_supports`, `contradicts`, and `does_not_address`. The result keeps all
passage distributions, the passage with the strongest full support, and the
largest contradiction probability. A failed fetch is reported as a flag, not
as a low support score.

`counts` groups factual claims by their top verdict. `flagged` lists claims with
fetch, quote, evidence, threshold, or budget flags. The
`unqualified_factual_relay_blocked` field is true if a factual claim has no
fetchable citation or any passage reaches `contradict_threshold`. The default
thresholds are provisional until fitted on the verification eval. `usage`
reports tokens, requests, cache hits, cost, and budget exhaustion.

### `jev_triage_threads(threads, budget_usd=0.50, request_threshold=0.5)`

Scores requests for human input across several normalized threads. Each thread
has `thread_id`, chronological `events`, `contract`, and optional `status` and
`journal_priority`. Events have `id`, `ts`, `role`, `kind`, and `text`. The
contract has `title`, `first_prompt`, and `human_amendments`. Results contain
each request, its raw probabilities, source event IDs, coverage, a snapshot
pointer, a thread bucket, and usage. The threshold is provisional.

Candidate requests come from assistant prose. For each candidate, separate
Noul questions check whether it requests input, whether later human dialogue
answers it, whether the assistant withdrew it, and whether the latest
assistant statement says it blocks progress. Long dialogue is scored in
overlapping windows. With full coverage, no later human turn makes the request
unanswered; incomplete coverage makes it unknown. A running agent with newer
assistant progress is not marked blocked. If the contract states acceptance
criteria, another Noul checks whether exactly one action remains.

### `jev_triage_paseo(agent_ids, tail=400, budget_usd=0.50, request_threshold=0.5)`

Reads local Paseo agent indexes and native Claude or Codex transcripts, then
calls the same scorer. When a native transcript is unavailable, it reads
`paseo logs` text and marks coverage as truncated. The tool accepts agent IDs
only. It does not accept log text.

## Backends and auto-selection

| backend | how | snippets | titles |
|---|---|---|---|
| `brave` | Brave Search HTTP API, needs `BRAVE_API_KEY` | real | verbatim |
| `claude` | headless `claude -p`, WebSearch results read off the stream-json stream | none | verbatim |
| `codex` | headless `codex exec`, parses the markdown list the model writes | none | model-transcribed |

`SIEVE_SEARCH_BACKEND` picks one. Otherwise `brave` is used whenever
`BRAVE_API_KEY` is set, and `claude` if it is not. `SIEVE_SEARCH_CMD` replaces the
CLI command line, and `SIEVE_SEARCH_TIMEOUT` the 90 s per-call limit. The CLI
backends require an installed, authenticated `claude` or `codex` executable.

`codex exec` does not put search results on its event stream in the measured setup: the
`web_search` event carries only the query, so that backend parses the markdown
bullet list the model writes afterwards, which makes its titles transcribed rather
than verbatim. Claude's `usage.server_tool_use.web_search_requests` reports 0
even when results come back, so sieve counts tool results instead.

## Cost notes

Measured 2026-09-22, same query, three variants each:

| backend | wall time | cost |
|---|---|---|
| `brave` | 2.1 s | a fraction of a cent |
| `claude` | 16.6 s | ~$0.15 for the three calls |
| `codex` | 29.0 s | not measured |

The `brave` and `codex` rows exclude the backend's own charges: Brave bills
separately under its API pricing, and Codex runs under a Codex subscription
rather than metered API cost. The `claude` figure is the API-equivalent cost
that Claude Code itself reports for the call, not a separate metered charge.

In this measurement, one headless Claude call used ~65k tokens and cost ~$0.05
with ~13 s wall time, even for a small query. Claude Code's own system prompt and tool schemas are the floor, even
with MCP, settings and extra tools stripped off.

sieve calculates Jev costs from input tokens at $0.042 per million, the rate
configured in `sieve/jev.py`. The eval includes a 1,090-file repository
at HEAD. A `jev_grep` files pass over its 1080-file parent snapshot cost $0.054
and 5.6 s cold; over a 190-file repository, $0.0084 and 2.0 s.
`jev_grep` and `jev_verify` expose a `budget_usd` cap in the MCP interface. They return partial
results with `budget_exhausted: true` when the budget stops scoring. The check
uses estimated token costs, so actual spend can exceed the cap.
`jev_rank` and `jev_search` expose no budget parameter. Search `usage.cost_usd`
covers Jev reranking; backend charges are separate. These are measured costs,
not a current provider price list.

## Design rules

- Jev selects or scores over a closed, code-defined set. Text comes from code.
- Batch every question that shares a state into one request.
- Validate every answer client-side: probabilities cover the offered set and
  sum to ~1; a Choice (a selection from fixed options) must pick the max-probability
  option. Reject answers that fail validation.
- Relevance floats are filters, not truth. Thresholds are evaluated on real
  asks, not copied from cookbooks.
- The API key is read from the environment. It never appears in a harness config
  file or in this repository.

## Out of scope

- Jev does not generate summaries or write queries. The Codex search backend
  does use generated text to extract results.
- Indexing or embeddings. Every call enumerates fresh; the cache covers repeats.
- Multi-hop code tracing.

## Status

Implemented and evaluated:

- `jev_grep`, `jev_rank`, `jev_search` and `jev_verify` served over stdio by `uv run sieve`.
- Recall eval on five past asks in four private repositories, written up
  anonymised in [`evals/`](evals/recall-2026-09-22.md). The gate required
  files-mode recall@10 ≥ 0.8 on 4 of 5 asks and got 2 of 5, so it failed.
  recall@10 is the fraction of ground-truth files retrieved in the top 10.
  A looser, post-hoc check found the primary fix file in the top 10 for 5 of 5
  asks. That check does not replace the failed gate. The eval used
  `threshold=0.0`; the default filter can omit additional files.
  The eval also produced two shipped changes: outline previews, and 8 units per
  request instead of 4.
- Citation eval for `jev_verify` on 40 hand-built cases from public sources,
  half true and half altered, in [`evals/`](evals/verify-2026-09-22.md).
  Thresholds fitted on 20 and tested on the other 20: no altered claim
  accepted, no true claim flagged, 18 of 20 four-way verdicts correct on each
  half. Scope alterations come back as contradicts rather than partial support.
- All three search backends exercised live. On the acceptance query, `brave` and
  `claude` put the right page first; `codex` missed it and transcribed its links.
- Registration verified headless in Claude Code, Codex and OpenCode, and through
  a Paseo (an agent management app) plugin that injects the server into every agent.
  That plugin is separate from the installation instructions above.

Roadmap:

- `jev_route(ask)`: Choice over a code-defined set of routes, for an orchestrator
  that has to pick a project for an incoming request.
- A Noul (a probability-valued judgment) gate for auto-approving read-only shell commands that no static rule
  matches.
- Sharper `jev_grep` criteria for large repositories, where 35 files can
  legitimately answer "would a developer have to open this".
- `jev_triage_threads(events, contract)`: ranks agent threads for a human
  briefing. Code enumerates candidate requests for human input; one Noul per
  request decides whether later human dialogue answered it and whether progress
  is waiting on it. Shipped with a Paseo adapter; eval in
  [`evals/`](evals/triage-2026-09-22.md). On 30 private thread snapshots the
  candidate method did not beat whole-window scoring on the test half (6 of
  15 buckets right against 10 of 15), so the chief should treat its output as
  a filter to inspect, not a ranking to trust.

## Stack

Python 3.12, `uv`, `typesafe-sdk` (async client), `mcp` (stdio; FastMCP is
`MCPServer` in mcp 2.x), `tree-sitter-language-pack`, `pytest`. The brave
backend uses `httpx2`, which the TypeSafe SDK already pins.

Tests: `uv run pytest -q -m "not live"` for the offline suite. `uv run pytest -m
live` also hits the real API and needs `TYPESAFE_API_KEY`.

## References

- Docs index: https://docs.typesafe.ai/llms.txt
- Pipeline constants borrowed from `superagents-lab/jev-search`: 40 per rerank
  batch, 0.6 source threshold, 8 results per lane.
- Answer validation pattern from `browser-use/jev-ultrafast`.

## License

MIT. See [LICENSE](LICENSE).

Repository enumeration in `sieve/enumerate.py` is adapted from
[`keltokhy/jgrep`](https://github.com/keltokhy/jgrep), MIT, Copyright (c) 2026
Khaled Eltokhy. Its license notice is reproduced in [NOTICE](NOTICE) and in the
module itself.
