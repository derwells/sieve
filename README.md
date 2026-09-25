# sieve

sieve is a local MCP server that puts Jev, TypeSafe's System One model, in a
coding agent's hands. Jev returns probabilities over a closed set of candidates
and never generates text. The general tool is `jev_ask`: the agent writes the
question, enumerates the answer space, passes the items, and gets one answer per
item. The rest — `jev_grep`, `jev_rank`, `jev_search`, `jev_verify`, `jev_route`,
`jev_triage_*` — are shortcuts for recurring shapes over the same core, with the
candidate enumeration or the criteria already written. The agent still opens and
reads the survivors itself.

The server runs locally but sends file previews and candidate text to the
TypeSafe API, so a TypeSafe API key is required. Get one at
https://docs.typesafe.ai/introduction/quickstart.

See [Cost notes](#cost-notes) for pricing and [Status](#status) for the recall
gate results.

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
selects the Brave backend. To search through a local SearXNG instance instead,
see [Local SearXNG](#local-searxng). If no env file exists,
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

### `jev_ask(question, items, kind="judge", yes=None, no=None, options=None, levels=None, top_k=None, threshold=0.0, max_chars=4000, budget_usd=0.50)`

One question you wrote, asked of every item you hold. Write the question,
enumerate the answer space, call it. `kind` picks the primitive:

| `kind` | answer space | per item |
|---|---|---|
| `judge` | `yes` and `no` criteria | `{id, probability}`, sorted, cut by `top_k` and `threshold` |
| `choose` | `options` as `{label: when it applies}` | `{id, choice, probabilities}` |
| `score` | `levels`, an ordered list, lowest first | `{id, level, label, expected, probabilities}` |

Items are plain strings or `{id, text}`; a missing `id` falls back to the list
position. Each item's text is cut to `max_chars`, 40 items go in one request,
requests run concurrently under the same budget check the other tools use, and
every answer is validated client-side before it is returned. Answers are cached
in sqlite under `~/.cache/sieve/ask/`, keyed on (model, question, item text,
criteria), so repeating a question costs nothing.

```json
{
  "results": [{"id": "crash", "probability": 0.98}, {"id": "dark-mode", "probability": 0.03}],
  "kind": "judge",
  "items_scored": 2,
  "tokens": 1032,
  "input_tokens": 1012,
  "output_tokens": 20,
  "requests": 1,
  "cache_hits": 0,
  "cost_usd": 0.000043,
  "budget_exhausted": false
}
```

Write the criteria concretely and make them mutually exclusive; Jev reads them
literally, and vague criteria give probabilities near 0.5 across the board. Put
shared context — the traceback, the spec, the goal — in the question rather than
repeating it in every item. Add your own `none` option to `choose` when no listed
label may fit.

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

### `jev_search(query, top_k=10, variants=3, depth=30)`

Searches the web and reranks the results. `depth` is the candidate pool, the
number of deduped hits Jev reranks. It is capped at 50 and is never below
`top_k`. `top_k` is how many of those hits come back. Each variant asks the
backend for an even share of the pool, at least 10 hits. A backend that pages
(`searxng`) asks the original query for the whole pool, because variants overlap
heavily and another page costs less than another variant. The pool is interleaved
rank by rank across variants, deduped, and cut to `depth`. A sparse query returns
a smaller pool with `pool_short: true`. sieve never pads the pool and never adds
variants to fill it. Code requests 2 to 4 query variants: the original, one with
filler words stripped, a reordered rephrase, and `docs` and `github` suffixes when
the query names software. Duplicate variants are removed, so fewer may run.
The variants run concurrently through one backend. Hits are deduped by canonical URL (lowercase host, no `www.`, no default port, no
fragment, no tracking parameters), and the survivors are reranked against the
*original* query through the `jev_rank` path. Only the top k reach the agent. The JSON below is abbreviated: `usage` also
contains `input_tokens`, `output_tokens`, and `budget_exhausted`. Partial backend
failures add `backend_errors`; if every variant fails, the tool raises an error.
Backend results are cached in sqlite under `~/.cache/sieve/search/` for
`SIEVE_SEARCH_CACHE_TTL` seconds (default 3600; `0` turns it off). The cache is
trimmed to the newest 2,000 entries. It stores queries and snippets, so its
directory is kept at mode 700 and the database and any SQLite sidecar files at
600. A cached call shows `cached: true` and does
not count in `backend_requests`. Jev usage is reported separately in `usage`.

Returns:

```json
{
  "results": [{"url": "https://docs.typesafe.ai/...", "title": "Re-ranking", "snippet": "...", "probability": 0.93, "engines": ["google"]}],
  "variants": ["typesafe jev rerank cookbook", "..."],
  "backend": "searxng",
  "usage": {"tokens": 12400, "requests": 2, "cost_usd": 0.00052, "cache_hits": 0},
  "depth": 50,
  "hits_requested": [50, 17, 17],
  "hits_found": 99,
  "hits_deduped": 69,
  "candidates_scored": 50,
  "backend_requests": 6,
  "backend_calls": [{"query": "...", "hits": 50, "wall_seconds": 2.0, "usage": {"requests": 3, "pages": 3, "stopped": "count", "unresponsive_engines": []}}],
  "wall_seconds": 3.0
}
```

String rules produce query variants. The backend supplies each result's url,
title and snippet, and for `searxng` the upstream `engines` and any `published`
date. Jev assigns a relevance probability to each deduped hit.

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

sieve fetches each cited source once, up to 2 MB (25 MB for a PDF), and records
the SHA256 hash of its bytes as `source_version`. A PDF is turned into text in a
child process with a 30 s timeout, a 1 GB memory limit and a 200-page cap. The
child runs `pdftotext` (poppler) when it is installed and `pypdf` otherwise, and
the text is NFKC-normalised so ligatures match typed quotes. An encrypted PDF, a
scanned PDF with no text layer, or a parse failure is reported as a fetch failure
with its reason. Raw PDF bytes are never scored.

It finds supplied quotes by normalised text match (quote characters and dashes
folded, whitespace collapsed, no space before closing punctuation, and one closing
`.`, `;` or `,` on the quote ignored unless a digit precedes it),
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

## `sieve-ask`: the same path from a shell

`bin/sieve-ask` is `jev_ask` for callers that are not MCP clients. Same core,
same batching, caching, budget and validation. Items are a JSON list of strings
or `{id, text}` objects on stdin or in `--items FILE`.

```sh
# one yes/no probability per item
printf '%s' '["fix crash on launch","add dark mode"]' | bin/sieve-ask judge \
  "Does the item describe a bug?" \
  --yes "It reports broken or incorrect behaviour." \
  --no "It asks for new behaviour or is not about behaviour."

# one option per item, with the full distribution
bin/sieve-ask choose "Which team owns this ticket?" --items tickets.json \
  --option "web=browser UI, CSS, React" \
  --option "api=HTTP endpoints, auth, database" \
  --option "none=no listed team fits"

# one level on an ordered scale per item, lowest first
bin/sieve-ask score "How badly does this hurt someone using the product today?" --items bugs.json \
  --level "a blemish nobody is blocked by" \
  --level "an annoyance with a workaround" \
  --level "work cannot be completed or the result is wrong"
```

`judge` takes `--top-k`, `--threshold`; all three take `--budget-usd`,
`--max-chars` (default 4000 per item), `--model`, and `--no-cache`. From Python,
`sieve.ask.ask(question, items, kind, ...)` dispatches on `kind`, and
`judge(...)`, `choose(...)` and `score(...)` are async with the same arguments as
keywords. `bin/sieve-ask` sources the same env file as `bin/sieve-mcp`.

## Backends and auto-selection

| backend | how | snippets | titles |
|---|---|---|---|
| `searxng` | a SearXNG instance's JSON API at `SIEVE_SEARXNG_URL` | real | verbatim |
| `brave` | Brave Search HTTP API, needs `BRAVE_API_KEY` | real | verbatim |
| `claude` | headless `claude -p`, WebSearch results read off the stream-json stream | none | verbatim |
| `codex` | headless `codex exec`, reads the `web_search` results on its event stream | real (search results) | verbatim |

`SIEVE_SEARCH_BACKEND` picks one. Otherwise sieve uses `searxng` when
`SIEVE_SEARXNG_URL` is set, `brave` when `BRAVE_API_KEY` is set, and `claude`
if neither is set. Only the chosen backend is called; sieve does not fall back to
another backend on failure. `SIEVE_SEARCH_CMD` replaces the
CLI command line, and `SIEVE_SEARCH_TIMEOUT` the 90 s per-call limit. The CLI
backends require an installed, authenticated `claude` or `codex` executable.

### Local SearXNG

[SearXNG](https://docs.searxng.org/) is a self-hosted metasearch engine. It sends
one query to several upstream engines and merges their results. `bin/sieve-searxng`
runs the official container, pinned to
`ghcr.io/searxng/searxng:2026.9.25-12f8b6515` by digest, and needs Docker. The
container is bound to `127.0.0.1` only and has the JSON API enabled:

```sh
bin/sieve-searxng start     # detached, --restart unless-stopped
bin/sieve-searxng status    # container state and /healthz
bin/sieve-searxng logs 50
bin/sieve-searxng stop      # removes the container; it stays down after a reboot
```

The container returns after a reboot as long as Docker itself starts at boot,
until you run `stop`. On first start the script writes
`~/.local/share/sieve-searxng/settings.yml` and never overwrites it. The
settings enable `json` under `search.formats`, turn the limiter off (the
instance is single-user on loopback), and enable `google` and `yahoo`.
The secret key goes in `secret.env` beside it with mode 600. `SEARXNG_PORT`
(default 8888), `SEARXNG_HOME`, `SEARXNG_CONTAINER` and `SEARXNG_IMAGE`
override the defaults. To select it, add to `~/.config/sieve/env`:

```sh
SIEVE_SEARCH_BACKEND=searxng
SIEVE_SEARXNG_URL=http://127.0.0.1:8888
```

The backend requests up to `SIEVE_SEARXNG_MAX_PAGES` pages per query (default 3,
at most 5). Each request is limited by `SIEVE_SEARXNG_TIMEOUT` (default 12 s),
and all pages for one query share `SIEVE_SEARXNG_DEADLINE` (default 30 s). Paging
stops early once it has the requested hits or a page returns no new URL. Page 1
is retried once on a connection error, timeout or 5xx. A failure on a later page
keeps the earlier pages and is recorded in `page_errors`. Engines that SearXNG
reports as unresponsive (CAPTCHA, rate limit, timeout) are listed in each call's
`usage.unresponsive_engines`. If no engine returns anything, the call fails.

Upstream engines rate-limit and CAPTCHA automated traffic, so which engines are
healthy changes over time, and an instance needs occasional settings and image
updates. The [2026-09-25 trial](evals/searxng-2026-09-25.md) filled a 50-hit
pool on all four test queries. At times, only one or two engines were answering.

### CLI backend notes

Since codex-cli 0.156.1, each completed `web_search` event carries `results[]`
with `url`, `title`, `snippet` and a `ref_id`. Search results (`turnNsearchM`)
have real snippets. Opened pages (`turnNviewM`, from `open_page` or
`find_in_page`) keep their url and title, but their "Total lines: N" placeholder
snippet is dropped. Results without a url are skipped. Hits come only from these
events. `usage` reports `hits_source: "events"` and counts links in the model's
own list that no event observed as `unobserved_links`. Older releases put only
the query on the event. For those, and only when no event has results, the
backend parses the markdown bullet list the model writes, so titles are
transcribed and snippets empty (`hits_source: "transcript"`). CLI children get an
empty stdin, because `codex exec` reads a non-terminal stdin, which inside the
MCP server is the protocol pipe. Claude's `usage.server_tool_use.web_search_requests` reports 0
even when results come back, so sieve counts tool results instead.

## Cost notes

Measured 2026-09-22, same query, three variants each:

| backend | wall time | cost |
|---|---|---|
| `searxng` | 2.5–5.9 s at `depth=50` (measured 2026-09-25) | none beyond Jev, about $0.0005 per call |
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

- Jev selects or scores over a closed set. The fixed tools define that set in
  code; `jev_ask` lets the caller define it, and enforces the same closure.
- Batch every question that shares a state into one request.
- Validate every answer client-side: probabilities cover the offered set and
  sum to ~1; a Choice (a selection from fixed options) must pick the max-probability
  option. The API rounds probabilities to two decimals, so a pick up to 0.01 below
  the reported maximum still counts. Reject answers that fail validation.
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

- `jev_ask` served over stdio by `uv run sieve`, with `judge`, `choose` and
  `score` exercised live through a real stdio client in
  [`tests/test_ask_live.py`](tests/test_ask_live.py). Not separately evaluated:
  the question is the caller's, so its accuracy is the caller's to check.
- `jev_grep`, `jev_rank`, `jev_search` and `jev_verify` served over stdio by `uv run sieve`.
- Recall eval on five past asks in four private repositories, written up
  anonymised in [`evals/`](evals/recall-2026-09-22.md). At the current default
  of 8 units per request, the files mode gate requires recall@10 at least 0.8
  on 4 of 5 asks. Strict recall over every previously existing file edited by
  the fix reached 3 of 5, so it failed. Relaxed recall over each ask's single
  primary fix file reached 5 of 5, so it passed. Recall@10 is the fraction of
  ground truth files retrieved in the top 10. The eval used `threshold=0.0`;
  the default filter can omit additional files. The earlier eval led to outline
  previews and 8 units per request instead of 4. A criteria sweep did not
  justify another prompt change.
- Citation eval for `jev_verify` on 40 hand-built cases from public sources,
  half true and half altered, in [`evals/`](evals/verify-2026-09-22.md).
  Thresholds fitted on 20 and tested on the other 20: no altered claim
  accepted, no true claim flagged, 18 of 20 four-way verdicts correct on each
  half. Scope alterations come back as contradicts rather than partial support.
- All three search backends exercised live. On the acceptance query, `brave` and
  `claude` put the right page first; `codex` missed it and transcribed its links.
  On 2026-09-25, with codex-cli 0.156.1, the `codex` backend read 20 observed
  results for that query from its events and returned 10 hits, all with real
  snippets.
- The `searxng` backend was trialled live on four query shapes, in
  [`evals/`](evals/searxng-2026-09-25.md). Each query filled a pool of 50 deduped
  candidates with real snippets in at most 7 requests. It also puts the right
  page first on the acceptance query.
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
