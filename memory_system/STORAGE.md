# Memory storage integrity

`run_add` prepares writes in a private, user-scoped in-memory SQLite snapshot. Extraction,
governance and summary generation can await providers without exposing partial
writes to searches or holding a transaction open on the live database. Successful
preparation replays its SQL mutations in one `BEGIN IMMEDIATE` transaction,
including FTS, triples, provenance, summary and the request completion ledger.
Errors and cancellation discard the snapshot; publication errors roll back.
A per-user revision detects intervening writes for the same user. Different users
can prepare and commit independently; a same-user conflict raises a retryable Add
error. Same-user Adds through one Store are serialized before preparation, avoiding
repeated model work and ordinary in-process conflicts. File databases use WAL and
a short `BEGIN IMMEDIATE` publish transaction.

## Derived representations

UPDATE regenerates entities, keywords, retrieval text and graph triples from the
final merged body, embeds the final representation, replaces FTS text, and removes
old graph edges. This adds an extraction call and an embedding call on UPDATE.
Historical source links are retained and new evidence links are appended. The
legacy synchronous `update_amu_content` method cannot call a model: it clears the
old vector/metadata/graph and refreshes FTS, rather than retaining stale indices.
Use the Add pipeline for a fully indexed update.

Real embedding failures abort and roll back Add/Search instead of silently
creating hash vectors. Each AMU records `embedding_space` (model, dtype and
dimension); vector search ignores legacy rows whose space is unknown and vectors
from another configured space. Re-embed legacy rows to restore their vector
recall. Hash embeddings remain available only in explicit `AML_FAKE=1` tests.

Each extracted fact owns its nested `triples` array. The model no longer emits
cross-array `fact_index` references. ADD and SUPERSEDE attach only that fact's
triples; UPDATE rebuilds nested triples from the merged body, and NOOP retains
the existing graph.

## Source evidence

`source_messages` stores every incoming message unchanged, keyed by
`(request_id, message_index)`, with user/session IDs, role, text and timestamp.
Each extracted fact requires `evidence` entries containing a local message index
and an exact quote. Code validates the quote against that message, then records
request-relative indices and request IDs. Invalid or missing evidence cannot be
replaced by coarse batch links on an atomic fact. Source links also reject missing
messages and cross-user links.

In real mode, one additional semantic verification call per extraction batch
checks quoted evidence, speaker identity, negation, nested triples and single-valued
state metadata. UPDATE also verifies the merged body against original sources.
This is model-based verification, not a mathematical guarantee of entailment.
A rejected or failed extraction batch is preserved as an unstructured episode,
with its original messages and no inferred graph; the reason is logged as a warning.
Fake mode bypasses only the semantic model call, not quote or interval checks.

## State transitions and time ranges

SUPERSEDE requires explicit `state` metadata on both memories: the same subject
and single-valued attribute, with different values. Events and episodes cannot
supersede anything; old rows without state metadata are conservatively preserved.
Unsafe or backdated model decisions become ADD. UPDATE/NOOP cannot overwrite a
different extracted state. Storage rejects malformed or inverted validity intervals
before modifying them; Add publication remains atomic.

`temporal` records `raw`, `reference_time`, inclusive `start`/`end`, and `precision`.
The model copies the source expression; code resolves supported ISO dates,
weekdays, calendar weeks/months/years and common relative expressions. Month/year
expressions retain their full range. Unsupported/ambiguous expressions retain
raw text with unknown bounds. Missing or conflicting evidence timestamps do not
fall back to the wall clock for event resolution. `event_time` is populated only
for explicitly supported instants. State validity uses observation time when no
exact transition instant is available; it must not be treated as event time.

Temporal recall checks range overlap, and Search returns temporal metadata and
includes its precision in answer context. `temporal`, `state`, and `evidence` JSON
columns migrate automatically on opening older databases. Existing incorrect
records are not rewritten by migration; rebuild them from original messages.

Inspect evidence in Python with `Store.sources_for_amu(amu_id)`, or join
`triples.amu_id -> amu_sources -> source_messages` in SQL. This records model
attribution; it does not independently prove the semantic truth of each triple.

The source tables are created automatically when opening an existing database.
Existing rows are preserved. Old missing source messages and previously incorrect
graph associations cannot be reconstructed automatically: re-ingest original
input into a fresh database to rebuild them. The local evaluation runner still
uses a temporary database and deletes it after evaluation.

## Validation

Run the offline suites from the repository root:

```powershell
memory_system/.venv/Scripts/python.exe memory_system/scripts/selftest_storage.py
memory_system/.venv/Scripts/python.exe memory_system/scripts/selftest_contract.py
memory_system/.venv/Scripts/python.exe memory_system/scripts/selftest_datasets.py
```

## Full-content debug log

By default, successful Add requests append one UTF-8 JSON object per line to
`memory_system/logs/memory-debug.jsonl` (relative to the project, independent of
current working directory). Both the API and local evaluation use this log; it
survives deletion of the evaluation's temporary database. The directory is ignored
by Git. Records contain actual stored data, including unredacted source text and
complete vectors.

Each `memory.add.committed` event includes request/user/session IDs, commit log
time, original request messages, final session summary, and affected `memories`:

- `content` and metadata: saved AMU body and labels.
- `embedding.values`: all stored float32 coordinates, with dimensions and norm.
- `full_text_index`: the actual FTS rows, including body and retrieval key.
- `triples`: saved graph edges for that AMU.
- `sources`: source message IDs, role, original text and timestamp.
- `valid_to` / `supersedes`: include the closed predecessor on SUPERSEDE.

UPDATE records the final body and regenerated representations. NOOP with a target
records that target and its evidence; without a target, the event still records
request messages and summary. Retries of completed requests emit no duplicate
snapshot. Rolled-back requests emit no committed event. A log write failure emits
a warning without changing an already committed Add's success. This debug file is
best-effort diagnostics, not a transactional audit ledger; a process crash between
database commit and file append can leave a missing event.

Set `AML_MEMORY_DEBUG_LOG` to a custom filename, or set it to an empty string in
`.env` to disable logging. Files append across runs without automatic rotation.
`configured_embedding_model` describes the current configuration, not verified
provenance of older or fallback vectors. `fake` distinguishes offline test runs.

```powershell
# Override path before starting the evaluation/API process:
$env:AML_MEMORY_DEBUG_LOG = 'C:\Users\aapoo\Desktop\aml\memory_system\logs\memory-debug.jsonl'
# Watch the last committed Add as formatted JSON:
Get-Content memory_system/logs/memory-debug.jsonl -Tail 1 | ConvertFrom-Json | ConvertTo-Json -Depth 12
```

Regression tests: `python scripts/selftest_memory_debug.py` from `memory_system`.

## Search recall and diagnostics

Historical questions include expired AMUs across vector, FTS, graph and type
recall. Query understanding supplies `include_history`; legacy/fallback plans
also use temporal intent, date bounds and historical wording. The optional Search
request field `include_history` overrides this decision (`true` includes history,
`false` uses current AMUs). Defaults in Store remain current-only for governance.
Temporal bounds also drive a dedicated event/validity interval recall route and
are passed to the reranker. `reference_time` anchors relative dates; local eval
passes the dataset's `question_date`, while online requests default to current UTC.

`sessions_fts` adds BM25 recall for summaries using original/expanded queries.
Opening an older database backfills its existing summaries; summary writes update
this index atomically. Summary candidates have stable `summary_` IDs and type
`session_summary`, and are explicitly marked as derived context during reranking
and answering. Summary recall currently uses text matching, not summary vectors.

Candidates are reranked in parallel batches (`RERANK_CANDIDATES`, default 40),
bounded by `AML_RERANK_MAX_CANDIDATES` (default 200 and never below requested
`top_k`). `top_k=100` can therefore return 100 relevant results when available.
Explicit scores below `AML_SEARCH_MIN_RELEVANCE` (default 0.3) are rejected. If
any batch fails or omits a candidate, the entire request falls back to one coherent
RRF ordering; relevance and RRF scales are never mixed.

Search items add `memory_type` and structured `sources`. Source bodies are
deduplicated across results and bounded by per-item and total character budgets;
omitted bodies retain request/message references and an omission reason. The
response reports `source_count` and caps returned references separately. Full
sources remain in SQLite and debug logs. Summary sources cover the same user's
session. Older memories without source records return an empty list.

Each search appends a UTF-8 JSONL event to
`memory_system/logs/search-debug.jsonl`. Override with `AML_SEARCH_DEBUG_LOG`, or
set it to an empty string to disable. Each event includes:

- `search_id`, query, user, options, requested top_k, timestamps and status.
- `plan` and the effective `include_history` decision.
- `routes`: named channels, query variants and candidates with full text, rank
  and channel scores when available (including empty routes).
- `fused`: deduplicated candidates with RRF scores and fusion ranks.
- `rerank`: per-candidate batch, score, keep flag and decision reason;
  `rerank_rejected`, `global_rrf_fallback`, or `kept`.
- `pre_rerank_excluded`: the fusion tail omitted by the configured cost budget.
- `top_k_excluded`: relevant candidates omitted only because of the result limit.
- `returned`: final response items including source evidence.

Failed searches record error type and any diagnostics collected before failure.
Log I/O errors warn without failing retrieval. Logs append across runs, contain
full content, and are excluded from Git by the existing logs-directory rule.
Run `python scripts/selftest_search.py` for the offline search regression suite.
