# Memory storage integrity

`run_add` prepares writes in a private, user-scoped in-memory SQLite snapshot. Extraction,
governance and summary generation can await providers without exposing partial
writes to searches or holding a transaction open on the live database. Successful
preparation replays its SQL mutations in one `BEGIN IMMEDIATE` transaction,
including FTS, triples, provenance, summary and the request completion ledger.
Errors and cancellation discard the snapshot; publication errors roll back.
A per-user revision detects intervening writes for the same user. Different users
can prepare and commit independently; a same-user conflict is retried internally
up to three times. Same-user Adds through one Store are serialized before preparation.
File databases use WAL and a short `BEGIN IMMEDIATE` publish transaction. This is
a single-node reference design, not horizontal scaling; production uses a database-
level request claim and Postgres/pgvector.

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

Full-content Add and Search logs are disabled by default. Setting
`AML_MEMORY_DEBUG_LOG` or `AML_SEARCH_DEBUG_LOG` to a filename enables local JSONL
diagnostics containing unredacted data; these files must be access-controlled.
`DELETE /memory/{user_id}` removes matching records from enabled logs as well as
the database. The directory is ignored by Git, which is not a privacy control.

Each `memory.add.committed` event includes request/user/session IDs, commit log
time, original request messages, final session summary, and affected `memories`:

- `content` and metadata: saved AMU body and labels.
- `embedding.values`: all stored float32 coordinates, with dimensions and norm.
- `full_text_index`: the actual FTS rows, including body and retrieval key.
- `triples`: saved graph edges for that AMU.
- `sources`: source message IDs, role, original text and timestamp.
- `valid_to` / `supersedes` / `superseded_by`: include both directions of a
  SUPERSEDE chain; graph edges retain the fact's validity interval too.

UPDATE records the final body and regenerated representations. NOOP with a target
records that target and its evidence; without a target, the event still records
request messages and summary. Retries of completed requests emit no duplicate
snapshot. Rolled-back requests emit no committed event. A log write failure emits
a warning without changing an already committed Add's success. This debug file is
best-effort diagnostics, not a transactional audit ledger; a process crash between
database commit and file append can leave a missing event.

Set `AML_MEMORY_DEBUG_LOG` to a custom filename to enable logging. Files append
across runs without automatic rotation.
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
passes the dataset's `question_date`, while online requests default to the user's
latest known memory time (`Store.latest_time`), not the service wall clock.
An explicit time scope removes versions whose validity interval does not overlap;
an unbounded change-history question retains the chain so changes can be explained.

`sessions_fts` adds BM25 recall for summaries using original/expanded queries.
Opening an older database backfills its existing summaries; summary writes update
this index atomically. Summary candidates have stable `summary_` IDs and type
`session_summary`, and are explicitly marked as derived context during reranking
and answering. Per the ULM design (§3.1) the rolling summary is extraction context
only, so this route is disabled unless `AML_SUMMARY_ROUTE=1`.

## ULM lifecycle (segments, MemCells, MemScenes, heat, forgetting)

Each Add is cut into topic segments by embedding-similarity drops
(`AML_SEGMENT_SIM_DROP`, `AML_SEGMENT_MIN_MESSAGES`, capped by
`AML_EXTRACT_BATCH_MESSAGES`). A segment yields one MemCell: an `episode` AMU
holding an extracted third-person narrative and compressed view (unless
`AML_STORE_EPISODES=0`) plus its atomic facts and triples, all sharing a `cell_id`.
Raw messages remain in source references. Episodes bypass governance.

Governance is preceded by a novelty gate: a neighbour with cosine ≥
`AML_NOVELTY_DUP_THRESHOLD` and identical normalised text is a NOOP without an LLM
call; facts with no close neighbour are ADDed without one. `UPDATE` merges text,
keeps the union of both memories' entities/keywords/triples and re-embeds once.
NOOP/UPDATE targets record the contributing session in `support_sessions`; profile
items supported by ≥ `AML_PROFILE_STABLE_SESSIONS` sessions are persisted as
`profile_status=stable`; preferences start as transient with
`AML_PROFILE_TRANSIENT_TTL_DAYS`, while explicit profile facts are static and rules
are always recalled.

Cells are clustered into `scenes` rows (0.7·cosine + 0.3·keyword Jaccard ≥
`AML_SCENE_JOIN_THRESHOLD`) with a running centroid, keyword union and a
deterministic line summary (`AML_SCENE_SUMMARY_LLM=1` uses the model). Search adds
a scene->cell route: top `AML_SCENE_TOP_M` scenes by query similarity, then the
best cells inside them. Scene heat uses log-saturated visits/interactions plus
recency and recency-decayed surprise; scenes above
`AML_HEAT_PROMOTE_THRESHOLD` reset their interaction count (promotion), and scenes
beyond `AML_MAX_HOT_SCENES` are cold-tiered. Recall updates `recall_count`,
`strength`, scene `visit_count` with relative SQL updates and never bumps the user
revision, so it cannot invalidate a concurrent staged Add.

Forgetting uses `R = exp(-age / (30 days · strength))`; each recall adds
`AML_FORGET_RECALL_BONUS_DAYS / AML_FORGET_STRENGTH_DAYS` to the dimensionless
strength multiplier. With
`AML_FORGET_THRESHOLD > 0` memories below it move to `tier='cold'`, which the
dense and sparse routes skip (graph/temporal routes still reach them). A cold
memory that is recalled returns to `hot`, as does its Scene. Nothing is deleted
except through the explicit compliance purge endpoint.

The search loop runs up to `AML_SEARCH_MAX_ROUNDS` (default 2): after reranking,
prompt 08 judges whether the evidence is necessary and sufficient; if not, its
follow-up queries drive one more recall round. If the final verdict is
insufficient with confidence below `AML_ABSTAIN_CONFIDENCE`, the search returns an
empty list. Verifier failures fail open. `plan`-type memories (foresight) are
prefixed with `[plan; status: pending|expired]` relative to the anchor time and
are dropped when a query time scope does not intersect their window.

Candidates are LLM-scored only for the fused head (`AML_RERANK_MAX_CANDIDATES`,
default 40, one batch of `RERANK_CANDIDATES`). Kept items are ordered by relevance;
the unscored tail follows in fusion order (`unscored_fused`) so `top_k=100` stays
full without penalising unscored evidence.
Explicit scores below `AML_SEARCH_MIN_RELEVANCE` (default 0.3) are rejected. If
any batch fails or omits a candidate, the entire request falls back to one coherent
RRF ordering; relevance and RRF scales are never mixed.

Search items add `memory_type` and structured `sources`. Source bodies are
deduplicated across results and bounded by per-item and total character budgets;
omitted bodies retain request/message references and an omission reason. The
response reports `source_count` and caps returned references separately. Full
sources remain in SQLite and debug logs. Summary sources cover the same user's
session. Older memories without source records return an empty list.

Fact, multi-hop, temporal and profile intents use Memory-Only: episode candidates
are removed when atomic evidence exists, and source references omit their bodies.
Narrative/document intents use Memory-Doc: episode candidates are preferred and
bounded source text is included. Sensitive AMUs are absent from every route unless
both `AML_SENSITIVE_RECALL_ENABLED=1` and request `include_sensitive=true`.

`POST /feedback` is the only task-experience write path. It requires an explicit
success/failure signal and distills at most three strategy/workflow/skill/playbook
items; only environment-verified skills receive `verified=true`.

When `AML_SEARCH_DEBUG_LOG` is explicitly configured, each search appends a UTF-8
JSONL event to that path. Each event includes:

- `search_id`, query, user, options, requested top_k, timestamps and status.
- `plan` and the effective `include_history` decision.
- `routes`: named channels, query variants and candidates with full text, rank
  and channel scores when available (including empty routes).
- `fused`: deduplicated candidates with RRF scores and fusion ranks.
- `rerank`: per-candidate batch, score, keep flag and decision reason;
  `rerank_rejected`, `global_rrf_fallback`, `unscored_fused`, or `kept`.
- `rounds`: per-round query set, fused/ranked counts and the verifier verdict;
  `abstained` marks an empty response caused by the verifier.
- `scenes`: the top MemScenes chosen for the scene->cell route;
  `foresight_dropped`: plan memories outside the query time scope.
- `anchor_time`: the reference time used for relative dates.
- `pre_rerank_excluded`: the fusion tail omitted by the configured cost budget.
- `top_k_excluded`: relevant candidates omitted only because of the result limit.
- `returned`: final response items including source evidence.

Failed searches record error type and any diagnostics collected before failure.
Log I/O errors warn without failing retrieval. Logs append across runs, contain
full content, and are excluded from Git by the existing logs-directory rule.
Run `python scripts/selftest_search.py` for the offline search regression suite and
`python scripts/selftest_ulm.py` for the lifecycle (segments/scenes/heat/verifier) suite.
