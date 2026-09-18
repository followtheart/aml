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

真實 embedding 失敗會中止並回滾 Add；Search 則保留詞面召回並回報向量路徑降級，兩者均不偷偷生成假向量。
Each AMU records `embedding_space` (model, dtype and
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
passes the dataset's `question_date`, and online requests without it fall back
to the receive-time wall clock — never to the user's latest memory time
(ULM §5.1: the anchor must not be inferred from future messages). Undated
histories stay undated: no synthetic timeline is fabricated and messages keep
`timestamp=NULL` (ULM §2.1).
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
holding the original conversation with message roles and an optional compressed view (unless
`AML_STORE_EPISODES=0`) plus its atomic facts and triples, all sharing a `cell_id`.
Raw messages also remain in source references. Episodes bypass governance.
Their observed status describes the conversation, not the truth or ownership of
every quoted statement. A grounding failure does not itself mark the raw episode
sensitive; explicit model/source privacy annotations still apply.

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
dense and sparse routes can skip. Single-pass Search explicitly includes cold
memories alongside hot ones, without a verifier-driven fallback. A cold
memory that is recalled returns to `hot`. Search 不更新 Scene 訪問熱度；顯式 feedback
會更新對應場景。冷場景及其中的冷 AMU 現在也能參與同輪場景召回。Nothing is deleted
except through the explicit compliance purge endpoint.

Search 使用 `graph_cascade_v7`：一次規劃、多路召回、至多一次有依據的多跳補查、圖融合、粗排／Cross-Encoder／列表排序及一次裝包。
規劃新增與選項順序對齊的 `option_queries`，通用建議用空字串；缺失或無效項目回退到
選項前提／首句。原問題、所有選項與最多三個子問題保留，六條軟上限僅限制額外擴展。
選項始終是待驗證假設，不是來源證據。
選擇題不注入畫像 digest；無選項問題只注入有詞面關聯的畫像。
向量／原文／全文／畫像預設配額分別為 40／40／20／8，使用者規則另行保留，均不因 top_k 增加。
多跳、敘事、文件，或任一查詢匹配的獨立原始來源不足 3 個時，啟用 graph／scene，
合計不超過 12 條；冷資料仍可由直接路徑命中。配額可透過 `AML_RECALL_*` 調整。
多跳補查只可使用原查詢和首輪種子證據中已見詞彙，必須附種子與需求 ID，預設最多兩條。
新命中需保留連接種子及必要依賴，補查失敗或預算不足不抹去首輪候選。
不執行充分性模型、無限補查迴圈或 persona 措辭過濾。
`evidence_status=retrieved` means a nonempty packet was returned, not that its
sufficiency was verified; `verification_status=not_run`. Empty packets report
`not_found`（無已知降級）；降級且無證據時回報 `incomplete`，含爭議記憶時回報 `conflicting`。
`plan`-type memories (foresight) are
prefixed with `[plan; status: pending|expired]` relative to the anchor time and
are dropped when a query time scope does not intersect their window.

圖融合先以內容先驗及查詢代表限制候選，才讀取該使用者的來源／依賴／三元組 metadata。
`idx_triples_candidate(user_id, amu_id, id)` 支援按候選索引讀取；每候選最多 8 條、合計最多 512 條，
以輪流准入避免第一個高扇出節點吃完配額。歷史、敏感、撤回、抑制及 stale 過濾在建立關聯前執行。
圖資料與來源均由同一 Search snapshot 讀取，跨使用者或失效依賴不可成邊。
候選圖最多 256 節點（規則另行保留）、4,096 條邊；關係方向及依賴版本保留，反向傳播較弱。
相同來源不構成額外佐證；直接與間接路徑重複命中也不累加融合票數。
關係鏈最多三個記錄節點，橋接來源會在重排前納入固定證據單元。

相同偏好及來源／內容／狀態相同的 episode 仍合併，保留 equivalent_ids。
單元預設有效上限 4,000 UTF-8 bytes（Search、CE 及包預算的最小值）。
來源挑選保留說話者、角色、時間、完整必要引用、原文 offsets 和依賴版本；不合要求的單元記錄 unit_omitted。
在模型排序後僅可移除包內已可見的重複來源片段，不新增模型未讀過的證據。

級聯預設粗排 50、CE 精排保留 12、LLM 全列表比較 10，需求代表占有界保留位置；`top_k` 不擴大模型池。
CE 使用真正的提供者 Rerank API，每批 24、並發 2、請求 10 秒、階段 12 秒。
每個返回索引須唯一、完整且分數有限；保留原始 logits，不作機率閾值或跨批 LLM 分數校準。
成功批次保留結果，少量失敗項可進入 LLM 列表比較。LLM 返回完整 permutation、irrelevant 和 disjoint groups，
只對格式錯誤最多修復一次。輸入上限、token 預留、呼叫數、截止時間和取消均受共同 Search 預算約束。
所有不可用階段都有錯誤診斷；列表失敗回退 CE 順序，CE 也無結果時回退圖先驗並限制未評分項數。

打包先按列表順序准入完整證據組，最後仍維持原列表相對順序；後來加入的項目不能使已准入組因去重而拆散。
無法整組容納時以 `atomic_group_top_k/token_budget/redundant` 記錄；組員被時間範圍或 stale 過濾時以
`atomic_group_scope` 省略整組。明確 irrelevant 項與未選尾部不會補回，需求缺口仍可觀察。
使用者指令與遺忘規則保留，權限／版本／來源及字節預算檢查照常執行。

`coverage_manifest.selection_mode=graph_cascade`。`fusion` 記錄圖大小與權重，
`cascade.coarse/cross_encoder/fine/listwise` 記錄各階段候選、批次、錯誤與選中順序；
`evidence_groups` 記錄最終完整組，`requested_evidence_groups` 記錄模型要求的組。
`rerank_status` 為 `ok/recovered/partial/fallback/not_run`；`search_degraded` 標示未恢復的失敗。
設定及部署說明見 [GRAPH_CASCADE_IMPLEMENTATION.md](GRAPH_CASCADE_IMPLEMENTATION.md)。

`source_fts` indexes immutable source messages and is backfilled on opening old
databases. Recall joins through `amu_sources` to enforce user ownership, valid
versions, view readiness, retractions, sensitivity and cold-tier policy. Snapshot
reads use the projected source links. Purge removes the source index too.
This migration never clears old sensitive labels or rewrites old episodes.
全文與原文路徑共用停用詞及中文相鄰雙字詞；中英文混合查詢保留兩種文字。
中文候選先依命中詞數排序再套用上限。原文路徑對不同 AMU 數量設限，
避免單條記憶的大量來源連結耗盡 SQL 列數上限。
Graph traversal can seed from retrieved AMUs when planner concepts do not match
entity labels. 圖過濾保留最多三跳的連接路徑，上限 120 條三元組；正向邊權重高於反向邊，關係詞匹配亦參與邊權重。
場景召回輪流考慮所有查詢的場景排名，避免單一查詢占滿場景額度。
The profile route sorts by query overlap rather than insertion order and does
not boost assistant-only inferred advice as user rules.
無詞面關聯的偏好不獲額外畫像路徑加權，但仍可透過向量等路徑召回。
融合候選在去重及評分前補齊原始 AMU 的狀態，確保 disputed／inferred 標記不丟失。

Search items add `memory_type` and structured `sources`. Source spans are
deduplicated across results; omitted bodies retain request/message references
and an omission reason. Validated support quotes are retained when choosing
excerpts. Whole ordinary evidence items compete for `top_k` and the packet's UTF-8 byte budget,
without clipping their supporting quotes. The response reports `source_count`;
full sources remain in SQLite and debug logs. Summary sources cover the same
user's session. Older memories without source records return an empty list.

原子事實及 episode 均可召回。長文件使用完整原文句段，保留 offsets、來源角色、局部語境和必要引文；相關的 assistant 草稿亦可使用。
重排後只可移除包內已可見的重複來源片段，不再追加重排未讀過的依賴。
若 episode 本次選取的來源片段全部已在包內，且沒有額外片段，則以
`redundant_episode` 排除；較寬來源範圍不會因較短引用已出現而被排除。
Forget rules are packed before other candidates and consume the shared byte
budget, but not ordinary evidence slots. The response can exceed `top_k` by the
number of included constraints; `coverage_manifest` reports `evidence_count`,
`constraint_count` and `constraint_bytes`. Identical preferences with compatible
validity/privacy/status metadata share one item, retaining their source union
and `equivalent_ids`. Hash v4 在 v3 的合併 ID 與約束分類之外，新增覆蓋 ID 與分數類型；v1/v2/v3 仍可讀取。Core profiles do
not bypass ranking through a second injection path. Sensitive AMUs are absent from every route unless
both `AML_SENSITIVE_RECALL_ENABLED=1` and request `include_sensitive=true`.

`coverage_manifest.coverage` 將問題／選項／子問題分為 `packed`、`candidate_only`、`missing`。
標記依實際證據單元重算；補查標記還要求目標文字可見且連接種子仍在。
這是詞面或有依據的補查覆蓋，不代表語義充分，也不能證明選項為真；`missing_evidence` 列出未裝包需求。

普通檔案搜尋使用獨立的唯讀 WAL 交易，不再複製整個使用者資料庫。快照讀取 API 檢查使用者範圍；
記憶體資料庫與歷史 `as_of` 查詢仍會建立私有投影。純向量快取按資料庫 ID、使用者、向量空間、
刪除 epoch 和 SQLite trigger 維護的向量代數隔離。合格 ID 每次從當前快照重查，只補齊入選 AMU。
Purge 清除快取，也阻止舊快照稍後完成的索引重新寫入快取；快照清理後再次檢查即時刪除 epoch 才交付結果。

預設向量搜尋仍精確，使用快取向量和分塊 top-k，避免每次載入所有 AMU 正文及全量排序。
`AML_VECTOR_APPROXIMATE=1` 才會在超過 `AML_VECTOR_EXACT_LIMIT` 時啟用有界的隨機超平面候選搜尋；
它可能漏掉鄰居，啟用前需以部署資料評估。診斷包含近似模式、合格／評分數及候選搜尋預算耗盡狀態。
首次索引、合格 ID、精確距離計算及歷史投影仍隨資料量增長。同步工作移入有執行緒數上限的背景池，
透過協作式截止檢查和 SQLite 中斷停止工作；取消會等待工作停止再關閉快照，因此清理可能超出名義截止時間。

實作對照、可重現驗證及限制：
[SEARCH_PIPELINE_REPAIRS.md](SEARCH_PIPELINE_REPAIRS.md).

`POST /feedback` is the only task-experience write path. It requires an explicit
success/failure signal and distills at most three strategy/workflow/skill/playbook
items; only environment-verified skills receive `verified=true`.

When `AML_SEARCH_DEBUG_LOG` is explicitly configured, each search appends a UTF-8
JSONL event to that path. Each event includes:

- `search_id`, query, user, options, requested top_k, timestamps and status.
- `plan` and the effective `include_history` decision.
- `query_specs`：實際短查詢、規劃／回退來源及 option_index。
- `expansion`：是否啟用 graph／scene、原因、直接命中數與合計上限。
- `routes`: named channels, query variants and candidates with full text, rank
  and channel scores when available (including empty routes).
- `fused`：去重候選、內容先驗、圖訊號及融合排名；不是 RRF。
- `fusion`、`fusion_edges`、`fusion_triples`：圖大小、權重及有來源的有向關聯。
- `cascade`：粗排、CE 批次映射、精排池、列表輸入／輸出及格式修復記錄。
- `rerank`：相容診斷，保留選中候選的 CE 原始分數或 null；分數類型由候選 `_score_kind` 說明。
- `rerank_pool_size`、`rerank_errors`、`deduplicated` 和 `graph_seed_ids`。
- `selection`：`graph_cascade` 模式、選中 ID、列表順序與各階段省略原因。
- `evidence_groups`：模型要求共同使用的組。回應 manifest 區分要求組及實際完整裝包組。
- 舊 trace 的 `rerank_batches/rerank_reserved_ids` 僅供歷史回放；v7 詳細資料使用 `cascade`。
- `rounds`: one query set, fused/ranked counts and `verification_status=not_run`;
  `abstained` marks an empty packet, not a model sufficiency decision.
- `scenes`: the top MemScenes chosen for the scene->cell route;
  `foresight_dropped`: plan memories outside the query time scope.
- `anchor_time`: the reference time used for relative dates.
- `top_k_excluded`: candidates omitted because of the result limit (not necessarily relevant).
- `returned`: final response items including source evidence.

Failed searches record error type and any diagnostics collected before failure.
Log I/O errors warn without failing retrieval. Logs append across runs, contain
full content, and are excluded from Git by the existing logs-directory rule.
Run `python scripts/selftest_search.py` for the offline search regression suite and
`python scripts/selftest_ulm.py` for the lifecycle (segments/scenes/heat/verifier) suite.
