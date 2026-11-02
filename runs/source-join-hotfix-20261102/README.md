# Source-link join hotfix

Branch: `v4.0.2-261029.0-hotfix`. Control code: commit `4e00cab`, preserved in
`baseline_store.py`. The production service was not restarted or migrated.

The failing source-text query used FTS5 MATCH, then scanned all memories of a
user for each matching source before checking their links. The affected user
had 7,823 memories. The hotfix creates a covering index on
`amu_sources(request_id,message_index,amu_id)` during ordinary Store startup,
selects that index for source links, and uses SQLite CROSS JOIN ordering to
look up linked memories by their primary key. Existing privacy/history/cold
filters, BM25, ordering, source diversity, and post-deduplication limits remain
in the same code paths. No SQL LIMIT is added before diversity selection.

The index is created idempotently even when both FTS backfill switches are off.
On the isolated full database it took 14.575 seconds to create; file growth was
about 74 MiB after checkpointing. These observations are not deployment SLAs.

## Validation

- Two new regression tests pass: adding 2,000 unrelated user memories does not
  multiply SQL work or change source results (English and Chinese); opening an
  old database twice migrates its source-link index without changing data,
  revisions, or snapshot results, with both FTS backfills disabled.
- 384 synthetic combinations exactly match the original source_search method,
  comparing complete returned dictionaries, scores, content and order. These
  include permissions, history, cold/stale/retracted records, shared sources,
  malformed cross-user links, diversity on/off, and k=1/2/5/50.
- The 78 existing tests selected for broader regression checking produce the
  identical 14 failure entries (including subtests) before and after the fix,
  with zero errors and no new failure entries. These pre-existing failures are
  mostly legacy ranking/packet expectations; they were not changed here.

## Production-snapshot measurements

`prepare_snapshot.py` made a pinned read-only copy of the full production
database, preserving the full FTS corpus and BM25 statistics. All mutations,
index creation, replay writes and checkpoints were confined to that copy.

`benchmark.py` compared eight actual query specifications from the failed
request plus three narrow terms present in its sources. Each call had a
20-second diagnostic cap. Candidate calls ran first, then control calls with
the new index removed; caches were not evicted between these arms.

- Control: all eight broad queries reached the cap.
- Candidate: seven broad queries completed in 0.109–5.733 seconds; the first
  broad query also reached the cap. The initial benchmark therefore exited
  unsuccessfully, and its original report/log are retained.
- All three narrow query pairs completed with exactly matching output digests
  (IDs, content, scores and order).
- The subsequent candidate follow-up requested POSIX_FADV_DONTNEED on only the
  isolated copy. fincore confirmed zero resident file pages before the run.
  With a 120-second cap, the first query completed in 46.859 seconds, and the
  remaining seven in 0.106–0.960 seconds. All eleven queries completed; every
  result comparable with the first run matched exactly.

These are retrieval measurements, not answer-accuracy scores. The timed-out
control queries do not provide completed output-equality comparisons for the
eight broad production queries. See the full reports for ordering/cache limits.

## Isolated HTTPS replay

The hotfix was started on loopback port 18680 against the snapshot using the
existing provider settings and verified TLS hostname/certificate. The same
request that previously returned live HTTP 500 after 300.25 seconds returned
HTTP 200 in 162.736 seconds, with 12 items and `evidence_status=retrieved`.
There were 10 recorded provider calls and 46,334 reported tokens. Query
understanding timed out and used its existing passthrough fallback; six missing
evidence entries remained. This is neither a complete-evidence claim nor an
accuracy improvement claim. The earlier live run and isolated run are not
resource-matched timing trials. The isolated service was stopped afterward.

Reports: `validation-summary.json`, `equivalence.json`, `benchmark.json`,
`candidate-recheck.json`, `isolated-https.json`, and the corresponding logs.
Raw requests and the temporary SQLite database are excluded from Git.

Before deployment, retain the configured backfill switches, allow startup to
create the new covering index, and verify live search behavior. Existing
running production workers still use the pre-hotfix query until restarted.
