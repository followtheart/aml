# Skip audited startup FTS backfills

On 2026-11-02, the source index audit found all 440,309 source messages
represented exactly once in `source_fts`, with matching ownership and content
hashes. A separate session audit found all 16,346 session summaries represented
exactly once in `sessions_fts`, with matching summary text. See `report.json`
and `session-index-report.json`. Internal FTS5 posting-list integrity was not
checked by these audits.

Added independent, default-enabled startup switches in `memory_system/app/config.py`
and guarded only the corresponding backfill statements in `app/store.py`:

```
AML_SOURCE_FTS_BACKFILL=0
AML_SESSION_FTS_BACKFILL=0
```

Both settings are applied in the `env` object of
`/root/memory-system-runtime/service.json`. The service launcher uses that object
as its complete process environment, so a systemd Environment override alone
does not reach the application. A protected pre-change configuration backup is
`/root/memory-system-runtime/service.json.before-skip-backfill-20261102`.
No credentials are copied into this report.

Temporary-database checks passed for default-enabled behavior, independent
switches, skipping missing rows, preserving original records, and searching
existing source/session FTS records with both switches disabled. `git diff
--check` passed. Normal incremental indexing code was not changed.

The final restart began at 07:32:36 CST. Public HTTPS health returned 200 at
07:32:39, with certificate verification enabled. The actual new process
environment was checked for both settings. See `restart-health-probe.json`
and `restart-search-probe.json` for live endpoint checks. The search probe
replays one recent successful query against existing data; it is not an
accuracy benchmark or evidence of broader retrieval quality.

To re-enable either startup backfill, set its value to `1` in the runtime
configuration and restart `memory-system-https.service`. The original slow
backfill queries are still present and should be optimized before doing so on
this large database.
