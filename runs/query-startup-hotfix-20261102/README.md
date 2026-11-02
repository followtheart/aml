# First-query SDK initialization hotfix

The query-understanding stage previously wrapped a coroutine containing the
first synchronous LiteLLM import in an eight-second timeout. A fresh-process
diagnostic measured 7.091 seconds for import and entry-point resolution. In the
preceding isolated HTTPS run, the provider call itself was cancelled after
1.807 seconds, while query understanding fell back. This supports SDK cold
initialization consuming most of the stage budget.

The ASGI lifespan now loads the provider SDK and resolves its completion and
embedding entry points in a worker thread before declaring startup complete.
It sends no model inference request. Fake mode skips initialization; dependency
failures prevent readiness. The query stage retains its eight-second default,
now configurable through `AML_QUERY_UNDERSTAND_TIMEOUT_SECONDS`, capped by the
remaining overall search budget. Stage errors still fall back while budget
exhaustion and external cancellation propagate. Logs include exception type,
stage elapsed time and effective stage limit without user text or credentials.

Validation:

- Nine new tests cover readiness gating, nonblocking SDK imports, startup
  failure, fake mode, no inference during initialization, successful plans,
  stage cancellation/fallback, remaining budget, and external cancellation.
- Thirteen selected tests passed including the source-join regressions and
  existing reference-time and search-deadline checks.
- Two independent fresh processes using the real model completed their first
  planning request without fallback in 5.255 and 3.938 seconds. SDK startup
  took 4.761 and 5.952 seconds separately. Each used one provider call.
- An isolated ASGI server loaded the SDK in 4.624 seconds before startup
  completion. Its first HTTPS search's planning call completed in 4.398 seconds
  without fallback. The full request returned HTTP 200 in 157.859 seconds with
  12 items, two missing-evidence entries, 10 provider calls and 45,332 reported
  tokens. This does not establish answer accuracy or complete evidence.
  The full response metadata is recorded in `isolated-https.json`.

The HTTP replay uses the previous production-snapshot copy, verified TLS and
real providers. Its pre-planning database anchor-time read was also observed
to be slow; this patch addresses SDK initialization, not all cold database
reads or provider/network delays. These limited probes do not guarantee every
future query completes within eight seconds and are not accuracy measurements.

Production service/configuration were not changed or restarted. Results and
logs in this directory contain summary metadata, not the real query text.
