"""Measure SDK startup and the first real planning call in a fresh process."""
import asyncio
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
runtime = json.loads(Path('/root/memory-system-runtime/service.json').read_text())
os.environ.update(runtime['env'])
sys.path.insert(0, '/root/aml/memory_system')
from app import budget, config, llm, provider_startup, schemas, search_pipeline

logging.basicConfig(level=logging.INFO)
request = schemas.SearchRequest(**json.loads(Path(
    '/root/aml/runs/https-status-20261102-1503/private-probe-request.json').read_text()))
assert request.options  # no profile database access for this real planning fixture


async def main():
    report = {'started_at': datetime.now(timezone.utc).isoformat(),
              'stage_limit_s': config.QUERY_UNDERSTAND_TIMEOUT_SECONDS,
              'scope': 'fresh process; real first query-understanding call after SDK-only startup preparation'}
    started = time.monotonic()
    async with provider_startup.lifespan(None):
        report['sdk_startup_s'] = round(time.monotonic() - started, 3)
        planned = time.monotonic()
        with budget.scope(seconds=300, calls=20, tokens=128000) as limits:
            plan = await search_pipeline._understand(None, request, '2026-11-02T00:00:00Z')
            report.update(planning_s=round(time.monotonic()-planned, 3),
                          planning_error=plan.get('_planning_error'),
                          intent=plan.get('intent'), sub_queries=len(plan.get('sub_queries', [])),
                          provider_calls=limits.calls, provider_tokens=limits.tokens)
    report['passed'] = report['planning_error'] is None and report['provider_calls'] > 0
    (ROOT / (sys.argv[1] + '.json')).write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


asyncio.run(main())
