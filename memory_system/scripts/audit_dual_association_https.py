"""Audit a complete fresh HTTPS association pair; never call a model.

Requires the preregistration and per-arm registrations written before the
question phase. Reuses the ingestion, source-transport and accounting audits.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import statistics
import subprocess
import sys
from types import SimpleNamespace

import audit_validation_baseline
from compare_https_experiments import compare

ROOT = Path(__file__).resolve().parents[1]


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def audit(directory):
    plan = json.loads((directory / 'preregistration.json').read_text())
    if plan['protocol'] != 'fresh-paired-https-add-search-local-answer':
        raise ValueError('Unexpected preregistration protocol')
    arms, files = {}, {}
    for name in ('control', 'treatment'):
        run = directory / name
        process = json.loads((directory / (name + '-process.json')).read_text())
        if process['exit_code'] not in (0, 1):
            raise ValueError('Launcher failed before completion: ' + name)
        manifest = json.loads((run / 'isolated-manifest.json').read_text())
        if manifest['audit_exit_code'] != 0 or 'server_exit_code' not in manifest:
            raise ValueError('Automatic audit or service shutdown incomplete')
        files[name] = manifest['files']
        subprocess.run([sys.executable, str(ROOT / 'scripts/audit_https_stages.py'),
            '--run', str(run), '--prepared', plan['input']['input']],
            check=True, stdout=subprocess.DEVNULL)
        details = audit_validation_baseline.audit(SimpleNamespace(run=run,
            registration=run / 'audit-registration.json', progress=False,
            gold_csv=ROOT / 'data/raw/personamem-v2.csv'))
        write(run / 'complete-audit.json', details)
        results = read_rows(run / 'completed/data/results/personamem-v2-32k.jsonl')
        traces = read_rows(run / 'server-search-debug.jsonl')
        expected = dict(plan['baseline_versions']['settings'])
        expected.update({k.removeprefix('AML_'): bool(v) for k, v in plan['arms'][name].items()})
        for row in results + traces:
            if row['versions']['settings'] != expected:
                raise ValueError('Unexpected runtime settings: ' + name)
        accesses = read_rows(run / 'server-access.jsonl')
        times = {}
        for route in ('/add', '/search'):
            entries = [r for r in accesses if r['route'] == route and r['method'] == 'POST']
            seconds = sorted(r['duration_ms'] / 1000 for r in entries)
            times[route] = dict(requests=len(entries), statuses=dict(Counter(str(r['status']) for r in entries)),
                total_seconds=sum(seconds), median_seconds=statistics.median(seconds) if seconds else None,
                p95_seconds=seconds[math.ceil(.95 * len(seconds)) - 1] if seconds else None,
                max_seconds=max(seconds, default=None))
        if times['/add']['requests'] != plan['input']['expected_add_chunks'] or times['/search']['requests'] != plan['questions']:
            raise ValueError('HTTPS request inventory differs from preregistration')
        with sqlite3.connect((run / 'memory.db').resolve().as_uri() + '?mode=ro', uri=True) as conn:
            bindings = conn.execute('SELECT COUNT(*) FROM association_views').fetchone()[0]
            views = conn.execute('SELECT COUNT(*) FROM (SELECT DISTINCT user_id,request_id,message_index,view_index FROM association_views)').fetchone()[0]
        if (name == 'control' and bindings) or (name == 'treatment' and not bindings):
            raise ValueError('Write association was not exercised as configured')
        generations = Counter(t['coverage_manifest'].get('query_association', {}).get('status', 'absent') for t in traces)
        if name == 'control' and generations != Counter(absent=len(traces)):
            raise ValueError('Control unexpectedly used query associations')
        if name == 'treatment' and generations.get('ok', 0) == 0:
            raise ValueError('No successful query association in treatment')
        index_batches = []
        for line in (run / 'server.log').read_text().splitlines():
            if 'Association indexing: ' in line:
                index_batches.append(json.loads(line.split('Association indexing: ', 1)[1]))
        sessions = {}
        for cid in {r['conversation_id'] for r in results}:
            rows = [r for r in results if r['conversation_id'] == cid]
            sessions[cid] = dict(questions=len(rows), score=sum(r['score'] for r in rows))
        arms[name] = dict(score=details['summary']['score'], accuracy=details['summary']['accuracy'],
            questions=len(results), terminal_errors=details['summary']['terminal_errors'],
            reader_internal_errors=details['accounting']['reader_internal_errors'],
            search_runtime_errors=details['accounting']['search_runtime_errors'],
            search_degraded=sum(bool(r.get('search_degraded')) for r in results),
            sessions=sessions, timing=times, wall_seconds=process['finished_at'] - process['started_at'],
            association_views=views, association_bindings=bindings, query_association_statuses=dict(generations),
            association_indexing_totals={k: sum(s[k] for s in index_batches) for k in index_batches[0]} if index_batches else {},
            stage_summary=details['summary']['stage_summary'],
            provider_error_counts={kind: sum(stage['statuses'].get('error', 0)
                for stage in details['accounting'][kind].values()) for kind in ('add', 'search', 'reader')},
            provider_usage={kind: {metric: sum(stage[metric] for stage in details['accounting'][kind].values())
                for metric in ('calls', 'tokens')} for kind in ('add', 'search', 'reader')})
    if files['control'] != files['treatment']:
        raise ValueError('Arms used different code snapshots')
    paired = compare(directory / 'control', directory / 'treatment')
    if paired['matched_questions'] != plan['questions'] or paired['partial']:
        raise ValueError('Incomplete pair')
    discordant = paired['wins'] + paired['losses']
    pvalue = (min(1., 2 * sum(math.comb(discordant, k) for k in range(min(paired['wins'], paired['losses']) + 1)) / 2**discordant)
              if discordant else 1.)
    report = dict(questions=plan['questions'], arms=arms, paired=paired,
        accuracy_delta=arms['treatment']['accuracy'] - arms['control']['accuracy'],
        mcnemar_exact_two_sided_p=pvalue,
        same_frozen_files=len(files['control']),
        preregistration_sha256=hashlib.sha256((directory / 'preregistration.json').read_bytes()).hexdigest(),
        auditor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        limitations=plan['limitations'],
        inference='Single paired cohort with independent stochastic ingestion; question outcomes within a history are correlated. The p-value is descriptive, not proof of generalization.')
    write(directory / 'final-audit.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.run.resolve())
    print(json.dumps(dict(questions=report['questions'], arms={k: {field: v[field] for field in
        ('score', 'accuracy', 'terminal_errors', 'query_association_statuses')} for k, v in report['arms'].items()},
        wins=report['paired']['wins'], losses=report['paired']['losses'], accuracy_delta=report['accuracy_delta']), ensure_ascii=False))
