"""Audit completed experiments' cross-encoder coverage and measured call cost.

Only reads archived files. Provider durations are summed service time, not wall
time or money; scheduled batches can differ from actual calls after budget errors.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re


def audit(run):
    if (run / 'exit-code').read_text().strip() != '0':
        raise ValueError(f'Experiment did not finish successfully: {run}')
    log = run / 'completed/logs/search-debug.jsonl'
    output = run / 'output.log'
    raw = log.read_bytes()
    searches = [json.loads(line) for line in raw.splitlines() if line.strip()]
    rows, anomalies = [], []
    for search in searches:
        ce = search['cascade']['cross_encoder']
        settings = search['versions']['settings']
        coarse = search['cascade']['coarse']['selected_ids']
        indexed = {c['id']: c for c in search['ranked']}
        finite = [mid for mid in coarse if type(indexed[mid].get('_ce_score')) in (int, float)
                  and math.isfinite(indexed[mid]['_ce_score'])]
        if len(finite) != ce['scored_count'] or len(coarse) != ce['input_count']:
            anomalies.append(dict(search_id=search['search_id'], reason='score_count_mismatch'))
        for batch in ce['batches']:
            if (batch['candidate_count'] > settings['CE_BATCH_SIZE']
                    or batch['input_bound'] > settings['CE_MAX_REQUEST_BYTES']):
                anomalies.append(dict(search_id=search['search_id'], batch=batch['batch'],
                                      reason='configured_batch_limit_exceeded'))
        rows.append(dict(search_id=search['search_id'], status=ce['status'],
            candidates=len(coarse), scored=len(finite), batches=len(ce['batches']),
            batch_status=dict(Counter(b['status'] for b in ce['batches'])),
            document_submissions=sum(b['candidate_count'] for b in ce['batches']),
            max_input_bound=max((b['input_bound'] for b in ce['batches']), default=0),
            settings={key: settings[key] for key in ('CE_BATCH_SIZE', 'CE_MAX_REQUEST_BYTES',
                'CE_DEADLINE_SECONDS', 'RERANK_CONCURRENCY', 'SEARCH_MAX_CALLS')}))
    calls = []
    output_raw = output.read_bytes()
    for line in output_raw.decode('utf-8').splitlines():
        if 'provider_call ' not in line:
            continue
        fields = dict(re.findall(r'\b(\w+)=([^\s]+)', line))
        if fields.get('kind') == 'rerank' and fields.get('stage', '').startswith('search.cross_encoder'):
            calls.append(fields)
    return dict(run=str(run.resolve()), searches=len(rows),
        candidates=sum(r['candidates'] for r in rows), scored=sum(r['scored'] for r in rows),
        complete_searches=sum(r['candidates'] == r['scored'] for r in rows),
        scheduled_batches=sum(r['batches'] for r in rows), actual_provider_calls=len(calls),
        provider_status=dict(Counter(c['status'] for c in calls)),
        provider_document_submissions=sum(int(c.get('input_count', 0)) for c in calls),
        provider_seconds_sum=sum(float(c['duration_s']) for c in calls),
        source_sha256={str(log): hashlib.sha256(raw).hexdigest(),
                       str(output): hashlib.sha256(output_raw).hexdigest()},
        anomalies=anomalies, rows=rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    reports = [audit(run) for run in args.runs]
    if args.output:
        args.output.write_text(json.dumps(reports, ensure_ascii=False, indent=2) + '\n')
    for report in reports:
        print(json.dumps({key: value for key, value in report.items()
                          if key not in ('rows', 'source_sha256')}, ensure_ascii=False))
    raise SystemExit(int(any(r['anomalies'] for r in reports)))
