"""Inventory exact query/document reuse from audited complete-source CE calls.

Read-only source inputs, no labels and no provider calls. Pair-level identity
does not imply batch-level identity; both are recorded independently.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import config, cross_encoder
from probe_source_ce import build_views, records


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory(run, archive, cohort):
    audit = json.loads((archive/'audit.json').read_text())['cohorts'][cohort]
    for suffix, digest in audit['artifact_hashes'].items():
        if sha(archive/f'{cohort}.{suffix}') != digest:
            raise ValueError('Archived CE artifact changed: '+suffix)
    plan = json.loads((archive/f'{cohort}.plan.json').read_text())
    if plan['source_policy'] != 'all_eligible_complete_user_sources':
        raise ValueError('Incomplete source policy')
    old = {j['qa_id']: j for j in plan['jobs']}
    live = records(archive/f'{cohort}.live.jsonl')
    scores = {}; usage = {}
    for event in live:
        qa = event['qa_id']; job = old[qa]
        if event.get('error') or event['view_ids'] != job['batches'][event['batch']]:
            raise ValueError('Failed or mismatched archived request')
        if len(event['view_ids']) != len(event['scores']):
            raise ValueError('Incomplete scores')
        views = {v['id']: v for v in job['views']}
        if event['request_bound'] != cross_encoder.request_size(job['query'], [views[k]['text'] for k in event['view_ids']]):
            raise ValueError('Archived request changed')
        for key, value in zip(event['view_ids'], event['scores']):
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError('Invalid score')
            scores[qa, key] = value
        u = usage.setdefault(qa, dict(calls=0, tokens=0))
        for k in u:
            u[k] += event[k]
    if len(scores) != sum(len(j['views']) for j in old.values()):
        raise ValueError('Archived source coverage incomplete')
    manifest = json.loads((run/'isolated-manifest.json').read_text())
    if manifest.get('audit_exit_code') != 0 or 'server_exit_code' not in manifest:
        raise ValueError('Current cohort not audited and terminal')
    if manifest['data_sha256'] != plan['prepared_sha256'] or sha(Path(manifest['data'])) != plan['prepared_sha256']:
        raise ValueError('Prepared dataset changed')
    results = records(run/'completed/data/results/personamem-v2-32k.jsonl')
    by_search = {r['search_id']: r for r in results}
    if len(results) != manifest['questions'] or len(by_search) != len(results):
        raise ValueError('Incomplete current results')
    jobs = []
    for t in records(run/'server-search-debug.jsonl'):
        r = by_search[t['search_id']]; qa = r['qa_id']; prior = old[qa]
        if r.get('error_stage') or r['packet_hash'] != t['packet_hash'] or r['user_id'] != t['user_id']:
            raise ValueError('Current trace identity mismatch')
        ce = t['cascade']['cross_encoder']
        if ce['status'] != 'ok' or ce['model'] != plan['ce_model']:
            raise ValueError('CE model/status mismatch')
        byid = {c['id']: c for c in t['ranked']}
        coarse = [byid[k] for k in t['cascade']['coarse']['selected_ids']]
        with patch.multiple(config, **t['versions']['settings']):
            views, members, batches, omitted = build_views(coarse, t['user_id'], ce['query'], max_sources=None)
        prior_texts = {}
        for view in prior['views']:
            prior_texts.setdefault(view['text'], []).append(view['id'])
        mapping = {}; missing = []; ambiguous = []
        for view in views.values():
            keys = prior_texts.get(view['text'], []) if ce['query'] == prior['query'] else []
            values = {scores[qa, k] for k in keys}
            if len(values) == 1:
                mapping[view['id']] = dict(archived_ids=keys, score=next(iter(values)))
            elif values:
                ambiguous.append(view['id'])
            else:
                missing.append(view['id'])
        prior_views = {v['id']: v for v in prior['views']}
        old_batches = [[prior_views[k]['text'] for k in batch] for batch in prior['batches']]
        same_batches = ce['query'] == prior['query'] and [[v['text'] for v in b] for b in batches] == old_batches
        jobs.append(dict(qa_id=qa, search_id=t['search_id'], query=ce['query'], query_equal=ce['query']==prior['query'],
            views=list(views.values()), members=members, batches=[[v['id'] for v in b] for b in batches], omitted=omitted,
            reused=mapping, missing=missing, ambiguous=ambiguous, exact_batch_sequence=same_batches,
            archived_usage=usage[qa]))
    if len(jobs) != len(results) or {j['qa_id'] for j in jobs} != set(old):
        raise ValueError('Question set changed')
    summary = dict(cohort=cohort, questions=len(jobs), provider_calls=0,
        exact_query_questions=sum(j['query_equal'] for j in jobs),
        complete_reuse_questions=sum(not j['missing'] and not j['ambiguous'] for j in jobs),
        exact_batch_sequence_questions=sum(j['exact_batch_sequence'] for j in jobs),
        views=sum(len(j['views']) for j in jobs), reused_views=sum(len(j['reused']) for j in jobs),
        missing_views=sum(len(j['missing']) for j in jobs), ambiguous_views=sum(len(j['ambiguous']) for j in jobs),
        current_batch_calls=sum(len(j['batches']) for j in jobs),
        incomplete_questions=[j['qa_id'] for j in jobs if j['missing'] or j['ambiguous']])
    paths=[Path(__file__),Path(__file__).with_name('probe_source_ce.py'),archive/'audit.json',
           run/'server-search-debug.jsonl',run/'completed/data/results/personamem-v2-32k.jsonl',run/'isolated-manifest.json']
    return dict(summary=summary, jobs=jobs, run=str(run.resolve()), archive=str(archive.resolve()),
        input_hashes={str(p.resolve()):sha(p) for p in paths},
        limitation=__doc__, archived_hashes=audit['artifact_hashes'])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--archive', type=Path, default=Path('runs/all-user-source-ce-20261024'))
    p.add_argument('--cohort', choices=['regression','validation'], required=True)
    p.add_argument('--output', type=Path, required=True)
    a=p.parse_args()
    with patch('socket.socket.connect', side_effect=AssertionError('Offline inventory')):
        result=inventory(a.run,a.archive,a.cohort)
    with a.output.open('x') as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2);stream.write('\n')
    print(json.dumps(result['summary']))
