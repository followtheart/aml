"""Freeze a completed HTTPS run, then compare ranking on identical inputs.

The SQLite backup is an immutable provenance artifact. Retrieval is NOT rerun:
archived candidates, queries and CE scores are the experimental entry point.
Live comparison uses production cascade/recovery/packing in both arms. Optional
answer scoring is kept separate from ranking and never enters model prompts.
"""
import argparse
import asyncio
import copy
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import sys
import time
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import budget, config, eval_scoring, run_metadata
from replay_https_ranking_budget import run_ranking


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode()).hexdigest()


def records(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def freeze(run, output):
    manifest = json.loads((run/'isolated-manifest.json').read_text())
    if 'server_exit_code' not in manifest or manifest.get('audit_exit_code') != 0:
        raise ValueError('Snapshot requires a completed and audited HTTPS experiment')
    prepared = Path(manifest['data'])
    if digest(prepared) != manifest['data_sha256']:
        raise ValueError('Prepared dataset changed since the experiment')
    qa = {(str(d['conversation_id']), str(q['id'])): q for d in records(prepared) for q in d['qa']}
    traces = {s['search_id']: s for s in records(run/'server-search-debug.jsonl')}
    cases = []
    for result in records(run/'completed/data/results/personamem-v2-32k.jsonl'):
        key = (str(result['conversation_id']), str(result['qa_id']))
        trace = traces[result['search_id']]
        validate_trace(trace)
        cases.append(dict(case_id=':'.join(key), qa=qa[key], trace=trace,
                          input_fingerprint=fingerprint(trace)))
    if not cases or len({c['case_id'] for c in cases}) != len(cases):
        raise ValueError('Empty or duplicate experiment cases')
    output.mkdir(parents=True, exist_ok=False)
    # SQLite backup includes committed WAL pages; a filesystem copy would not.
    source = sqlite3.connect((run/'memory.db').resolve().as_uri()+'?mode=ro', uri=True)
    dest = sqlite3.connect(output/'memory.db')
    try:
        source.backup(dest)
        if dest.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
            raise ValueError('Invalid database snapshot')
    finally:
        source.close(); dest.close()
    (output/'cases.jsonl').write_text(''.join(json.dumps(c, ensure_ascii=False, allow_nan=False)+'\n' for c in cases))
    write(output/'origin.json', dict(source_run=str(run.resolve()), manifest=manifest,
                                   trace_sha256=digest(run/'server-search-debug.jsonl')))
    write(output/'manifest.json', dict(format='experience-replay-v1', cases=len(cases),
          files={name:digest(output/name) for name in ('memory.db','cases.jsonl','origin.json')},
          entry_point='archived candidates and frozen CE; SQLite is provenance, not replayed retrieval',
          execution_kind='replay', observations=False))
    return verify(output)


def validate_trace(trace):
    if trace['cascade']['cross_encoder']['status'] != 'ok':
        raise ValueError('Partial CE results need a separate replay protocol')
    rows = {r['id']: r for r in trace['ranked']}
    if len(rows) != len(trace['ranked']):
        raise ValueError('Duplicate candidate IDs')
    for mid in trace['cascade']['coarse']['selected_ids']:
        score = rows[mid].get('_ce_score')
        if isinstance(score, bool) or not isinstance(score, (float,int)) or not math.isfinite(score):
            raise ValueError('Missing or invalid frozen CE score')


def verify(snapshot):
    manifest = json.loads((snapshot/'manifest.json').read_text())
    if manifest.get('format') != 'experience-replay-v1' or set(manifest['files']) != {'memory.db','cases.jsonl','origin.json'}:
        raise ValueError('Unsupported snapshot manifest')
    for name, expected in manifest['files'].items():
        if digest(snapshot/name) != expected:
            raise ValueError(f'Snapshot integrity mismatch: {name}')
    cases = records(snapshot/'cases.jsonl')
    if not cases or len(cases) != manifest['cases'] or len({c['case_id'] for c in cases}) != len(cases):
        raise ValueError('Invalid case inventory')
    for case in cases:
        if fingerprint(case['trace']) != case['input_fingerprint']:
            raise ValueError('Candidate snapshot changed')
        validate_trace(case['trace'])
    return manifest


async def run_arm(case, staged, *, answer=False):
    trace = copy.deepcopy(case['trace'])
    # Apply recorded search policy, including the SAME total budget in both arms.
    settings = {k:v for k,v in trace['versions']['settings'].items() if hasattr(config,k)}
    settings['LISTWISE_STAGED'] = staged
    start = time.monotonic()
    with patch.multiple(config, **settings):
        row = await run_ranking(trace, config.RERANK_MAX_PROMPT_BYTES,
                                shared_sources=config.LISTWISE_SHARED_SOURCES)
        row['ranking_seconds'] = time.monotonic()-start
        if answer:
            with budget.scope(seconds=240, calls=20, tokens=128000) as limits:
                prediction, score, diagnostics = await eval_scoring.evaluate(case['qa'], row['packet'])
                row.update(prediction=prediction, score=score, answer_diagnostics=diagnostics,
                           answer_usage=dict(calls=limits.calls, tokens=limits.tokens))
    return row


def admission(row):
    cascade = row['cascade']
    return [cascade['coarse']['selected_ids'], cascade['fine']['selected_ids'],
            cascade['listwise']['candidate_ids']]


async def compare(snapshot, output, *, execute=False, repeats=3, selected=None, answer=False):
    if not execute or config.FAKE:
        raise ValueError('Live comparison requires --execute and AML_FAKE=0')
    if repeats < 1:
        raise ValueError('repeats must be positive')
    manifest = verify(snapshot)
    cases = records(snapshot/'cases.jsonl')
    if selected:
        if not set(selected) <= {c['case_id'] for c in cases}:
            raise ValueError('Unknown case selection')
        cases = [c for c in cases if c['case_id'] in selected]
    output.mkdir(parents=True, exist_ok=False)
    write(output/'metadata.json', dict(snapshot=str(snapshot.resolve()), snapshot_hash=digest(snapshot/'manifest.json'),
          versions=run_metadata.versions(), repeats=repeats, execution_kind='replay', answer_scoring=answer,
          cases=[c['case_id'] for c in cases],
          caveat='Frozen retrieval/CE; upstream latency and tokens omitted. No live feedback writes or causal proof.'))
    pairs = []
    for repeat in range(repeats):
        for index, case in enumerate(cases):
            pair = dict(case_id=case['case_id'], repetition=repeat+1,
                        input_fingerprint=case['input_fingerprint'], arms={})
            # Alternate order to limit systematic provider-order effects.
            order = [False, True] if (repeat+index)%2 == 0 else [True, False]
            for staged in order:
                arm = 'staged' if staged else 'legacy'
                try:
                    result = await run_arm(case, staged, answer=answer)
                    result['status'] = 'complete'
                except Exception as exc:
                    result = dict(status='error', error=type(exc).__name__, detail=str(exc)[:300])
                with (output/'results.jsonl').open('a') as stream:
                    stream.write(json.dumps(dict(case_id=case['case_id'], repetition=repeat+1,
                                 input_fingerprint=case['input_fingerprint'], arm=arm, **result),
                                 ensure_ascii=False, allow_nan=False)+'\n')
                pair['arms'][arm] = result
            ok = all(r['status']=='complete' for r in pair['arms'].values())
            pair['comparable'] = ok and admission(pair['arms']['legacy']) == admission(pair['arms']['staged'])
            pair['reason'] = 'same_admission' if pair['comparable'] else 'admission_changed' if ok else 'arm_failed'
            if pair['comparable']:
                pair['score_delta'] = (pair['arms']['staged']['score']-pair['arms']['legacy']['score']) if answer else None
            pairs.append(pair)
            print(json.dumps({k:v for k,v in pair.items() if k!='arms'}), flush=True)
    verify(snapshot)  # Detect accidental modification during execution.
    good = [p for p in pairs if p['comparable']]
    summary = dict(pairs=len(pairs), comparable=len(good), invalid_pairs=len(pairs)-len(good),
                   execution_kind='replay', answer_scoring=answer,
                   mean_score_delta=sum(p['score_delta'] for p in good)/len(good) if answer and good else None,
                   automatic_promotion=False, snapshot_unchanged=True)
    summary['arms'] = {}
    for arm in ('legacy', 'staged'):
        completed = [p['arms'][arm] for p in pairs if p['arms'][arm]['status'] == 'complete']
        summary['arms'][arm] = dict(
            completed=len(completed), failed=len(pairs)-len(completed),
            ranking_seconds=sum(r.get('ranking_seconds', 0) for r in completed),
            ranking_tokens=sum(r.get('usage', {}).get('tokens', 0) for r in completed),
            ranking_calls=sum(r.get('usage', {}).get('calls', 0) for r in completed),
            degraded=sum(bool(r['cascade']['listwise'].get('errors')) for r in completed),
            selected_candidates=sum(len(r['cascade']['listwise'].get('selected_ids', [])) for r in completed),
            answer_score=sum(r['score'] for r in completed) if answer else None)
    write(output/'summary.json', summary)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command', required=True)
    p=sub.add_parser('freeze');p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p=sub.add_parser('verify');p.add_argument('--snapshot',type=Path,required=True)
    p=sub.add_parser('compare');p.add_argument('--snapshot',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--execute',action='store_true');p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--cases',default='');p.add_argument('--answer',action='store_true')
    a=parser.parse_args()
    if a.command=='freeze':result=freeze(a.run,a.output)
    elif a.command=='verify':result=verify(a.snapshot)
    else:result=asyncio.run(compare(a.snapshot,a.output,execute=a.execute,repeats=a.repeats,
                                   selected=a.cases.split(',') if a.cases else None,answer=a.answer))
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
