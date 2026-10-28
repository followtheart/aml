"""Paired CE scoring views with durable batches and complete-cohort failures.

Only inferred primary assertions are replaced in the CE scoring document.
Query, candidates, displayed sources, fine/main rules and budgets stay fixed.
Persist every batch, attempt every planned arm once, and report missing score
vectors explicitly. A failed arm is never replaced with archived or zero scores.
No selected retry, automatic full rerun, HTTPS or answer-accuracy claim.
"""
import argparse
import asyncio
import copy
import csv
import hashlib
import json
import logging
from pathlib import Path
import sqlite3
import sys
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import budget, config, cross_encoder, rerank_query, run_metadata
import probe_full_option_ce as common
from probe_ce_relevance_guidance import bounded_readers, sha, records
from ce_inferred_primary_view import view
from evaluate_recall_context import reference_turns
from evaluation_model_guard import check_runtime
import audit_https_funnel as funnel

POLICY = 'inferred_primary_ce_ablation_all_original_sources'


def csv_batches(path, batch_bytes=4*1024*1024):
    """CSV parser receives physical records from binary read batches.

    Preserve line endings and quoted multiline fields across batch boundaries.
    File I/O never uses readline or file iteration.
    """
    if type(batch_bytes) is not int or batch_bytes <= 0:
        raise ValueError('Batch size must be a positive integer')
    def physical_records():
        pending = b''
        with path.open('rb') as stream:
            while block := stream.read(batch_bytes):
                complete = (pending + block).split(b'\n')
                pending = complete.pop()
                yield from ((record+b'\n').decode('utf-8') for record in complete)
            if pending:
                yield pending.decode('utf-8')
    yield from csv.DictReader(physical_records())


def source_batches(db):
    cursor = db.execute('SELECT * FROM source_messages')
    while batch := cursor.fetchmany(256):
        yield from batch


def append(path, value):
    with path.open('a') as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + '\n')


def queries_for(trace):
    req = SimpleNamespace(query=trace['query'], options=trace['options'])
    plan = dict(copy.deepcopy(trace['plan']), _query_specs=copy.deepcopy(trace['query_specs']))
    with patch.object(config, 'CE_FULL_OPTION_TEXT', False):
        query = rerank_query.build(req, plan)
    if query != trace['cascade']['cross_encoder']['query']:
        raise ValueError('Historical CE query mismatch')
    return dict(control=query, candidate=query)


def batches(query, docs):
    result, pending = [], []
    for index, doc in enumerate(docs):
        if (len(doc.encode()) > config.CE_MAX_DOCUMENT_BYTES
                or cross_encoder.request_size(query, [doc]) > config.CE_MAX_REQUEST_BYTES):
            raise ValueError('Document exceeds intact input bound')
        if pending and (len(pending) >= config.CE_BATCH_SIZE or
                cross_encoder.request_size(query, [docs[i] for i in pending] + [doc]) > config.CE_MAX_REQUEST_BYTES):
            result.append(pending); pending = []
        pending.append(index)
    if pending:
        result.append(pending)
    return result


def prepare(run):
    with bounded_readers():
        jobs, manifest, conversations = common.prepare(run, query_builder=queries_for)
    for job in jobs:
        values = [view(c) for c in job['coarse']]
        job['documents'] = dict(control=job['docs'], candidate=[v[0] for v in values])
        job['views'] = [v[1] for v in values]
        job['changed'] = job['documents']['candidate'] != job['docs']
        with patch.multiple(config, **job['t']['versions']['settings']):
            job['batches'] = {arm: batches(job['queries'][arm], docs)
                              for arm, docs in job['documents'].items()}
    return jobs, manifest, conversations


def preflight(jobs):
    return dict(questions=len(jobs), reconstructed_fine_and_main=len(jobs),
        changed_questions=sum(j['changed'] for j in jobs),
        changed_documents=sum(v['changed'] for j in jobs for v in j['views']),
        unchanged_reuse_questions=sum(not j['changed'] for j in jobs),
        planned_calls=sum(len(b) for j in jobs if j['changed'] for b in j['batches'].values()))


def planned_job(job):
    return dict(qa_id=job['qa_id'], conversation_id=job['conversation_id'],
        search_id=job['t']['search_id'], queries=job['queries'], changed=job['changed'],
        candidate_ids=[c['id'] for c in job['coarse']], views=job['views'],
        document_hashes={a: [hashlib.sha256(d.encode()).hexdigest() for d in docs]
                         for a, docs in job['documents'].items()}, batches=job['batches'])


async def score_arm(job, arm, output):
    docs = job['documents'][arm]; query = job['queries'][arm]
    scores = [None] * len(docs); batch_rows = []
    with patch.multiple(config, **dict(job['t']['versions']['settings'], RATE_LIMIT_RETRIES=0)):
        check_runtime()
        with budget.scope(seconds=240, calls=100, tokens=1000000) as usage:
            for number, indexes in enumerate(job['batches'][arm]):
                before_calls, before_tokens = usage.calls, usage.tokens
                row = dict(qa_id=job['qa_id'], conversation_id=job['conversation_id'],
                    arm=arm, batch=number, indexes=indexes)
                try:
                    values = await cross_encoder.rerank(query, [docs[i] for i in indexes],
                        stage='probe.ce_inferred_primary')
                    # Validate finite scores before recording the successful batch.
                    import math
                    if len(values) != len(indexes) or any(type(v) not in (float, int) or not math.isfinite(v) for v in values):
                        raise ValueError('Invalid batch score vector')
                    for i, value in zip(indexes, values):
                        scores[i] = value
                    row.update(status='complete', scores=values)
                except Exception as exc:
                    row.update(status='failed', error_type=type(exc).__name__, scores=None)
                row.update(calls=usage.calls-before_calls, tokens=usage.tokens-before_tokens)
                append(output/'batches.jsonl', row)
                batch_rows.append(row)
            event = dict(qa_id=job['qa_id'], conversation_id=job['conversation_id'], arm=arm,
                status='complete' if all(r['status']=='complete' for r in batch_rows) else 'failed',
                query=query, scores=scores, calls=usage.calls, tokens=usage.tokens,
                batches=len(batch_rows), failed_batches=[r['batch'] for r in batch_rows if r['status']=='failed'])
    append(output/'responses.jsonl', event)
    print(json.dumps({k:event[k] for k in ('qa_id','arm','status','calls','tokens')}), flush=True)
    return event


def selections(job, events):
    original = [c['_ce_score'] for c in job['coarse']]
    variants = {'archived': common.base.replay(job, original)}
    for arm in ('control', 'candidate'):
        event = events.get(arm)
        values = original if not job['changed'] else event['scores'] if event['status']=='complete' else None
        variants[arm] = common.base.replay(job, values) if values is not None else None
    delta = None
    if variants['control'] is not None and variants['candidate'] is not None:
        delta = max(0, events.get('candidate', {}).get('tokens', 0)-events.get('control', {}).get('tokens', 0))
        adjusted = copy.deepcopy(job)
        headroom = adjusted['t']['cascade']['admission_budget']['token_headroom']
        if headroom is not None:
            adjusted['t']['cascade']['admission_budget']['token_headroom'] = max(0, headroom-delta)
        values = original if not job['changed'] else events['candidate']['scores']
        variants['candidate_cost_adjusted'] = common.base.replay(adjusted, values)
    else:
        variants['candidate_cost_adjusted'] = None
    return variants, delta


def coverage_rows(run, jobs, fixed, conversations, gold_csv):
    # Read references only after all selections are fixed. Source text comes
    # independently from the frozen database rather than the scoring document.
    csv.field_size_limit(32*1024*1024)
    wanted = {int(j['qa_id']) for j in jobs}; gold = {}
    for i, row in enumerate(csv_batches(gold_csv)):
        if i in wanted:
            gold[str(i)] = row
        if i >= max(wanted):
            break
    raw = {}
    with sqlite3.connect((run/'memory.db').resolve().as_uri()+'?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        for row in source_batches(db):
            row = dict(row)
            raw.setdefault(row['user_id'], {})[funnel.identity(row)] = row
    convs = {c['conversation_id']:c for c in conversations}; rows = []
    for job, (variants, delta) in zip(jobs, fixed):
        conv = convs[job['conversation_id']]
        question = next(q for q in conv['qa'] if q['id']==job['qa_id'])
        turns = reference_turns(question, conv['source_history'], gold[job['qa_id']])
        rows.append(dict(qa_id=job['qa_id'], conversation_id=job['conversation_id'],
            changed=job['changed'], additional_token_cost=delta,
            variants={a:common.coverage(job, v, turns, raw[job['t']['user_id']]) if v is not None else None
                      for a,v in variants.items()}))
    return rows


def summarize(jobs, rows, events):
    summary = dict(**preflight(jobs), calls=sum(e['calls'] for e in events),
        reported_tokens=sum(e['tokens'] for e in events),
        failed_arms=[dict(qa_id=e['qa_id'],arm=e['arm'],failed_batches=e['failed_batches'])
                     for e in events if e['status']!='complete'], variants={})
    for arm in ('archived','control','candidate','candidate_cost_adjusted'):
        valid = [r for r in rows if r['variants'][arm] is not None]
        summary['variants'][arm] = dict(evaluated_questions=len(valid),
            unavailable_questions=[r['qa_id'] for r in rows if r['variants'][arm] is None],
            stages={stage:dict(reference_questions=sum(any(r['variants'][arm][stage]['user']) for r in valid),
                reference_turns=sum(sum(r['variants'][arm][stage]['user']) for r in valid)) for stage in ('fine','main')})
    summary['limitation'] = 'Counts cover explicitly evaluated questions only; any unavailable arm fails the adoption gate. Positive CE-token differences adjust main headroom, not live deadlines. Failed usage may be unknown.'
    return summary


async def run(args):
    jobs, manifest, conversations = prepare(args.run)
    check = preflight(jobs)
    if args.preflight:
        print(json.dumps(check)); return
    args.output.mkdir(parents=True, exist_ok=False)
    for name in ('responses.jsonl', 'batches.jsonl'):
        (args.output/name).touch(exist_ok=False)
    scripts = [Path(__file__),Path(__file__).with_name('ce_inferred_primary_view.py'),
        Path(__file__).with_name('probe_ce_relevance_guidance.py'),Path(common.__file__),
        Path(common.base.__file__),Path(__file__).with_name('probe_source_ce.py'),
        Path(__file__).with_name('probe_coarse_fine_core.py'),
        Path(__file__).with_name('audit_ce_inferred_primary.py')]
    plan = dict(policy=POLICY, run=str(args.run.resolve()), prepared=manifest['data'],
        data_sha256=manifest['data_sha256'], versions=run_metadata.versions(), preflight=check,
        script_hashes={str(p.resolve()):sha(p) for p in scripts},
        trace_sha256=sha(args.run/'server-search-debug.jsonl'),
        result_sha256=sha(args.run/'completed/data/results/personamem-v2-32k.jsonl'),
        stage_sha256=sha(args.run/'funnel/stage-transport.json'), database_sha256=sha(args.run/'memory.db'),
        diagnostic_dispatch=dict(rate_limit_retries=0, ordinary_attempts=1, arm_seconds=240,
                                 arm_max_calls=100, arm_max_tokens=1000000),
        jobs=[planned_job(j) for j in jobs], failure_policy=__doc__)
    (args.output/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(check), flush=True)
    events, fixed = [], []
    for job in jobs:
        pair = {}
        if job['changed']:
            for arm in ('control','candidate'):
                pair[arm] = await score_arm(job, arm, args.output)
                events.append(pair[arm])
        fixed.append(selections(job, pair))
    rows = coverage_rows(args.run, jobs, fixed, conversations, args.gold_csv)
    report = dict(summary=summarize(jobs, rows, events), rows=rows)
    (args.output/'result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    # Exit0 means every planned question was processed; failed_arms is a
    # separate required gate and must never be hidden by process success.
    (args.output/'exit-code').write_text('0\n')
    print(json.dumps(report['summary']), flush=True)


if __name__ == '__main__':
    logger = logging.getLogger('aml.metrics'); logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler()); logger.propagate=False
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--gold-csv',type=Path,default=Path(__file__).resolve().parents[1]/'data/raw/personamem-v2.csv')
    asyncio.run(run(parser.parse_args()))
