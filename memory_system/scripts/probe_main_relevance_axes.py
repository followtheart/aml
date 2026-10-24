"""Separate question relevance from option relevance in the primary selector.

Frozen whole-candidate membership, sources and options; no reference annotations
enter request construction. Only two valid irrelevant judgments authorize an
irrelevant candidate. Labels are retrieval judgments, never personal facts.
This diagnostic does not execute recovery, packing or the answer reader.
"""
import argparse
import asyncio
from collections import Counter
import copy
import json
import logging
from pathlib import Path
import time

from probe_main_source_layout import prepare as historical_jobs, decode as original_decode, records, sha, fingerprint
from app import budget, config, llm
from evaluation_model_guard import check_runtime

LABELS = ('useful', 'context', 'uncertain', 'irrelevant')
OLD = ('For each requested candidate key return priority '
       '(integer 0..100, lower is better; ties allowed) and relevance '
       '("useful", "context", "uncertain", or "irrelevant"). ')
NEW = ('For each key return priority (integer 0..100, lower is better; ties allowed), '
       'question_relevance and option_relevance (useful, context, uncertain, irrelevant). ')
GUIDANCE = (' Judge question_relevance against the original question on its own, and option_relevance '
    'against all supplied answer alternatives. Make both judgments for every candidate. '
    'Use the same useful/context/uncertain/irrelevant definitions on each axis. '
    'A source may be relevant topical context for the question while failing to establish any '
    'personal premise in the options; record that distinction rather than discarding the source. '
    'Neither axis certifies a personal fact. Do not infer a preference, identity or experience '
    'from quoted, fictional, third-party or assistant text. Generic shared words alone are not '
    'relevance. Mark an axis irrelevant only when no useful role exists for that axis. '
    'Priority remains a comparative ordering judgment, not deletion authority. '
    'Groups may include a candidate unless BOTH relevance axes are irrelevant.')


def proposal(request):
    after = copy.deepcopy(request)
    if after['prompt'].count(OLD) != 1:
        raise ValueError('Unexpected primary keyed request')
    after['prompt'] = after['prompt'].replace(OLD, NEW, 1)
    props = after['schema']['properties']['candidates']['properties']
    for value in props.values():
        if set(value['properties']) != {'priority', 'relevance'}:
            raise ValueError('Unexpected candidate schema')
        label = value['properties'].pop('relevance')
        if label['enum'] != list(LABELS):
            raise ValueError('Requires existing context label')
        value['properties'].update(question_relevance=copy.deepcopy(label), option_relevance=copy.deepcopy(label))
        value['required'] = ['priority', 'question_relevance', 'option_relevance']
    after['system'] += GUIDANCE
    after['max_tokens'] = max(256, 80*len(props))
    after['reservation'] = (len(after['prompt'].encode()) + len(json.dumps(after['schema']).encode())
        + len(after['system'].encode()) + 1024 + after['max_tokens'])
    # Invert all intentional changes and compare the entire original request.
    restored = copy.deepcopy(after)
    restored['prompt'] = restored['prompt'].replace(NEW, OLD, 1)
    restored['system'] = restored['system'][:-len(GUIDANCE)]
    restored['max_tokens'] = request['max_tokens']; restored['reservation'] = request['reservation']
    for value in restored['schema']['properties']['candidates']['properties'].values():
        label = value['properties'].pop('question_relevance')
        if value['properties'].pop('option_relevance') != label:
            raise ValueError('Label definitions differ')
        value['properties']['relevance'] = label
        value['required'] = ['priority', 'relevance']
    if restored != request or len(after['prompt'].encode()) > len(request['prompt'].encode()):
        raise ValueError('Unexpected content/admission change')
    return after


def decode(payload, job, arm):
    if arm == 'control':
        return original_decode(payload,job)
    clean = copy.deepcopy(payload)
    if isinstance(clean,dict) and isinstance(clean.get('candidates'),dict):
        for key,value in clean['candidates'].items():
            if (isinstance(value,dict) and set(value)=={'priority','question_relevance','option_relevance'}
                    and value['question_relevance'] in LABELS and value['option_relevance'] in LABELS):
                clean['candidates'][key] = dict(priority=value['priority'], relevance=min(
                    (value['question_relevance'],value['option_relevance']),key=LABELS.index))
            else:
                # A missing axis or off-contract old relevance field must not
                # become a valid candidate record in the original validator.
                clean['candidates'][key] = dict(invalid_axes=value)
    return original_decode(clean,job)


def boundary_checks():
    job=dict(candidate_ids=['owned'],tie_order=[0])
    checked=0
    for q,o,kept in [('irrelevant','irrelevant',False),('context','irrelevant',True),
                     ('irrelevant','context',True),('uncertain','irrelevant',True)]:
        d=decode(dict(candidates={'c0':dict(priority=12,question_relevance=q,option_relevance=o)},groups=[]),job,'treatment')
        assert d['valid'] and bool(d['selected_ids'])==kept
        checked+=1
    for bad in [dict(priority=12,relevance='useful'),dict(priority=12,question_relevance='context'),
                dict(priority=12,question_relevance='useful',option_relevance='invented'),
                dict(priority=True,question_relevance='useful',option_relevance='irrelevant')]:
        d=decode(dict(candidates={'c0':bad},groups=[]),job,'treatment')
        assert not d['valid'] and d['selected_ids'] is None
        checked+=1
    return checked


def prepare(run):
    jobs=historical_jobs(run)
    traces={t['search_id']:t for t in records(run/'server-search-debug.jsonl')}
    for job in jobs:
        before=job['requests']['control'];after=proposal(before)
        headroom=traces[job['search_id']]['cascade']['admission_budget']['token_headroom']
        if headroom is not None and after['reservation']>headroom:
            raise ValueError('Treatment exceeds original main token headroom')
        job.update(changed=True,requests=dict(control=before,treatment=after),
            original_token_headroom=headroom,reservation_delta=after['reservation']-before['reservation'])
    return jobs


async def main(args):
    check_runtime()
    if config.FAKE:raise ValueError('Require real fixed model')
    checks=boundary_checks();jobs=prepare(args.run)
    plan=dict(source_run=str(args.run.resolve()),source_trace_sha256=sha(args.run/'server-search-debug.jsonl'),
        script_sha256=sha(Path(__file__)),loader_sha256=sha(Path(__file__).with_name('probe_main_source_layout.py')),
        model=config.LLM_MODEL,protocol=__doc__,boundary_checks=checks,questions=len(jobs),jobs_sha256=fingerprint(jobs),
        live=args.live,expected_calls=2*len(jobs) if args.live else 0,
        reservation_delta_range=[min(j['reservation_delta'] for j in jobs),max(j['reservation_delta'] for j in jobs)],
        schedule=[dict(qa_id=j['qa_id'],order=['control','treatment'] if i%2==0 else ['treatment','control']) for i,j in enumerate(jobs)],
        advance_criterion='Require valid candidate-stage net reference improvement, plausible benefit beyond already delivered historical final sources, and semantic review of all added selections and losses. Report invalid group contracts separately; do not count invalid-output retention as a recovery. An HTTPS ablation must test downstream effects under original budgets.',
        historical_limit='Historical primary membership, input size and label-selected sets reconstruct; original request hashes/priorities/full primary payloads were not saved.')
    if args.live:
        before=json.loads((args.preflight/'plan.json').read_text())
        for k in ('source_run','source_trace_sha256','script_sha256','loader_sha256','model','questions','jobs_sha256','schedule'):
            if before[k]!=plan[k]:raise ValueError('Preflight mismatch: '+k)
        plan['preflight_sha256']=sha(args.preflight/'plan.json')
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
    (args.output/'requests.jsonl').write_text(''.join(json.dumps(j,ensure_ascii=False)+'\n' for j in jobs))
    print(json.dumps({k:plan[k] for k in ('questions','expected_calls','boundary_checks','reservation_delta_range')}),flush=True)
    rows=[]
    if args.live:
        handler=logging.FileHandler(args.output/'provider.log',mode='x')
        logger=logging.getLogger('aml.metrics');logger.setLevel(logging.INFO);logger.addHandler(handler)
        semaphore=asyncio.Semaphore(2)
        async def pair(job,schedule):
            async with semaphore:
                for arm in schedule['order']:
                    request=copy.deepcopy(job['requests'][arm]);bound=request.pop('reservation')
                    row=dict(qa_id=job['qa_id'],arm=arm,request_sha256=fingerprint(request),reservation=bound)
                    started=time.monotonic()
                    with budget.scope(seconds=45,calls=1,tokens=bound) as usage:
                        try:
                            async with asyncio.timeout(45):
                                payload=await llm.complete_json(**request,attempts=1,timeout=45,
                                    stage=f'diagnostic.main_relevance_axes.{job["qa_id"]}.{arm}')
                            row['payload']=payload;row['decoded']=decode(payload,job,arm)
                        except Exception as exc:row['error']=type(exc).__name__
                    row.update(calls=usage.calls,tokens=usage.tokens,seconds=time.monotonic()-started)
                    rows.append(row)
                    with (args.output/'live.jsonl').open('a') as stream:stream.write(json.dumps(row,ensure_ascii=False)+'\n')
                    print(json.dumps(dict(completed=len(rows),qa_id=row['qa_id'],arm=arm,error=row.get('error'))),flush=True)
        try:await asyncio.gather(*(pair(j,s) for j,s in zip(jobs,plan['schedule'],strict=True)))
        finally:logger.removeHandler(handler);handler.close()
    summary=dict(rows=len(rows),calls=sum(r['calls'] for r in rows),tokens=sum(r['tokens'] for r in rows),
        errors=dict(Counter(r['error'] for r in rows if 'error' in r)),
        invalid_responses=sum(not r['decoded']['valid'] for r in rows if 'decoded' in r),
        plan_sha256=sha(args.output/'plan.json'),requests_sha256=sha(args.output/'requests.jsonl'),
        responses_sha256=sha(args.output/'live.jsonl') if rows else None)
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    complete=len(rows)==plan['expected_calls'];(args.output/'exit-code').write_text('0\n' if complete else '1\n')
    print(json.dumps(summary),flush=True)
    if not complete:raise RuntimeError('Incomplete diagnostic')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--preflight',type=Path);parser.add_argument('--live',action='store_true')
    args=parser.parse_args()
    if args.live and args.preflight is None:parser.error('--live requires --preflight')
    asyncio.run(main(args))
