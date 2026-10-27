"""Paired main-ranking diagnostic: display shared sources before candidate cards.

All sources, candidate bodies, IDs, ownership, options, instructions, schema and
byte budgets are unchanged. Only the two JSON fields change display order.
Historical main membership is frozen; independent single-attempt calls do not
measure downstream recovery, packet transport or full HTTPS answer accuracy.
No reference labels enter request construction or scheduling.
"""
import argparse
import asyncio
from collections import Counter
import copy
import hashlib
import json
import logging
from pathlib import Path
import sys
import time
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import budget, cascade_rerank, config, fine_evidence, listwise_contract, listwise_sources, llm, schemas
from evaluation_model_guard import check_runtime, check_versions


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def source_first(original):
    prefix, raw = original.split('\n', 1)
    data = json.loads(raw)
    if list(data) != ['candidates', 'shared_sources']:
        raise ValueError('Unexpected shared-source layout')
    reordered = dict(shared_sources=data['shared_sources'], candidates=data['candidates'])
    result = prefix + '\n' + json.dumps(reordered, ensure_ascii=False, separators=(',', ':'))
    if json.loads(result.split('\n', 1)[1]) != data or len(result.encode()) != len(original.encode()):
        raise ValueError('Source/candidate content or byte length changed')
    return result


def prepare(run):
    if (run / 'exit-code').read_text().strip() != '0':
        raise ValueError('Require a completed successful arm')
    parent = run.parent
    plan = json.loads((parent / 'plan.json').read_text())
    project = Path(__file__).resolve().parents[1]
    for name, expected in plan['source_files'].items():
        if name.startswith(('memory_system/app/', 'prompts/')):
            if sha(project.parent / name) != expected or sha(parent / 'source' / name) != expected:
                raise ValueError('Source or prompt differs from completed HTTPS')
    results = records(run / 'completed/data/results/personamem-v2-32k.jsonl')
    traces = {t['search_id']: t for t in records(run / 'server-search-debug.jsonl')}
    if len(results) != plan['questions'] or len(traces) != len(results):
        raise ValueError('Incomplete cohort')
    jobs = []
    for result in sorted(results, key=lambda r: int(r['qa_id'])):
        t = traces[result['search_id']]
        check_versions(t['versions'], 'main layout source')
        if result.get('error_stage') or result.get('fake') or result['transport'] != 'https':
            raise ValueError('Invalid source result')
        rows = {c['id']: c for c in t['ranked']}
        main = t['cascade']['listwise']
        candidates = [copy.deepcopy(rows[mid]) for mid in main['candidate_ids']]
        upstream = copy.deepcopy(t['plan'])
        upstream['_coverage_requirements'] = [{k: r[k] for k in ('id', 'text', 'origin') if k in r}
            for r in t['coverage_manifest']['coverage']['requirements']]
        upstream['_fusion_edges'] = t['fusion_edges']
        upstream['_fusion_triples'] = t['fusion_triples']
        upstream['_fine_path_groups'] = t['cascade']['fine']['selection'].get('adaptive', {}).get('path_groups', [])
        req = schemas.SearchRequest(user_id=t['user_id'], query=t['query'], options=t['options'])
        with patch.multiple(config, **t['versions']['settings']), patch('socket.socket.connect', side_effect=AssertionError('Offline preparation')):
            if not config.LISTWISE_KEYED_OUTPUT or not config.LISTWISE_SHARED_SOURCES:
                raise ValueError('Requires keyed shared-source main ranking')
            before = cascade_rerank.listwise_request(req, upstream, candidates)
            original = listwise_sources.shared_candidates
            with patch.object(listwise_sources, 'shared_candidates', side_effect=lambda rows: source_first(original(rows))):
                after = cascade_rerank.listwise_request(req, upstream, candidates)
        if len(before['prompt'].encode()) != main['input_bytes']:
            raise ValueError('Historical main input size did not reconstruct')
        if any(before[k] != after[k] for k in before if k != 'prompt'):
            raise ValueError('Request fields other than prompt changed')
        changed = before['prompt'] != after['prompt']
        if changed:
            block = original(candidates).replace('Return candidate indices, never source indices.',
                'Return candidate output keys, never source indices.', 1)
            if before['prompt'].count(block) != 1 or after['prompt'] != before['prompt'].replace(block, source_first(block), 1):
                raise ValueError('Unexpected prompt mutation')
        if len(before['prompt'].encode()) != len(after['prompt'].encode()):
            raise ValueError('Admission/budget differs')
        # Logs retain labels, not the full original primary payload. Validate
        # their selected set; do not claim an exact historical response replay.
        labels = main['candidate_relevance']
        if (main['candidate_contract_status'] != 'valid' or set(labels) != set(main['candidate_ids'])
                or {mid for mid, label in labels.items() if label != 'irrelevant'} != set(main['primary_selected_ids'])):
            raise ValueError('Historical primary selection does not match recorded labels')
        tie_order = sorted(range(len(candidates)), key=lambda i: fine_evidence.key(candidates[i]), reverse=True)
        jobs.append(dict(qa_id=result['qa_id'], search_id=t['search_id'], changed=changed,
            candidate_ids=main['candidate_ids'], tie_order=tie_order,
            historical_labels=labels, requests=dict(control=before, treatment=after)))
    if len({j['qa_id'] for j in jobs}) != plan['questions']:
        raise ValueError('Question identity mismatch')
    return jobs


def decode(payload, job):
    state = listwise_contract.State(len(job['candidate_ids']), keyed=True, tie_order=job['tie_order'])
    state.accept(payload)
    return dict(errors=state.errors, candidate_records=state.accepted_candidates,
        valid=not state.errors, selected_ids=[job['candidate_ids'][i] for i in state.ranking
            if i not in state.irrelevant] if state.ranking is not None else None,
        groups=state.groups)


async def main(args):
    check_runtime()
    if config.FAKE:
        raise ValueError('Real fixed model required')
    jobs = prepare(args.run)
    plan = dict(source_run=str(args.run.resolve()), source_trace_sha256=sha(args.run / 'server-search-debug.jsonl'),
        script_sha256=sha(Path(__file__)), model=config.LLM_MODEL, protocol=__doc__,
        questions=len(jobs), changed_pairs=sum(j['changed'] for j in jobs),
        jobs_sha256=fingerprint(jobs), live=args.live, expected_calls=2*len(jobs) if args.live else 0,
        schedule=[dict(qa_id=j['qa_id'], order=['control', 'treatment'] if i % 2 == 0 else ['treatment', 'control']) for i,j in enumerate(jobs)],
        advance_criterion='Audit every response and semantically inspect added reference and non-reference selections. Require primary reference improvement without reference losses or invalid-output retention before a production HTTPS ablation. Fixed historical recovery is not a fresh end-to-end result.',
        historical_limit='Main request size, source implementation and labels are reconstructable; original primary request hashes/priorities/raw payloads were not saved. No claim of exact historical provider-response replay.')
    if args.live:
        previous = json.loads((args.preflight / 'plan.json').read_text())
        for k in ('source_run','source_trace_sha256','script_sha256','model','questions','changed_pairs','jobs_sha256','schedule'):
            if previous[k] != plan[k]:
                raise ValueError('Preflight mismatch: ' + k)
        plan['preflight_sha256'] = sha(args.preflight / 'plan.json')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'plan.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2)+'\n')
    (args.output / 'requests.jsonl').write_text(''.join(json.dumps(j, ensure_ascii=False)+'\n' for j in jobs))
    print(json.dumps({k:plan[k] for k in ('questions','changed_pairs','expected_calls')}), flush=True)
    rows = []
    if args.live:
        handler = logging.FileHandler(args.output / 'provider.log', mode='x')
        logger = logging.getLogger('aml.metrics'); logger.setLevel(logging.INFO); logger.addHandler(handler)
        semaphore = asyncio.Semaphore(2)
        async def pair(job, schedule):
            async with semaphore:
                for arm in schedule['order']:
                    request = copy.deepcopy(job['requests'][arm]); bound = request.pop('reservation')
                    row = dict(qa_id=job['qa_id'], arm=arm, changed=job['changed'], request_sha256=fingerprint(request), reservation=bound)
                    started = time.monotonic()
                    with budget.scope(seconds=45, calls=1, tokens=bound) as usage:
                        try:
                            async with asyncio.timeout(45):
                                payload = await llm.complete_json(**request, attempts=1, timeout=45,
                                    stage=f'diagnostic.main_source_layout.{job["qa_id"]}.{arm}')
                            row['payload'] = payload
                            row['decoded'] = decode(payload, job)
                        except Exception as exc:
                            row['error'] = type(exc).__name__
                    row.update(calls=usage.calls, tokens=usage.tokens, seconds=time.monotonic()-started)
                    rows.append(row)
                    with (args.output / 'live.jsonl').open('a') as stream:
                        stream.write(json.dumps(row, ensure_ascii=False)+'\n')
                    print(json.dumps(dict(completed=len(rows), qa_id=row['qa_id'], arm=arm, error=row.get('error'))), flush=True)
        try:
            await asyncio.gather(*(pair(j,s) for j,s in zip(jobs,plan['schedule'],strict=True)))
        finally:
            logger.removeHandler(handler); handler.close()
    summary = dict(rows=len(rows), calls=sum(r['calls'] for r in rows), tokens=sum(r['tokens'] for r in rows),
        errors=dict(Counter(r['error'] for r in rows if 'error' in r)),
        invalid_responses=sum(not r['decoded']['valid'] for r in rows if 'decoded' in r),
        plan_sha256=sha(args.output/'plan.json'), requests_sha256=sha(args.output/'requests.jsonl'),
        responses_sha256=sha(args.output/'live.jsonl') if rows else None)
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    complete = len(rows) == plan['expected_calls']
    (args.output/'exit-code').write_text('0\n' if complete else '1\n')
    print(json.dumps(summary),flush=True)
    if not complete:
        raise RuntimeError('Incomplete diagnostic cohort')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--preflight',type=Path)
    parser.add_argument('--live',action='store_true')
    args=parser.parse_args()
    if args.live and args.preflight is None:
        parser.error('--live requires --preflight')
    asyncio.run(main(args))
