"""Paired CE topic diagnostic on current adaptive fine/main admission.

Reuse the existing ce_option_topics rule unchanged. Score both queries only
where the rule triggers; retain historical scores elsewhere. This is a CE
transport diagnostic, not an answer or full HTTPS evaluation.
"""
import argparse
import ast
import asyncio
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import budget, config, cross_encoder
from ce_option_topics import queries_for
from evaluation_model_guard import check_runtime, check_versions
from probe_ce_source_units import score_documents
from probe_source_ce import records, select
import audit_https_funnel as audit


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(run):
    check_runtime()
    if config.FAKE or (run / 'exit-code').read_text().strip() != '0':
        raise ValueError('Require real provider and completed successful arm')
    results = records(run / 'completed/data/results/personamem-v2-32k.jsonl')
    by_search = {r['search_id']: r for r in results}
    if len(by_search) != len(results):
        raise ValueError('Duplicate results')
    jobs = []
    for t in records(run / 'server-search-debug.jsonl'):
        r = by_search[t['search_id']]
        check_versions(t['versions'], 'source trace')
        if t['cascade']['cross_encoder']['status'] != 'ok':
            raise ValueError('Historical CE incomplete')
        pool = {c['id']: c for c in t['ranked']}
        coarse = [pool[mid] for mid in t['cascade']['coarse']['selected_ids']]
        queries = queries_for(t)
        docs = [c.get('_rank_text', c['content']) for c in coarse]
        with patch.multiple(config, **t['versions']['settings']):
            check_runtime()
            if config.CE_MODEL != t['cascade']['cross_encoder']['model']:
                raise ValueError('CE model changed')
            archived = select(t, coarse)
            if ([c['id'] for c in archived['fine']] != t['cascade']['fine']['selected_ids'] or
                    [c['id'] for c in archived['main']] != t['cascade']['listwise']['candidate_ids'] or
                    archived['main_bytes'] != t['cascade']['listwise']['input_bytes']):
                raise ValueError('Historical admission mismatch')
            for query in queries.values():
                for doc in docs:
                    if (len(doc.encode()) > config.CE_MAX_DOCUMENT_BYTES or
                            cross_encoder.request_size(query, [doc]) > config.CE_MAX_REQUEST_BYTES):
                        raise ValueError('Document exceeds untruncated CE bounds')
        jobs.append(dict(qa_id=r['qa_id'], t=t, coarse=coarse, docs=docs,
                         queries=queries, changed=queries['control'] != queries['candidate']))
    if len(jobs) != len(results) or len({j['qa_id'] for j in jobs}) != len(jobs):
        raise ValueError('Incomplete or duplicated cohort')
    return jobs


def replay(job, scores):
    coarse = copy.deepcopy(job['coarse'])
    if len(scores) != len(coarse) or not all(type(v) in (float, int) and math.isfinite(v) for v in scores):
        raise ValueError('Incomplete/nonfinite CE scores')
    for c, value in zip(coarse, scores):
        c['_ce_score'] = value
        c['_final'] = value
    with patch.multiple(config, **job['t']['versions']['settings']):
        return select(job['t'], coarse)


async def main(a):
    jobs = prepare(a.run)
    plan_path = a.output.with_suffix('.plan.json')
    live_path = a.output.with_suffix('.live.jsonl')
    if any(p.exists() for p in (a.output, plan_path, live_path)):
        raise FileExistsError(a.output)
    plan = dict(run=str(a.run), model=config.LLM_MODEL, ce_model=config.CE_MODEL,
                trace_sha256=sha(a.run / 'server-search-debug.jsonl'),
                rule_sha256=sha(Path(__file__).with_name('ce_option_topics.py')),
                script_sha256=sha(Path(__file__)), limitation=__doc__,
                jobs=[dict(qa_id=j['qa_id'], search_id=j['t']['search_id'], changed=j['changed'],
                           queries=j['queries'], candidate_ids=[c['id'] for c in j['coarse']],
                           document_hashes=[hashlib.sha256(d.encode()).hexdigest() for d in j['docs']])
                      for j in jobs])
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(questions=len(jobs), changed=[j['qa_id'] for j in jobs if j['changed']])), flush=True)
    live = []
    scored = []
    for job in jobs:
        original = [c['_ce_score'] for c in job['coarse']]
        scores = dict(archived=original)
        for name, query in job['queries'].items():
            if not job['changed']:
                scores[name] = original
                continue
            with patch.multiple(config, **job['t']['versions']['settings']):
                check_runtime()
                with budget.scope(seconds=240, calls=100, tokens=1000000) as usage:
                    values, batches = await score_documents(query, job['docs'])
            replay(job, values)  # Validate before recording a successful arm.
            row = dict(qa_id=job['qa_id'], arm=name, query=query, scores=values,
                       calls=usage.calls, tokens=usage.tokens, batches=batches)
            with live_path.open('a') as f:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
            live.append(row)
            scores[name] = values
            print(json.dumps({k: row[k] for k in ('qa_id', 'arm', 'calls', 'tokens')}), flush=True)
        scored.append((job, {name: replay(job, values) for name, values in scores.items()}))
    # References are consulted only after every request and selection is fixed.
    csv.field_size_limit(32 * 1024 * 1024)
    with a.gold_csv.open() as f:
        gold = list(csv.DictReader(f))
    raw = {}
    for event in records(a.run / 'server-memory-debug.jsonl'):
        raw.setdefault(event['user_id'], {}).update({audit.identity(s): s for s in event['source_messages']})
    rows = []
    for job, variants in scored:
        meta = gold[int(job['qa_id'])]
        if audit.norm(ast.literal_eval(meta['user_query'])['content']) != audit.norm(job['t']['query']):
            raise ValueError('Gold question identity mismatch')
        snippet = meta['related_conversation_snippet']
        try:
            snippet = json.loads(snippet)
        except ValueError:
            snippet = ast.literal_eval(snippet)
        result = {}
        for name, variant in variants.items():
            entry = dict(main_bytes=variant['main_bytes'])
            for stage in ('fine', 'main'):
                cards = [dict(s, text=s.get('content', ''), id=f'{i}:{k}')
                         for i, c in enumerate(variant[stage]) for k, s in enumerate(c['_packet_item'].get('sources', []))]
                entry[stage] = dict(ids=[c['id'] for c in variant[stage]], **{
                    role: [v['full'] for v in audit.turn_coverage(
                        [audit.norm(s['content']) for s in snippet if s['role'] == role],
                        cards, raw[job['t']['user_id']], role=role)] for role in ('user', 'assistant')})
            result[name] = entry
        rows.append(dict(qa_id=job['qa_id'], changed=job['changed'], variants=result))
    summary = dict(questions=len(rows), changed=sum(r['changed'] for r in rows),
                   calls=sum(r['calls'] for r in live), tokens=sum(r['tokens'] for r in live), variants={})
    for name in ('archived', 'control', 'candidate'):
        summary['variants'][name] = {stage: dict(
            reference_questions=sum(any(r['variants'][name][stage]['user']) for r in rows),
            user_turns=sum(sum(r['variants'][name][stage]['user']) for r in rows),
            gained=[r['qa_id'] for r in rows if any(r['variants'][name][stage]['user']) and not any(r['variants']['control'][stage]['user'])],
            lost=[r['qa_id'] for r in rows if any(r['variants']['control'][stage]['user']) and not any(r['variants'][name][stage]['user'])]) for stage in ('fine', 'main')}
    a.output.write_text(json.dumps(dict(summary=summary, rows=rows, plan=str(plan_path), limitation=__doc__), ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gold-csv', type=Path, default=Path(__file__).resolve().parents[1] / 'data/raw/personamem-v2.csv')
    asyncio.run(main(p.parse_args()))
