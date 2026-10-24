"""Measure missing CE scores, then replay fine/main for frozen source routes.

Original query/text pairs reuse their recorded qwen3-rerank scores. All newly
admitted or changed candidate texts are scored by the same real CE, once per
question/identity/text and shared between variants. Historical main admission
headroom is held fixed: this is a diagnostic, not a live Search or reader run.
All selections finish before reference scoring; no personal facts are inferred.
"""
import argparse
import ast
import asyncio
import copy
import csv
import hashlib
import json
import logging
from pathlib import Path
import sqlite3
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import budget, cascade_rerank, config, cross_encoder, schemas
from evaluation_model_guard import check_runtime, check_versions
from probe_source_ce import select
from probe_semantic_upstream import digest, records
import audit_https_funnel as audit


def identity(c):
    return c['id'] + ':' + hashlib.sha256(c['_rank_text'].encode()).hexdigest()


async def main(args):
    check_runtime()
    if config.FAKE or config.CE_MODEL != 'qwen3-rerank':
        raise ValueError('Require real unchanged CE')
    root, upstream = args.run.resolve(), args.upstream.resolve()
    if (upstream/'exit-code').read_text().strip() != '0':
        raise ValueError('Require complete upstream diagnostic')
    previous = json.loads((upstream/'summary.json').read_text())
    if digest(upstream/'selections.jsonl') != previous['selections_sha256']:
        raise ValueError('Upstream selections changed')
    traces = {t['search_id']: t for t in records(root/'search/server-search-debug.jsonl')}
    jobs = []
    for j in (json.loads(line) for line in (upstream/'selections.jsonl').open()):
        for v in j['variants'].values():
            v.pop('fused')
            v.pop('prepared')
        t = traces[j['search_id']]
        check_versions(t['versions'], 'Search archive')
        if t['cascade']['cross_encoder']['status'] != 'ok' or t['cascade']['cross_encoder']['model'] != config.CE_MODEL:
            raise ValueError('Require complete matching historical CE')
        query = t['cascade']['cross_encoder']['query']
        old = {identity(c): c['_ce_score'] for c in t['ranked'] if c.get('_ce_score') is not None}
        novel, batches = {}, []
        with patch.multiple(config, **t['versions']['settings']):
            req = schemas.SearchRequest(user_id=t['user_id'], query=t['query'], options=t['options'])
            query_plan = dict(t['plan'], _query_specs=t['query_specs'])
            if cascade_rerank.query_text(req, query_plan) != query:
                raise ValueError('Historical CE query differs')
            for v in j['variants'].values():
                for c in v['coarse']:
                    key = identity(c)
                    if key not in old:
                        novel.setdefault(key, dict(key=key, candidate_id=c['id'], text=c['_rank_text']))
            batch = []
            for item in novel.values():
                if (len(item['text'].encode()) > config.CE_MAX_DOCUMENT_BYTES
                        or cross_encoder.request_size(query, [item['text']]) > config.CE_MAX_REQUEST_BYTES):
                    raise ValueError('New candidate exceeds CE input bounds')
                if batch and (len(batch) >= config.CE_BATCH_SIZE or
                        cross_encoder.request_size(query, [v['text'] for v in batch+[item]]) > config.CE_MAX_REQUEST_BYTES):
                    batches.append(batch)
                    batch = []
                batch.append(item)
            if batch:
                batches.append(batch)
            base_coarse = copy.deepcopy(j['variants']['baseline']['coarse'])
            for c in base_coarse:
                c['_ce_score'] = old[identity(c)]
            base = select(t, base_coarse)
            if ([c['id'] for c in base['fine']] != t['cascade']['fine']['selected_ids']
                    or [c['id'] for c in base['main']] != t['cascade']['listwise']['candidate_ids']
                    or base['main_bytes'] != t['cascade']['listwise']['input_bytes']):
                raise ValueError('Historical fine/main differs Q'+j['qa_id'])
        jobs.append(dict(j=j, t=t, old=old, novel=novel, batches=batches, query=query))
    if len(jobs) != 56:
        raise ValueError('Require full 56-question cohort')
    args.output.mkdir(parents=True, exist_ok=False)
    plan = dict(script_sha256=digest(Path(__file__)), upstream_sha256=digest(upstream/'selections.jsonl'),
        source_trace_sha256=digest(root/'search/server-search-debug.jsonl'),
        llm_model=config.LLM_MODEL, ce_model=config.CE_MODEL, historical_fine_main_verified=len(jobs),
        expected_calls=sum(len(j['batches']) for j in jobs), new_candidate_texts=sum(len(j['novel']) for j in jobs),
        jobs=[dict(qa_id=j['j']['qa_id'], query=j['query'], novel=list(j['novel'].values()),
            batches=[[v['key'] for v in b] for b in j['batches']]) for j in jobs], limitation=__doc__)
    (args.output/'plan.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k:v for k,v in plan.items() if k!='jobs'}), flush=True)
    handler = logging.FileHandler(args.output/'provider.log', mode='x')
    logging.getLogger('aml.metrics').addHandler(handler)
    logging.getLogger('aml.metrics').setLevel(logging.INFO)
    outputs, events = [], []
    for job in jobs:
        j,t = job['j'],job['t']
        scores = dict(job['old'])
        with patch.multiple(config, **t['versions']['settings']):
            for index, batch in enumerate(job['batches']):
                texts = [v['text'] for v in batch]
                bound = cross_encoder.request_size(job['query'], texts)
                event = dict(qa_id=j['qa_id'], batch=index, keys=[v['key'] for v in batch], request_bound=bound)
                with budget.scope(seconds=45, calls=1, tokens=bound) as usage:
                    try:
                        values = await cross_encoder.rerank(job['query'], texts, stage='diagnostic.semantic_upstream_ce')
                        scores.update(zip(event['keys'], values))
                        event['scores'] = values
                    except Exception as exc:
                        event.update(error=type(exc).__name__)
                event.update(calls=usage.calls, tokens=usage.tokens)
                events.append(event)
                with (args.output/'live.jsonl').open('a') as out:
                    out.write(json.dumps(event)+'\n')
                if 'error' in event:
                    raise RuntimeError('Incomplete CE batch; recorded failure, no silent fallback')
            variants = {}
            with patch('socket.socket.connect', side_effect=AssertionError('Offline fine/main selection')):
                for mode, v in j['variants'].items():
                    revised = copy.deepcopy(t)
                    revised['cascade']['coarse'] = v['coarse_metadata']
                    revised['fusion_edges'], revised['fusion_triples'] = v['fusion_edges'], v['fusion_triples']
                    coarse = copy.deepcopy(v['coarse'])
                    for c in coarse:
                        c['_ce_score'] = scores[identity(c)]
                    variants[mode] = select(revised, coarse)
                    variants[mode]['coarse_scores'] = {c['id']: c['_ce_score'] for c in coarse}
            record = dict(qa_id=j['qa_id'], user_id=j['user_id'], variants=variants)
            outputs.append(record)
            with (args.output/'selections.jsonl').open('a') as out:
                out.write(json.dumps(record, ensure_ascii=False)+'\n')
        print(json.dumps(dict(qa_id=j['qa_id'], new_texts=len(job['novel']), calls=len(job['batches']))), flush=True)
    csv.field_size_limit(32*1024*1024)
    with args.gold_csv.open() as f:
        gold = list(csv.DictReader(f))
    raw = {}
    with sqlite3.connect((root/'frozen-memory.db').as_uri()+'?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        for s in db.execute('SELECT * FROM source_messages'):
            raw.setdefault(s['user_id'], {})[audit.identity(dict(s))] = dict(s)
    rows = []
    for record in outputs:
        snippet = ast.literal_eval(gold[int(record['qa_id'])]['related_conversation_snippet'])
        coverage = {}
        for mode,v in record['variants'].items():
            coverage[mode] = {}
            for stage in ('fine','main'):
                cards = [dict(s,text=s.get('content',''),id=f'{i}:{k}') for i,c in enumerate(v[stage])
                         for k,s in enumerate(c['_packet_item']['sources'])]
                coverage[mode][stage] = {role:[x['full'] for x in audit.turn_coverage(
                    [audit.norm(s['content']) for s in snippet if s['role']==role], cards,
                    raw[record['user_id']], role=role)] for role in ('user','assistant')}
        rows.append(dict(qa_id=record['qa_id'], coverage=coverage))
    summary = {}
    for mode in outputs[0]['variants']:
        summary[mode] = {}
        for stage in ('fine','main'):
            turns = [r['coverage'][mode][stage]['user'] for r in rows]
            summary[mode][stage] = dict(reference_questions=sum(map(any,turns)),user_turns=sum(map(sum,turns)),
                complete_sets=sum(bool(x) and all(x) for x in turns),
                gained=[r['qa_id'] for r in rows if any(r['coverage'][mode][stage]['user'])
                    and not any(r['coverage']['baseline'][stage]['user'])],
                lost=[r['qa_id'] for r in rows if any(r['coverage']['baseline'][stage]['user'])
                    and not any(r['coverage'][mode][stage]['user'])],
                turn_losses=[r['qa_id'] for r in rows if any(a and not b for a,b in zip(
                    r['coverage']['baseline'][stage]['user'], r['coverage'][mode][stage]['user']))])
    result = dict(summary=summary, rows=rows, questions=len(rows),
        calls=sum(e['calls'] for e in events), tokens=sum(e['tokens'] for e in events),
        plan_sha256=digest(args.output/'plan.json'), selections_sha256=digest(args.output/'selections.jsonl'),
        limitation=__doc__)
    (args.output/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    (args.output/'exit-code').write_text('0\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--upstream',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gold-csv',type=Path,default=Path('data/raw/personamem-v2.csv'))
    asyncio.run(main(p.parse_args()))
