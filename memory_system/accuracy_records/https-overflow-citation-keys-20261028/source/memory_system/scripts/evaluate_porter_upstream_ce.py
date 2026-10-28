"""Measure new CE pairs and replay exact production fine selection/main admission.

Preserves historical CE scores, including None for recorded failures. Original
fine/main must reproduce before any calls. New query/text pairs use the fixed
real CE. Historical main token headroom is frozen; no live Search or reader.
"""
import argparse
import ast
import asyncio
import copy
import csv
import hashlib
import inspect
import json
import logging
import math
from pathlib import Path
import sqlite3
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import budget, cascade_rerank, config, cross_encoder, schemas
from evaluation_model_guard import check_runtime
from probe_coarse_fine_core import admit
import audit_https_funnel as audit


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity(candidate):
    return candidate['id'] + ':' + hashlib.sha256(candidate['_rank_text'].encode()).hexdigest()


def selection_code():
    body = ast.parse(inspect.getsource(cascade_rerank.rank)).body[0].body
    def assignment(node, name):
        return isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)
    starts = [i for i, n in enumerate(body) if assignment(n, 'failed')]
    stops = [i for i, n in enumerate(body) if assignment(n, 'fine_ids')]
    if len(starts) != 1 or len(stops) != 1 or starts[0] >= stops[0]:
        raise ValueError('Production selection block changed')
    fragment = ast.Module(body=body[starts[0]:stops[0]], type_ignores=[])
    if any(isinstance(n, (ast.Await, ast.AsyncFor, ast.AsyncWith)) for n in ast.walk(fragment)):
        raise ValueError('Selection block is not offline')
    return compile(ast.fix_missing_locations(fragment), '<production-fine-selection>', 'exec'), hashlib.sha256(ast.dump(fragment).encode()).hexdigest()


def select(t, variant, scores, code):
    revised = copy.deepcopy(t)
    revised['cascade']['coarse'] = variant['coarse_metadata']
    revised['fusion_edges'], revised['fusion_triples'] = variant['fusion_edges'], variant['fusion_triples']
    coarse = copy.deepcopy(variant['coarse'])
    scored = {}
    for c in coarse:
        value = scores[identity(c)]
        if value is not None and (type(value) not in (int, float) or not math.isfinite(value)):
            raise ValueError('Invalid CE score')
        c['_ce_score'] = value
        if value is None:
            c.pop('_final', None)
            c['_score_kind'] = 'unscored'
        else:
            c['_final'], c['_score_kind'] = value, 'cross_encoder'
            scored[c['id']] = value
    plan = dict(copy.deepcopy(t['plan']), _query_specs=copy.deepcopy(t['query_specs']))
    env = dict(vars(cascade_rerank), coarse=coarse, scores=scored, trace=copy.deepcopy(revised['cascade']),
        req=schemas.SearchRequest(user_id=t['user_id'], query=t['query'], options=t['options']), plan=plan,
        requirements=t['coverage_manifest']['coverage']['requirements'])
    with patch.multiple(config, **t['versions']['settings']):
        exec(code, env)
        main, size = admit(revised, env['fine'], env['fine_selection'])
    return dict(fine=env['fine'], main=main, main_bytes=size, fine_selection=env['fine_selection'],
                coarse_scores={c['id']: c['_ce_score'] for c in coarse})


def score_outputs(args):
    csv.field_size_limit(32 * 1024 * 1024)
    with args.gold_csv.open() as stream:
        gold = list(csv.DictReader(stream))
    raw = {}
    with sqlite3.connect((args.run / 'memory.db').resolve().as_uri() + '?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        for s in db.execute('SELECT * FROM source_messages'):
            raw.setdefault(s['user_id'], {})[audit.identity(dict(s))] = dict(s)
    refs = {r['qa_id']: r for r in json.loads((args.run / 'funnel/stage-transport.json').read_text())['rows']}
    rows = []
    for line in (args.output / 'selections.jsonl').open():
        j = json.loads(line)
        snippet = ast.literal_eval(gold[int(j['qa_id'])]['related_conversation_snippet'])
        coverage = {}
        for mode, variant in j['variants'].items():
            coverage[mode] = {}
            for stage in ('fine', 'main'):
                cards = [dict(s, text=s.get('content', ''), id=f'{i}:{k}') for i, c in enumerate(variant[stage])
                         for k, s in enumerate(c['_packet_item']['sources'])]
                coverage[mode][stage] = {role: [x['full'] for x in audit.turn_coverage(
                    [audit.norm(s['content']) for s in snippet if s['role'] == role],
                    cards, raw[j['user_id']], role=role)] for role in ('user', 'assistant')}
        for ours, original in (('fine', 'fine'), ('main', 'listwise_input')):
            if coverage['baseline'][ours]['user'] != [v['full'] for v in refs[j['qa_id']]['visible'][original]['turn_coverage']]:
                raise ValueError('Historical reference transport differs: ' + j['qa_id'])
        rows.append(dict(qa_id=j['qa_id'], coverage=coverage))
    if len(rows) != args.expected_count:
        raise ValueError('Incomplete scored cohort')
    summary = {}
    for mode in ('baseline', 'porter_supplement'):
        summary[mode] = {}
        for stage in ('fine', 'main'):
            values = [r['coverage'][mode][stage]['user'] for r in rows]
            summary[mode][stage] = dict(reference_questions=sum(map(any, values)), user_turns=sum(map(sum, values)),
                complete_sets=sum(bool(v) and all(v) for v in values),
                gained=[r['qa_id'] for r in rows if any(r['coverage'][mode][stage]['user']) and not any(r['coverage']['baseline'][stage]['user'])],
                lost=[r['qa_id'] for r in rows if not any(r['coverage'][mode][stage]['user']) and any(r['coverage']['baseline'][stage]['user'])],
                turn_losses=[r['qa_id'] for r in rows if any(a and not b for a, b in zip(
                    r['coverage']['baseline'][stage]['user'], r['coverage'][mode][stage]['user'], strict=True))])
    return dict(summary=summary, rows=rows)


async def main(args):
    check_runtime()
    if config.FAKE or config.CE_MODEL != 'qwen3-rerank':
        raise ValueError('Requires the fixed real CE model')
    prior = json.loads((args.upstream / 'result.json').read_text())
    if prior['questions'] != args.expected_count or sha(args.upstream / 'selections.jsonl') != prior['selection_sha256']:
        raise ValueError('Incomplete or changed upstream selection')
    if prior['summary']['porter_supplement']['coarse']['turn_losses']:
        raise ValueError('Upstream failed retention gate')
    if any(sha(Path(p)) != v for p, v in prior['source_hashes'].items()):
        raise ValueError('Upstream implementation changed')
    traces = {t['search_id']: t for t in map(json.loads, (args.run / 'server-search-debug.jsonl').read_text().splitlines())}
    code, fragment_sha = selection_code()
    jobs = []
    for line in (args.upstream / 'selections.jsonl').open():
        j = json.loads(line)
        for v in j['variants'].values():
            v.pop('fused')
            v.pop('prepared')
        t = traces[j['search_id']]
        ce = t['cascade']['cross_encoder']
        if ce['status'] not in ('ok', 'partial') or ce['model'] != config.CE_MODEL:
            raise ValueError('Unexpected historical CE policy')
        coarse_ids = set(t['cascade']['coarse']['selected_ids'])
        old = {identity(c): c.get('_ce_score') for c in t['ranked'] if c['id'] in coarse_ids}
        successful = {mid for b in ce['batches'] if b['status'] == 'ok' for mid in b['candidate_ids']}
        actual = {c['id'] for c in t['ranked'] if c['id'] in coarse_ids and c.get('_ce_score') is not None}
        if successful != actual or len(actual) != ce['scored_count'] or len(old) != len(coarse_ids):
            raise ValueError('Historical scores not backed by successful calls')
        query = ce['query']
        with patch.multiple(config, **t['versions']['settings']):
            req = schemas.SearchRequest(user_id=t['user_id'], query=t['query'], options=t['options'])
            if cascade_rerank.query_text(req, dict(t['plan'], _query_specs=t['query_specs'])) != query:
                raise ValueError('Historical CE query changed')
            with patch('socket.socket.connect', side_effect=AssertionError('Offline baseline selection')):
                base = select(t, j['variants']['baseline'], old, code)
            if ([c['id'] for c in base['fine']] != t['cascade']['fine']['selected_ids']
                    or [c['id'] for c in base['main']] != t['cascade']['listwise']['candidate_ids']
                    or base['main_bytes'] != t['cascade']['listwise']['input_bytes']):
                raise ValueError('Original fine/main mismatch: ' + j['qa_id'])
            novel = {}
            for v in j['variants'].values():
                for c in v['coarse']:
                    key = identity(c)
                    if key not in old:
                        novel.setdefault(key, dict(key=key, candidate_id=c['id'], text=c['_rank_text']))
            batches, batch = [], []
            for item in novel.values():
                if (len(item['text'].encode()) > config.CE_MAX_DOCUMENT_BYTES or
                        cross_encoder.request_size(query, [item['text']]) > config.CE_MAX_REQUEST_BYTES):
                    raise ValueError('New CE text exceeds request bounds')
                if batch and (len(batch) >= config.CE_BATCH_SIZE or cross_encoder.request_size(query, [v['text'] for v in batch + [item]]) > config.CE_MAX_REQUEST_BYTES):
                    batches.append(batch)
                    batch = []
                batch.append(item)
            if batch:
                batches.append(batch)
        jobs.append(dict(j=j, t=t, old=old, base=base, novel=novel, batches=batches, query=query))
    if len(jobs) != args.expected_count:
        raise ValueError('Incomplete input cohort')
    args.output.mkdir(parents=True, exist_ok=False)
    source_files = [Path(__file__), Path('scripts/probe_coarse_fine_core.py'), Path('scripts/audit_https_funnel.py'), *sorted(Path('app').glob('*.py'))]
    plan = dict(questions=len(jobs), expected_calls=sum(len(j['batches']) for j in jobs),
        new_pairs=sum(len(j['novel']) for j in jobs), upstream_sha256=sha(args.upstream / 'selections.jsonl'),
        trace_sha256=sha(args.run / 'server-search-debug.jsonl'), database_sha256=sha(args.run / 'memory.db'),
        selection_ast_sha256=fragment_sha, source_hashes={str(p): sha(p) for p in source_files},
        historical_partial=[j['j']['qa_id'] for j in jobs if j['t']['cascade']['cross_encoder']['status'] == 'partial'],
        jobs=[dict(qa_id=j['j']['qa_id'], query=j['query'], novel=list(j['novel'].values()),
                   batches=[[v['key'] for v in b] for b in j['batches']]) for j in jobs], limitation=__doc__)
    (args.output / 'plan.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in plan.items() if k not in ('jobs', 'source_hashes')}), flush=True)
    handler = logging.FileHandler(args.output / 'provider.log', mode='x')
    logging.getLogger('aml.metrics').addHandler(handler)
    logging.getLogger('aml.metrics').setLevel(logging.INFO)
    events = []
    for job in jobs:
        scores = dict(job['old'])
        with patch.multiple(config, **job['t']['versions']['settings']):
            for index, batch in enumerate(job['batches']):
                texts = [v['text'] for v in batch]
                bound = cross_encoder.request_size(job['query'], texts)
                event = dict(qa_id=job['j']['qa_id'], batch=index, keys=[v['key'] for v in batch], request_bound=bound)
                with budget.scope(seconds=45, calls=1, tokens=bound) as usage:
                    try:
                        values = await cross_encoder.rerank(job['query'], texts, stage='diagnostic.lexical_porter_ce')
                        if len(values) != len(batch) or not all(math.isfinite(v) for v in values):
                            raise ValueError('Invalid new CE scores')
                        scores.update(zip(event['keys'], values, strict=True))
                        event['scores'] = values
                    except Exception as exc:
                        event['error'] = type(exc).__name__
                event.update(calls=usage.calls, tokens=usage.tokens)
                events.append(event)
                with (args.output / 'live.jsonl').open('a') as stream:
                    stream.write(json.dumps(event) + '\n')
                if 'error' in event:
                    raise RuntimeError('Incomplete new CE batch; original error retained, no rerun')
            with patch('socket.socket.connect', side_effect=AssertionError('Offline treatment selection')):
                treatment = select(job['t'], job['j']['variants']['porter_supplement'], scores, code)
            record = dict(qa_id=job['j']['qa_id'], user_id=job['j']['user_id'],
                search_id=job['j']['search_id'], variants=dict(baseline=job['base'], porter_supplement=treatment))
            with (args.output / 'selections.jsonl').open('a') as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + '\n')
        print(json.dumps(dict(qa_id=record['qa_id'], new_pairs=len(job['novel']), calls=len(job['batches']))), flush=True)
    final = score_outputs(args)
    if any(sha(Path(p)) != v for p, v in plan['source_hashes'].items()):
        raise ValueError('Implementation changed during CE evaluation')
    final.update(questions=len(jobs), calls=sum(e['calls'] for e in events), tokens=sum(e['tokens'] for e in events),
        plan_sha256=sha(args.output / 'plan.json'), selections_sha256=sha(args.output / 'selections.jsonl'),
        gold_sha256=sha(args.gold_csv), limitation=__doc__)
    (args.output / 'result.json').write_text(json.dumps(final, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in final.items() if k != 'rows'}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--upstream', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--expected-count', type=int, required=True)
    p.add_argument('--gold-csv', type=Path, default=Path('data/raw/personamem-v2.csv'))
    asyncio.run(main(p.parse_args()))
