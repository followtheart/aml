"""Replay source-semantic nominations through real fusion, preparation and coarse.

Original routes and original candidates' recorded semantic similarities are
frozen. Newly admitted candidates use the measured query vectors and original
AMU embeddings. Source vectors only nominate linked eligible memories; they
never replace memory scores, evidence, roles or personal-fact judgments.
No CE/LLM calls, no new graph expansion, no final packet/accuracy prediction.
References are opened only after every variant selection is fixed.
"""
import argparse
import ast
import copy
import csv
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import cascade_rerank, config, graph_fusion, schemas, search_pipeline as search, store
from evaluation_model_guard import check_runtime, check_versions
from probe_semantic_source_context import ranked_sources
import audit_https_funnel as audit


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def nominate(measurement, row, mode, st, t):
    sources = measurement['sources']
    indices = ranked_sources(row['modes'][mode])[:config.RECALL_SOURCE_LIMIT]
    lanes = []
    for si in indices:
        s = sources[si]
        if s['user_id'] != t['user_id']:
            raise ValueError('Cross-user source')
        found = {m['id']: m for m in st.get_amus_by_ids(s['amu_ids'],
            include_history=t['include_history'], include_sensitive=t['include_sensitive'])}
        lanes.append([found[mid] for mid in s['amu_ids'] if mid in found
                      and found[mid]['user_id'] == t['user_id']
                      and found[mid]['tier'] != 'cold'])
    # One linked memory per nominated source before repeated carriers.
    route = search._merge_query_results(lanes, config.RECALL_SOURCE_LIMIT)
    return route, dict(source_indices=indices, candidate_ids=[m['id'] for m in route])


def stages(st, t, route, vectors):
    req = schemas.SearchRequest(**{k: t[k] for k in
        ('user_id', 'query', 'options', 'top_k', 'reference_time', 'evidence_token_budget')},
        include_sensitive=t['requested_include_sensitive'])
    plan = copy.deepcopy(t['plan'])
    plan.update(_anchor=t['anchor_time'], _include_history=t['include_history'],
        _include_sensitive=t['include_sensitive'], _query_specs=t['query_specs'],
        _used_queries=[s['text'] for s in t['query_specs']],
        _coverage_requirements=t['coverage_manifest']['coverage']['requirements'],
        _fusion_query_vectors=vectors.tolist(), _routes=copy.deepcopy(t['routes']))
    old_ids = {c['id'] for r in t['routes'] for c in r['candidates']}
    embeddings = {c['id']: c.get('embedding') for c in st.get_amus_by_ids(list(old_ids),
        include_history=t['include_history'], include_sensitive=t['include_sensitive'])}
    routes = [[dict(c, embedding=embeddings.get(c['id'])) for c in r['candidates']] for r in t['routes']]
    if route is not None:
        search._route(plan, routes, 'source_semantic', copy.deepcopy(route))
    routes = [[c for c in group if search._profile_not_expired(c, t['anchor_time'], t['include_history'])]
              for group in routes]
    historical = {c['id']: c['_fusion'] for c in t['fused']}
    similarity = graph_fusion._candidate_similarity
    unrecorded_old = set()

    def frozen_similarity(c, qs):
        prior = historical.get(c['id'])
        if prior:
            return None if prior['semantic_basis'] == 'unavailable' else prior['semantic']
        if c['id'] in old_ids:
            unrecorded_old.add(c['id'])
        return similarity(c, qs)

    with patch.object(graph_fusion, '_candidate_similarity', frozen_similarity):
        fused = search._fuse_candidates(st, req, plan, routes)
    prepared = search._foresight_filter(search._prepare_candidates(st, req, plan, fused), plan)
    prepared = [c for c in prepared if c.get('view_status') != 'stale']
    coarse, metadata = cascade_rerank.coarse_shortlist(
        [c for c in prepared if not cascade_rerank.protected(c)], plan['_coverage_requirements'])
    return dict(fused=fused, prepared=prepared, coarse=coarse, coarse_metadata=metadata,
        unrecorded_original_similarities=sorted(unrecorded_old),
        fusion=plan['_graph_fusion'], omissions=plan.get('_unit_omitted', []),
        fusion_edges=plan.get('_fusion_edges', []), fusion_triples=plan.get('_fusion_triples', []))


def finish(args, jobs, st):
    # All variants are now frozen. Only the scorer below sees reference data.
    raw = {}
    for s in st.conn.execute('SELECT * FROM source_messages'):
        raw.setdefault(s['user_id'], {})[audit.identity(dict(s))] = dict(s)
    csv.field_size_limit(32*1024*1024)
    with args.gold_csv.open() as f:
        gold = list(csv.DictReader(f))
    rows = []
    for job in jobs:
        snippet = ast.literal_eval(gold[int(job['qa_id'])]['related_conversation_snippet'])
        coverage = {}
        for mode, v in job['variants'].items():
            coverage[mode] = {}
            for stage in ('prepared', 'coarse'):
                cards = [dict(s, text=s.get('content',''), id=f'{i}:{j}')
                    for i,c in enumerate(v[stage]) for j,s in enumerate(c['_packet_item']['sources'])]
                coverage[mode][stage] = {role:[x['full'] for x in audit.turn_coverage(
                    [audit.norm(s['content']) for s in snippet if s['role']==role],
                    cards, raw[job['user_id']], role=role)] for role in ('user', 'assistant')}
        rows.append(dict(qa_id=job['qa_id'], coverage=coverage))
    summary = {}
    for mode in ('baseline', 'whole', 'window2', 'exchange'):
        summary[mode] = {}
        for stage in ('prepared', 'coarse'):
            data = [r['coverage'][mode][stage]['user'] for r in rows]
            summary[mode][stage] = dict(reference_questions=sum(any(x) for x in data),
                user_turns=sum(map(sum,data)), complete_sets=sum(bool(x) and all(x) for x in data),
                gained=[r['qa_id'] for r in rows if any(r['coverage'][mode][stage]['user'])
                    and not any(r['coverage']['baseline'][stage]['user'])],
                lost=[r['qa_id'] for r in rows if any(r['coverage']['baseline'][stage]['user'])
                    and not any(r['coverage'][mode][stage]['user'])],
                turn_losses=[r['qa_id'] for r in rows if any(a and not b for a,b in zip(
                    r['coverage']['baseline'][stage]['user'],r['coverage'][mode][stage]['user']))])
    result = dict(summary=summary, rows=rows, questions=len(rows), historical_fusion_prepared_coarse_reproduced=len(rows),
        script_sha256=digest(Path(__file__)), database_sha256=digest(args.run/'frozen-memory.db'),
        selections_sha256=digest(args.output/'selections.jsonl'), limitation=__doc__)
    if len(rows) != 56 or len({r['qa_id'] for r in rows}) != 56:
        raise ValueError('Incomplete saved selections')
    provenance = args.output/'selection-provenance.json'
    if provenance.exists():
        result['selection_provenance'] = json.loads(provenance.read_text())
        if result['selection_provenance']['selections_sha256'] != result['selections_sha256']:
            raise ValueError('Saved selections changed')
    (args.output/'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    (args.output/'exit-code').write_text('0\n')
    print(json.dumps(summary), flush=True)


def main(args):
    check_runtime()
    if args.score_existing:
        with sqlite3.connect((args.run/'frozen-memory.db').resolve().as_uri()+'?mode=ro', uri=True) as db:
            db.row_factory = sqlite3.Row
            from types import SimpleNamespace
            finish(args, (json.loads(line) for line in (args.output/'selections.jsonl').open()), SimpleNamespace(conn=db))
        return
    args.output.mkdir(parents=True, exist_ok=False)
    root = args.run.resolve()
    source_root, exchange_root = root/'source-vector-diagnostic', root/'exchange-vector-diagnostic'
    measurements = []
    for directory in (source_root, exchange_root):
        p = json.loads((directory/'plan.json').read_text())
        r = json.loads((directory/'rankings.json').read_text())
        if digest(directory/'plan.json') != r['plan_sha256'] or digest(directory/'vectors.npz') != r['vectors_sha256']:
            raise ValueError('Changed measured inputs')
        measurements.append((p, {row['search_id']: row for row in r['rankings']}, directory))
    source_plan = measurements[0][0]
    vectors = np.load(source_root/'vectors.npz', allow_pickle=False)['vectors']
    vector_lookup = {s: vectors[i] for i, s in enumerate(source_plan['texts'])}
    if digest(root/'frozen-memory.db') != source_plan['database_sha256']:
        raise ValueError('Changed frozen database')
    st = store.Store(':memory:')
    with sqlite3.connect((root/'frozen-memory.db').as_uri()+'?mode=ro', uri=True) as db:
        db.backup(st.conn)
    st.conn.execute('PRAGMA query_only=ON')
    inputs = records(root/'http-reader-inputs.jsonl')
    qa = {job['payload']['search_id']: job['qa']['id'] for job in inputs}
    traces = records(root/'search/server-search-debug.jsonl')
    if len(traces) != len(qa):
        raise ValueError('Cohort mismatch')
    try:
        with patch('socket.socket.connect', side_effect=AssertionError('Offline upstream diagnostic')):
            for t in traces:
                check_versions(t['versions'], 'original Search')
                st.read_user_id = t['user_id']
                query_vectors = np.asarray([vector_lookup[s['text']] for s in t['query_specs']])
                with patch.multiple(config, **t['versions']['settings']):
                    base = stages(st, t, None, query_vectors)
                    if ([c['id'] for c in base['fused']] != [c['id'] for c in t['fused']]
                            or any(c['_fusion'] != old['_fusion'] or c['_fused'] != old['_fused']
                                   for c, old in zip(base['fused'], t['fused']))):
                        raise ValueError('Historical fusion differs Q'+qa[t['search_id']])
                    original = {c['id']: c for c in t['ranked']}
                    if ([c['id'] for c in base['coarse']] != t['cascade']['coarse']['selected_ids']
                            or any(c['_rank_text'] != original[c['id']]['_rank_text']
                                or c['_packet_item'] != original[c['id']]['_packet_item'] for c in base['prepared'])):
                        raise ValueError('Historical prepared/coarse differs Q'+qa[t['search_id']])
                    if base['unrecorded_original_similarities']:
                        raise ValueError('Unrecorded original fusion similarity')
                    variants = {'baseline': base}
                    nominations = {}
                    for p, ranks, _ in measurements:
                        row = ranks[t['search_id']]
                        for mode in row['modes']:
                            route, nominations[mode] = nominate(p, row, mode, st, t)
                            variants[mode] = stages(st, t, route, query_vectors)
                    record = dict(qa_id=qa[t['search_id']], search_id=t['search_id'], user_id=t['user_id'],
                        nominations=nominations, variants=variants)
                    with (args.output/'selections.jsonl').open('a') as out:
                        out.write(json.dumps(record, ensure_ascii=False, default=lambda a:a.tolist())+'\n')
                print(json.dumps(dict(qa_id=qa[t['search_id']], reproduced=True,
                    coarse_counts={m:len(v['coarse']) for m,v in variants.items()})), flush=True)
        finish(args, (json.loads(line) for line in (args.output/'selections.jsonl').open()), st)
    finally:
        st.conn.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--score-existing', action='store_true')
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gold-csv', type=Path, default=Path('data/raw/personamem-v2.csv'))
    main(p.parse_args())
