"""Verify the experimental production route against all frozen proposal stages."""
import argparse
import copy
import json
from pathlib import Path
import sqlite3
import sys
import time
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import cascade_rerank, config, graph_fusion, porter_recall, schemas, store
from evaluate_porter_upstream_ce import identity, select, selection_code, sha


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def verify_items(actual, expected, stage):
    if [c['id'] for c in actual] != [c['id'] for c in expected]:
        raise ValueError(stage + ': candidate order differs')
    # A new lexical hit also records a BM25 _score. This is retrieval metadata;
    # selection uses the unchanged CE/fusion keys, all checked below.
    keys = ('_rank_text', '_packet_item', '_fused', '_fusion', '_ce_score',
            '_relation_path_ids', '_supported_coverage_ids', '_coverage_ids', '_coverage_proofs')
    for a, b in zip(actual, expected, strict=True):
        for key in keys:
            if a.get(key) != b.get(key):
                raise ValueError(stage + ': ' + a['id'] + '/' + key + ' differs')


def main(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    target_result = json.loads((args.proposal / 'result.json').read_text())
    if target_result['questions'] != args.expected_count or target_result['selection_sha256'] != sha(args.proposal / 'selections.jsonl'):
        raise ValueError('Proposal changed or incomplete')
    targets = {j['qa_id']: j for j in records(args.proposal / 'selections.jsonl')}
    traces = {t['search_id']: t for t in records(args.run / 'server-search-debug.jsonl')}
    lexical = {j['search_id']: j for j in json.loads(args.lexical.read_text())['rows']}
    vp = json.loads((args.vectors / 'plan.json').read_text())
    vr = json.loads((args.vectors / 'result.json').read_text())
    if sha(args.vectors / 'plan.json') != vr['plan_sha256'] or sha(args.vectors / 'vectors.npz') != vr['vectors_sha256']:
        raise ValueError('Measured vectors changed')
    vectors = np.load(args.vectors / 'vectors.npz', allow_pickle=False)['vectors']
    lookup = {s: vectors[i] for i, s in enumerate(vp['texts'])}
    events = {}
    for e in records(args.ce / 'live.jsonl'):
        events.setdefault(e['qa_id'], []).append(e)
    inputs = [args.run / 'memory.db', args.run / 'server-search-debug.jsonl',
              args.proposal / 'selections.jsonl', args.upstream / 'selections.jsonl',
              args.ce / 'live.jsonl', args.lexical]
    input_hashes = {str(p): sha(p) for p in inputs}
    sources = [Path(__file__), Path('scripts/evaluate_porter_upstream_ce.py'), Path('scripts/probe_coarse_fine_core.py'), *sorted(Path('app').glob('*.py'))]
    source_hashes = {str(p): sha(p) for p in sources}
    st = store.Store(':memory:')
    with sqlite3.connect((args.run / 'memory.db').resolve().as_uri() + '?mode=ro', uri=True) as db:
        db.backup(st.conn)
    st.conn.execute('PRAGMA query_only=ON')
    code, fragment_sha = selection_code()
    rows = []
    try:
        with patch('socket.socket.connect', side_effect=AssertionError('Offline production parity')):
            for j in map(json.loads, (args.upstream / 'selections.jsonl').open()):
                t = traces[j['search_id']]
                expected = targets[j['qa_id']]
                st.read_user_id = t['user_id']
                req = schemas.SearchRequest(**{k: t[k] for k in ('user_id', 'query', 'options', 'top_k', 'reference_time', 'evidence_token_budget')},
                                            include_sensitive=t['requested_include_sensitive'])
                plan = dict(copy.deepcopy(t['plan']), _anchor=t['anchor_time'], _include_history=t['include_history'],
                    _include_sensitive=t['include_sensitive'], _query_specs=copy.deepcopy(t['query_specs']),
                    _used_queries=[s['text'] for s in t['query_specs']],
                    _coverage_requirements=t['coverage_manifest']['coverage']['requirements'],
                    _fusion_query_vectors=[lookup[s['text']].tolist() for s in t['query_specs']],
                    _routes=copy.deepcopy(t['routes']), _graph_fusion=copy.deepcopy(t['fusion']),
                    _fusion_edges=copy.deepcopy(t['fusion_edges']), _fusion_triples=copy.deepcopy(t['fusion_triples']))
                known = {c['id'] for r in t['routes'] for c in r['candidates']}
                embeddings = {c['id']: c.get('embedding') for c in st.get_amus_by_ids(list(known),
                    include_history=t['include_history'], include_sensitive=t['include_sensitive'])}
                routes = [[dict(c, embedding=embeddings.get(c['id'])) for c in r['candidates']] for r in t['routes']]
                prior = {c['id']: c['_fusion'] for c in t['fused']}
                similarity = graph_fusion._candidate_similarity
                def frozen_similarity(c, queries):
                    old = prior.get(c['id'])
                    if old is not None:
                        return None if old['semantic_basis'] == 'unavailable' else old['semantic']
                    if c['id'] in known:
                        raise ValueError('Missing recorded old semantic score')
                    return similarity(c, queries)
                base = j['variants']['baseline']
                with patch.multiple(config, **dict(t['versions']['settings'], RECALL_PORTER_SUPPLEMENT=True)), \
                        patch.object(graph_fusion, '_candidate_similarity', frozen_similarity):
                    started = time.monotonic()
                    fused, candidates = porter_recall.augment(st, req, plan, routes,
                        copy.deepcopy(base['fused']), copy.deepcopy(base['prepared']))
                    elapsed = time.monotonic() - started
                    info = plan['_porter_recall']
                    if info['status'] != 'ok' or info['lexical_ids'] != lexical[t['search_id']]['variants']['porter']['merged']:
                        raise ValueError('Production Porter lexical routes differ: ' + j['qa_id'])
                    if info['admitted_ids'] != expected['supplemental_ids']:
                        raise ValueError('Production new coarse candidates differ: ' + j['qa_id'])
                    verify_items(fused[:len(base['fused'])], base['fused'], 'old fused')
                    verify_items(candidates[:len(base['prepared'])], base['prepared'], 'old prepared')
                    ordinary = [c for c in candidates if not cascade_rerank.protected(c)]
                    coarse, meta = cascade_rerank.recall_shortlist(ordinary, plan['_coverage_requirements'], plan)
                    original = base['coarse']
                    verify_items(coarse[:len(original)], original, 'old coarse')
                    if meta['baseline_ids'] != base['coarse_metadata']['baseline_ids']:
                        raise ValueError('Original coarse eligibility differs')
                    actual = dict(coarse=coarse, coarse_metadata=meta,
                        fusion_edges=plan['_fusion_edges'], fusion_triples=plan['_fusion_triples'])
                    old_ids = set(t['cascade']['coarse']['selected_ids'])
                    scores = {identity(c): c.get('_ce_score') for c in t['ranked'] if c['id'] in old_ids}
                    charged = 0
                    for e in events.get(j['qa_id'], []):
                        scores.update(zip(e['keys'], e['scores'], strict=True))
                        charged += e['tokens']
                    for mode in ('retained_coarse', 'retained_coarse_charged'):
                        chosen_t = copy.deepcopy(t)
                        if mode.endswith('_charged'):
                            headroom = chosen_t['cascade']['admission_budget']['token_headroom']
                            if headroom is not None:
                                chosen_t['cascade']['admission_budget']['token_headroom'] = max(0, headroom - charged)
                        selected = select(chosen_t, actual, scores, code)
                        target = expected['variants'][mode]
                        for stage in ('fine', 'main'):
                            verify_items(selected[stage], target[stage], stage)
                        if selected['main_bytes'] != target['main_bytes'] or selected['coarse_scores'] != target['coarse_scores']:
                            raise ValueError('Production CE mapping/main bytes differ')
                    rows.append(dict(qa_id=j['qa_id'], elapsed_seconds=elapsed, supplemental_ids=info['admitted_ids'],
                                     coarse_count=len(coarse), status='passed'))
                if len(rows) % 10 == 0:
                    print(json.dumps(dict(verified=len(rows), expected=args.expected_count)), flush=True)
    finally:
        st.conn.close()
    if len(rows) != args.expected_count:
        raise ValueError('Incomplete production verification')
    if source_hashes != {str(p): sha(p) for p in sources} or input_hashes != {str(p): sha(p) for p in inputs}:
        raise ValueError('Implementation/input changed during verification')
    result = dict(status='passed', questions=len(rows), provider_calls=0, rows=rows,
        source_hashes=source_hashes, input_hashes=input_hashes, selection_ast_sha256=fragment_sha,
        elapsed_median=float(np.median([r['elapsed_seconds'] for r in rows])),
        elapsed_max=max(r['elapsed_seconds'] for r in rows),
        limitation='Measured optional local work on these frozen databases; not live HTTPS deadlines. Old recorded similarities are frozen, new candidates use measured query vectors.')
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('rows', 'source_hashes', 'input_hashes')}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--upstream', type=Path, required=True)
    p.add_argument('--ce', type=Path, required=True)
    p.add_argument('--proposal', type=Path, required=True)
    p.add_argument('--lexical', type=Path, required=True)
    p.add_argument('--vectors', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--expected-count', type=int, required=True)
    main(p.parse_args())
