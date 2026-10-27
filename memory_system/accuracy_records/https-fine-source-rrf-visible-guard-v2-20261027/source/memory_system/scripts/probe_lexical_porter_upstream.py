"""Replay a bounded supplemental Porter route through fusion, preparation, coarse.

All original routes remain, including historical graph/follow-up routes. This
is an upstream diagnostic, not a complete live Search or an answer evaluation.
No CE/LLM calls. New candidates use genuine measured query/AMU embeddings;
old candidates keep recorded semantic scores. References enter after selection.
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
from app import config, search_pipeline as search, store
import audit_https_funnel as audit
from probe_semantic_upstream import stages


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def nominate(proposal, t, st):
    known = {c['id'] for r in t['routes'] for c in r['candidates']}
    lanes = [[dict(id=mid) for mid in ids if mid not in known]
             for ids in proposal['variants']['porter']['merged'].values()]
    ids = [c['id'] for c in search._merge_query_results(lanes, 20)]
    found = {c['id']: c for c in st.get_amus_by_ids(ids, include_history=t['include_history'],
                                                  include_sensitive=t['include_sensitive'])}
    if set(found) != set(ids) or any(c['user_id'] != t['user_id'] for c in found.values()):
        raise ValueError('Nominated memories do not satisfy request eligibility')
    return [found[mid] for mid in ids]


def score(root, output, st, gold_csv):
    csv.field_size_limit(32 * 1024 * 1024)
    with gold_csv.open() as stream:
        gold = list(csv.DictReader(stream))
    raw = {}
    for s in st.conn.execute('SELECT * FROM source_messages'):
        raw.setdefault(s['user_id'], {})[audit.identity(dict(s))] = dict(s)
    rows = []
    expected = json.loads((root / 'funnel/stage-transport.json').read_text())
    if expected['summary']['progress_snapshot']:
        raise ValueError('Incomplete source funnel')
    refs = {(r['conversation_id'], r['qa_id']): r for r in expected['rows']}
    for line in (output / 'selections.jsonl').open():
        j = json.loads(line)
        snippet = ast.literal_eval(gold[int(j['qa_id'])]['related_conversation_snippet'])
        coverage = {}
        for mode, variant in j['variants'].items():
            coverage[mode] = {}
            for stage in ('prepared', 'coarse'):
                cards = [dict(s, text=s.get('content', ''), id=f'{i}:{k}')
                         for i, c in enumerate(variant[stage]) for k, s in enumerate(c['_packet_item']['sources'])]
                coverage[mode][stage] = {role: [x['full'] for x in audit.turn_coverage(
                    [audit.norm(s['content']) for s in snippet if s['role'] == role],
                    cards, raw[j['user_id']], role=role)] for role in ('user', 'assistant')}
        ref = refs[j['conversation_id'], j['qa_id']]
        for ours, old in (('prepared', 'fused'), ('coarse', 'coarse')):
            if coverage['baseline'][ours]['user'] != [v['full'] for v in ref['visible'][old]['turn_coverage']]:
                raise ValueError('Original reference transport differs: ' + j['qa_id'])
        rows.append(dict(qa_id=j['qa_id'], coverage=coverage, nominated_ids=j['nominated_ids'],
            candidate_counts={m: {s: len(v[s]) for s in ('fused', 'prepared', 'coarse')}
                              for m, v in j['variants'].items()}))
    summary = {}
    for mode in ('baseline', 'porter_supplement'):
        summary[mode] = {}
        for stage in ('prepared', 'coarse'):
            data = [r['coverage'][mode][stage]['user'] for r in rows]
            summary[mode][stage] = dict(reference_questions=sum(any(x) for x in data),
                user_turns=sum(map(sum, data)), complete_sets=sum(bool(x) and all(x) for x in data),
                gained=[r['qa_id'] for r in rows if any(r['coverage'][mode][stage]['user'])
                        and not any(r['coverage']['baseline'][stage]['user'])],
                lost=[r['qa_id'] for r in rows if not any(r['coverage'][mode][stage]['user'])
                      and any(r['coverage']['baseline'][stage]['user'])],
                turn_losses=[r['qa_id'] for r in rows if any(a and not b for a, b in zip(
                    r['coverage']['baseline'][stage]['user'], r['coverage'][mode][stage]['user'], strict=True))])
    return dict(summary=summary, questions=len(rows), rows=rows, gold_sha256=sha(gold_csv))


def main(args):
    root = args.run.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    if (root / 'exit-code').read_text().strip() != '0':
        raise ValueError('Requires completed source run')
    proposal = json.loads(args.proposal.read_text())
    if any(sha(Path(p)) != value for p, value in proposal['input_hashes'].items()):
        raise ValueError('Lexical diagnostic inputs changed')
    if str(root / 'server-search-debug.jsonl') not in proposal['input_hashes']:
        raise ValueError('Proposal is from another run')
    proposals = {r['search_id']: r for r in proposal['rows']}
    results = records(root / 'completed/data/results/personamem-v2-32k.jsonl')
    traces = {r['search_id']: r for r in records(root / 'server-search-debug.jsonl')}
    if len(results) != args.expected_count or set(proposals) != {r['search_id'] for r in results}:
        raise ValueError('Incomplete cohort')
    vp = json.loads((args.vectors / 'plan.json').read_text())
    vr = json.loads((args.vectors / 'result.json').read_text())
    if sha(args.vectors / 'plan.json') != vr['plan_sha256'] or sha(args.vectors / 'vectors.npz') != vr['vectors_sha256']:
        raise ValueError('Query vector hashes differ')
    if vp['model'] != config.EMBED_MODEL or vp['dimension'] != config.EMBED_DIM:
        raise ValueError('Embedding model/dimension differs')
    vectors = np.load(args.vectors / 'vectors.npz', allow_pickle=False)['vectors']
    lookup = {s: vectors[i] for i, s in enumerate(vp['texts'])}
    dependencies = [Path(__file__), Path('scripts/probe_semantic_upstream.py'),
                    Path('scripts/audit_https_funnel.py'), *sorted(Path('app').glob('*.py'))]
    hashes = {str(p): sha(p) for p in dependencies}
    st = store.Store(':memory:')
    with sqlite3.connect((root / 'memory.db').as_uri() + '?mode=ro', uri=True) as source:
        source.backup(st.conn)
    st.conn.execute('PRAGMA query_only=ON')
    original_route = search._route

    def route_descriptor(plan, routes, channel, *args, **kwargs):
        return original_route(plan, routes, 'lexical_porter' if channel == 'source_semantic' else channel,
                              *args, **kwargs)

    try:
        with patch('socket.socket.connect', side_effect=AssertionError('Offline upstream replay')), \
                patch.object(search, '_route', route_descriptor):
            for result in results:
                t = traces[result['search_id']]
                st.read_user_id = t['user_id']
                query_vectors = np.asarray([lookup[s['text']] for s in t['query_specs']])
                with patch.multiple(config, **t['versions']['settings']):
                    base = stages(st, t, None, query_vectors)
                    if ([c['id'] for c in base['fused']] != [c['id'] for c in t['fused']]
                            or any(c['_fusion'] != old['_fusion'] or c['_fused'] != old['_fused']
                                   for c, old in zip(base['fused'], t['fused'], strict=True))):
                        raise ValueError('Original fusion differs: ' + result['qa_id'])
                    original = {c['id']: c for c in t['ranked']}
                    if ([c['id'] for c in base['coarse']] != t['cascade']['coarse']['selected_ids']
                            or any(c['_rank_text'] != original[c['id']]['_rank_text']
                                or c['_packet_item'] != original[c['id']]['_packet_item'] for c in base['prepared'])):
                        raise ValueError('Original prepared/coarse differs: ' + result['qa_id'])
                    if base['unrecorded_original_similarities']:
                        raise ValueError('Original semantic scores unavailable')
                    nominated = nominate(proposals[t['search_id']], t, st)
                    variant = stages(st, t, nominated, query_vectors)
                    if variant['unrecorded_original_similarities']:
                        raise ValueError('Unexpected original semantic scores')
                    j = dict(qa_id=result['qa_id'], conversation_id=result['conversation_id'],
                        search_id=t['search_id'], user_id=t['user_id'], nominated_ids=[c['id'] for c in nominated],
                        variants=dict(baseline=base, porter_supplement=variant))
                    with (args.output / 'selections.jsonl').open('a') as stream:
                        stream.write(json.dumps(j, ensure_ascii=False, default=lambda a: a.tolist()) + '\n')
                print(json.dumps(dict(qa_id=result['qa_id'], original_reproduced=True, nominated=len(nominated))), flush=True)
            final = score(root, args.output, st, args.gold_csv)
    finally:
        st.conn.close()
    if hashes != {str(p): sha(p) for p in dependencies}:
        raise ValueError('Replay source code changed')
    final.update(source_hashes=hashes, proposal_sha256=sha(args.proposal), vector_result_sha256=sha(args.vectors / 'result.json'),
        selection_sha256=sha(args.output / 'selections.jsonl'), provider_calls=0, limitation=__doc__)
    (args.output / 'result.json').write_text(json.dumps(final, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(final['summary']), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--proposal', type=Path, required=True)
    p.add_argument('--vectors', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--expected-count', type=int, required=True)
    p.add_argument('--gold-csv', type=Path, default=Path('data/raw/personamem-v2.csv'))
    main(p.parse_args())
