"""Authenticate bridge support blocks and distinguish declared dependencies.

Read-only inventory: no scoring, source removal, admission change or gold-based
eligibility. Preserve raw packets and inspect frozen dependency versions. A
retrieval-only label does not prove that a support block is semantically useless.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
from unittest.mock import patch

from audit_retained_porter_https import require, same
from inventory_support_carriers import sha, batches
from inventory_visible_source_redundancy import records


def closure(ids, dependencies):
    found, queue = set(), list(ids)
    while queue:
        mid = queue.pop()
        for dep in dependencies.get(mid, []):
            if dep['source_id'] not in found:
                found.add(dep['source_id'])
                queue.append(dep['source_id'])
    return found


def observe(candidate, trace, db, dependencies, modules):
    pipeline, store, config, context = modules
    user = trace['user_id']
    sensitive = config.SENSITIVE_RECALL_ENABLED and trace['requested_include_sensitive']
    def get(ids, history=True):
        if not ids:
            return []
        query = ('SELECT * FROM amu WHERE id IN (' + ','.join('?' for _ in ids) + ')'
                 + ('' if history else ' AND valid_to IS NULL')
                 + ('' if sensitive else " AND sensitivity!='sensitive'")
                 + store._SUPPRESSED + ' AND user_id=?')
        result = []
        for row in batches(db.execute(query, [*ids, user])):
            # Embeddings are irrelevant to provenance and need not be retained.
            data = dict(row)
            data.pop('embedding', None)
            result.append(store.Store._row_to_dict(data))
        return result

    roots = [candidate['id'], *candidate.get('_equivalent_ids', [])]
    full = get(roots)
    full.sort(key=lambda row: roots.index(row['id']))
    mandatory = closure(roots, dependencies)
    bridges = get(candidate.get('_bridge_ids', []), bool(trace['include_history']))
    queue = [dict(d) for mid in roots for d in dependencies.get(mid, [])]
    queue.extend(dict(source_id=m['id'], source_version=m.get('version')) for m in bridges)
    visited, supports = {candidate['id']}, []
    unit = candidate['_packet_item']
    body = unit.get('_evidence_body')
    original = candidate.get('_rank_text', candidate['content'])
    primary = ('Conversation excerpt:' if candidate.get('type') == 'episode'
               and unit.get('sources') else candidate['content'])
    primary = pipeline._memory_prefix(candidate, full[0] if full else candidate, trace['anchor_time']) + primary
    reconstructed = primary
    while queue:
        dep = queue.pop()
        mid = dep['source_id']
        if mid in visited:
            continue
        visited.add(mid)
        require(len(visited) <= 64, 'Recorded dependency limit')
        values = get([mid])
        require(len(values) == 1 and values[0]['version'] == dep['source_version'],
                'Intact dependency version')
        support = values[0]
        prefix = '\n[support: ' + mid + '] ' + pipeline._memory_prefix(support, support, trace['anchor_time'])
        block = prefix + support['content']
        reconstructed += block
        supports.append(dict(id=mid, version=support['version'],
            origin='primary_dependency' if mid in mandatory else
                   'retrieval_bridge' if mid in candidate.get('_bridge_ids', []) else 'bridge_dependency',
            epistemic_status=support.get('epistemic_status'), memory_type=support.get('type'),
            protected=bool(support.get('type') == 'rule' or pipeline.profile.is_forget_rule(support)
                or support.get('resolution_status') != 'accepted' or support.get('valid_to')
                or (support.get('state') or {}).get('experience')),
            block_bytes=len(block.encode()), summary_bytes=len(support['content'].encode()),
            block_sha256=hashlib.sha256(block.encode()).hexdigest(),
            visible_as_standalone=mid in trace['cascade']['coarse']['selected_ids']))
        queue.extend(dict(d) for d in dependencies.get(mid, []))
    return dict(candidate_id=candidate['id'], in_fine=candidate['id'] in trace['cascade']['fine']['selected_ids'],
        in_main=candidate['id'] in trace['cascade']['listwise']['candidate_ids'],
        body_exact=isinstance(body, str) and body == reconstructed,
        rank_exact=isinstance(body, str) and context.with_evidence(body, unit['sources']) == original,
        body_excerpted=bool(candidate.get('_body_excerpted')),
        rank_excerpted=bool(candidate.get('_rank_excerpted')),
        protected=bool(candidate.get('type') == 'rule' or candidate.get('_user_rule') or unit.get('is_constraint')),
        relation_path_ids=candidate.get('_relation_path_ids', []),
        coverage_proofs=candidate.get('_coverage_proofs', []),
        declared_closure=sorted(mandatory), bridge_ids=candidate.get('_bridge_ids', []), supports=supports)


def main(args):
    run = args.run.resolve()
    inventory = json.loads(args.inventory.read_bytes())
    for name, digest in inventory['input_sha256'].items():
        same(sha(Path(name)), digest, 'Authenticated inventory input')
    cohort = args.inventory.stem
    previous = Path(__file__).resolve().parents[1] / f'runs/ce-inferred-primary-20261026/{cohort}/plan.json'
    same(sha(run / 'memory.db'), json.loads(previous.read_bytes())['database_sha256'], 'Frozen database')
    plan = json.loads((run.parent / 'plan.json').read_bytes())
    for name, digest in plan['source_files'].items():
        same(sha(run.parent / 'source' / name), digest, 'Frozen source')
    sys.path.insert(0, str(run.parent / 'source/memory_system'))
    from app import search_pipeline, store, config, answer_context
    modules = search_pipeline, store, config, answer_context
    results = {r['search_id']: r for r in records(run / 'completed/data/results/personamem-v2-32k.jsonl')}
    same(len(results), plan['questions'], 'Full cohort')
    counts, rows, seen = Counter(), [], set()
    with sqlite3.connect((run / 'memory.db').as_uri() + '?mode=ro', uri=True) as db, \
            patch('socket.socket.connect', side_effect=AssertionError('Offline provenance inventory')):
        db.row_factory = sqlite3.Row
        dependencies = {}
        for dep in batches(db.execute("SELECT * FROM view_dependencies WHERE view_kind='amu'")):
            dependencies.setdefault(dep['view_id'], []).append(dict(dep))
        for trace in records(run / 'server-search-debug.jsonl'):
            sid = trace['search_id']
            require(sid in results and sid not in seen, 'Unique trace')
            seen.add(sid)
            result = results[sid]
            same(trace['packet_hash'], result['packet_hash'], 'Packet')
            same(trace['user_id'], result['user_id'], 'User')
            coarse = set(trace['cascade']['coarse']['selected_ids'])
            candidates = [c for c in trace['ranked'] if c['id'] in coarse and c.get('_bridge_ids')]
            with patch.multiple(config, **trace['versions']['settings']):
                observed = [observe(c, trace, db, dependencies, modules) for c in candidates]
            for row in observed:
                counts['bridge_candidates'] += 1
                counts['body_exact'] += row['body_exact']
                counts['rank_exact'] += row['rank_exact']
                for support in row['supports']:
                    counts[support['origin']] += 1
                    if support['origin'] != 'primary_dependency' and support['epistemic_status'] == 'inferred':
                        counts['optional_inferred_supports'] += 1
                        counts['optional_inferred_unprotected'] += not support['protected']
            rows.append(dict(qa_id=result['qa_id'], conversation_id=result['conversation_id'], candidates=observed))
            print(json.dumps(dict(qa_id=result['qa_id'], bridge_candidates=len(observed))), flush=True)
    same(seen, set(results), 'Complete trace cohort')
    report = dict(status='complete', questions=len(rows), summary=dict(counts), rows=rows,
        provider_calls=0, production_changed=False, limitation=__doc__,
        input_sha256={str(path.resolve()): sha(path) for path in (
            Path(__file__), args.inventory, previous, run / 'memory.db',
            run / 'server-search-debug.jsonl', run.parent / 'plan.json')})
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(questions=len(rows), **counts)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
