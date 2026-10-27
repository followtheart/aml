"""Audit support inventory identities and rendering limits without model calls.

This supplement does not change the original inventory or selection policy.
Packet-source byte totals are not main-prompt savings. Literal containment and
lexical matches do not prove claim redundancy, relevance or safe removal.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

from inventory_support_carriers import sha
from inventory_visible_source_redundancy import records
from audit_retained_porter_https import require, same


def main(args):
    inventory = json.loads(args.inventory.read_bytes())
    cohort = args.inventory.stem
    project = Path(__file__).resolve().parents[1]
    run = project / f'runs/https-retained-porter-20261025-{cohort}/control'
    frozen = run.parent / 'source/memory_system'
    previous = project / f'runs/ce-inferred-primary-20261026/{cohort}/plan.json'
    expected_db = json.loads(previous.read_bytes())['database_sha256']
    same(sha(run / 'memory.db'), expected_db, 'Earlier audited database')
    same(inventory['input_sha256'][str(run / 'memory.db')], expected_db,
         'Inventory database identity')
    for name, digest in inventory['input_sha256'].items():
        same(sha(Path(name)), digest, 'Inventory input ' + name)
    plan = json.loads((run.parent / 'plan.json').read_bytes())
    for name, digest in plan['source_files'].items():
        same(sha(run.parent / 'source' / name), digest, 'Frozen source')
    sys.path.insert(0, str(frozen))
    from app import answer_context

    by_question = {(row['conversation_id'], row['qa_id']): row for row in inventory['rows']}
    same(len(by_question), plan['questions'], 'Full inventory cohort')
    by_search = {}
    for result in records(run / 'completed/data/results/personamem-v2-32k.jsonl'):
        sid = result['search_id']
        require(sid not in by_search, 'Unique result search identity')
        row = by_question[result['conversation_id'], result['qa_id']]
        same((row['user_id'], row['packet_hash']),
             (result['user_id'], result['packet_hash']), 'Inventory result identity')
        by_search[sid] = row
    same(len(by_search), len(by_question), 'Complete identity mapping')
    mismatches, overlaps, selected_rows, seen = [], [], [], set()
    with patch('socket.socket.connect', side_effect=AssertionError('Offline audit')):
        for trace in records(run / 'server-search-debug.jsonl'):
            sid = trace['search_id']
            require(sid in by_search and sid not in seen, 'Unique trace identity')
            seen.add(sid)
            row = by_search[sid]
            same(trace['packet_hash'], row['packet_hash'], 'Packet identity')
            same(trace['user_id'], row['user_id'], 'Trace user identity')
            pool = {c['id']: c for c in trace['ranked']}
            main = row['main']
            same([c['candidate_id'] for c in main],
                 trace['cascade']['listwise']['candidate_ids'], 'Main identity')
            authenticated = set()
            for item in main:
                candidate = pool[item['candidate_id']]
                unit = candidate['_packet_item']
                body = unit.get('_evidence_body')
                rank = candidate.get('_rank_text', candidate['content'])
                rendered = answer_context.with_evidence(body, unit['sources']) if isinstance(body, str) else None
                valid = rendered == rank
                same(valid, item['authenticated_full_view'], 'Rendering inventory')
                if valid:
                    authenticated.add(candidate['id'])
                else:
                    sources = [s for s in unit['sources'] if s.get('content')]
                    mismatches.append(dict(qa_id=row['qa_id'], candidate_id=candidate['id'],
                        memory_type=candidate.get('type'), document=candidate.get('_document'),
                        rank_excerpted=candidate.get('_rank_excerpted'),
                        body_available=isinstance(body, str),
                        unit_content_equals_rendered=unit.get('content') == rendered,
                        rank_bytes=len(rank.encode()), packet_bytes=len(unit['content'].encode()),
                        packet_sources=len(sources),
                        packet_sources_literal_in_rank=sum(s['content'] in rank for s in sources),
                        inherited_sources=item['inherited_source_count'],
                        inherited_only=item['matches_only_inherited'],
                        rank_sha256=hashlib.sha256(rank.encode()).hexdigest()))
            selected = [item for item in main if item['matches_only_inherited']]
            for item in selected:
                require(item['candidate_id'] in authenticated, 'Authenticate inherited-only view')
                selected_rows.append(dict(qa_id=row['qa_id'], candidate_id=item['candidate_id'],
                    primary_contained_elsewhere=item['complete_primary_visible_in_other_main'],
                    primary_contained_in_authenticated_owners=bool(item['primary_source_coverage']) and all(
                        authenticated.intersection(source['owners']) for source in item['primary_source_coverage']),
                    bridge_ids=item['bridge_ids'], relation_path_ids=item['relation_path_ids'],
                    protected=item['protected']))
            if selected and row['missing_fine_to_main_turns']:
                overlaps.append(dict(qa_id=row['qa_id'],
                    missing_turns=row['missing_fine_to_main_turns'],
                    candidates=[{key: c[key] for key in ('candidate_id', 'actual_prompt_bytes_removed',
                        'complete_primary_visible_in_other_main')} for c in selected],
                    reference_linked_missing=[c for c in row['missing_fine']
                        if c['candidate_id'] in row['reference_linked_missing_ids']]))
    same(seen, set(by_search), 'Complete audit')
    report = dict(status='passed', cohort=cohort, questions=len(seen),
        original_summary=inventory['summary'], database_sha256=expected_db,
        unauthenticated_views=mismatches, inherited_only=selected_rows,
        gap_overlap=overlaps, provider_calls=0, production_changed=False,
        limitations=__doc__,
        input_sha256={str(path.resolve()): sha(path) for path in
            (args.inventory, previous, Path(__file__), frozen / 'app/answer_context.py')})
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(status='passed', questions=len(seen),
        unauthenticated_views=len(mismatches), gap_overlap_questions=len(overlaps))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
