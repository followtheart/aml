"""Explain a terminal one-seat main-swap preflight without changing its policy.

Inventory every missing-fine candidate's eligibility before joining reference
annotations. Recompute nominees/donors exactly. Reference-linked carriers are
transport diagnostics, not semantic relevance labels or new selector rules.
"""
import argparse
import json
from pathlib import Path
import sys
from unittest.mock import patch

from audit_retained_porter_https import records, require, same, sha


def main(args):
    prior = json.loads(args.replay.read_text())
    run = args.run.resolve()
    root = run.parent
    trace_path = run / 'server-search-debug.jsonl'
    stage_path = run / 'funnel/stage-transport.json'
    result_path = run / 'completed/data/results/personamem-v2-32k.jsonl'
    for path in (trace_path, stage_path, result_path):
        same(sha(path), prior['input_sha256'][str(path)], 'Replay input ' + str(path))
    plan = json.loads((root / 'plan.json').read_text())
    for name, digest in plan['source_files'].items():
        same(sha(root / 'source' / name), digest, 'Frozen source ' + name)
    sys.path.insert(0, str(root / 'source/memory_system'))
    from app import fine_evidence, config
    by_question = {r['qa_id']: r for r in prior['rows']}
    results = {r['search_id']: {k:r[k] for k in ('qa_id','conversation_id','packet_hash')}
               for r in records(result_path)}
    inventory, seen = [], set()
    with patch('socket.socket.connect', side_effect=AssertionError('Offline inventory')):
        for t in records(trace_path):
            sid = t['search_id']
            require(sid in results and sid not in seen, 'Unexpected/duplicate search')
            seen.add(sid)
            result = results[sid]
            old = by_question[result['qa_id']]
            same(t['packet_hash'], old['packet_hash'], 'Packet')
            with patch.multiple(config, **t['versions']['settings']):
                c = t['cascade']
                pool = {r['id']: r for r in t['ranked']}
                coarse = [pool[mid] for mid in c['coarse']['selected_ids']]
                fine = [pool[mid] for mid in c['fine']['selected_ids']]
                submitted = [pool[mid] for mid in c['listwise']['candidate_ids']]
                same([r['id'] for r in submitted], old['original_main_ids'], 'Original main')
                adaptive = c['fine']['selection']['adaptive']
                requirements = [{k: r[k] for k in ('id', 'text', 'origin') if k in r}
                                for r in t['coverage_manifest']['coverage']['requirements']]
                facets = requirements + adaptive.get('context_facets', [])
                visible = {r['id']: fine_evidence.selection.fragments(r) for r in coarse}
                sparse = fine_evidence.sparse_matches(coarse, facets, visible)
                matches = {}
                for row in coarse:
                    mid = row['id']
                    values = fine_evidence.relevance(row, facets, visible[mid])
                    for rid, value in sparse[mid].items():
                        values[rid] = max(values.get(rid, (False, False, 0)), value)
                    matches[mid] = values
                protected = set(c['listwise']['carried_ids'])
                protected.update(mid for row in fine for mid in fine_evidence.dependencies(row))
                protected.update(mid for group in adaptive.get('path_groups', []) for mid in group)
                same(sorted(protected), old['ordering']['protected_ids'], 'Protected IDs')
                donors = [row for row in submitted if row['id'] not in protected
                    and row.get('_ce_score') is not None and not row.get('_user_rule')
                    and not row.get('_packet_item', {}).get('is_constraint')
                    and not fine_evidence.dependencies(row) and matches[row['id']]
                    and not any(v[0] for v in matches[row['id']].values())
                    and all(key[1] in ('user', 'assistant') for key, _ in visible[row['id']])]
                donors.sort(key=fine_evidence.key)
                same([r['id'] for r in donors], old['ordering']['eligible_donors'], 'Donor reconstruction')
                main_ids = set(old['original_main_ids'])
                missing, nominees = [], []
                for row in sorted(fine, key=fine_evidence.key, reverse=True):
                    mid = row['id']
                    if mid in main_ids:
                        continue
                    reasons = []
                    if row.get('_ce_score') is None:
                        reasons.append('unscored')
                    if fine_evidence.dependencies(row):
                        reasons.append('has_dependencies')
                    if not any(value[0] for value in matches[mid].values()):
                        reasons.append('no_user_source_lexical_match')
                    if not reasons:
                        nominees.append(mid)
                    higher_than = [d['id'] for d in donors if row.get('_ce_score') is not None
                                   and row['_ce_score'] > d['_ce_score']]
                    missing.append(dict(candidate_id=mid, ce_score=row.get('_ce_score'),
                        full_pool_ce_rank=next((i+1 for i,x in enumerate(sorted(coarse,key=fine_evidence.key,reverse=True))
                                               if x['id']==mid), None),
                        nominee_rejection_reasons=reasons, source_match_roles=sorted({
                            'user' if value[0] else 'non_user' for value in matches[mid].values()}),
                        lower_scored_eligible_donors=higher_than,
                        tried_pairs=[a for a in old['ordering']['attempts'] if a['added_id']==mid],
                        rank_text_bytes=len(row.get('_rank_text',row['content']).encode()),
                        original_omissions=[o for o in c['omitted'] if o['id']==mid and o['stage']=='listwise']))
                same(nominees, old['ordering']['eligible_nominees'], 'Nominee reconstruction')
                inventory.append(dict(qa_id=result['qa_id'], missing_fine=missing,
                    eligible_donor_count=len(donors), original_main_bytes=old['original_bytes']))
    same(seen, set(results), 'Complete traces')
    same(len(inventory), prior['summary']['baseline_exact_questions'], 'Complete cohort')
    stages = {r['qa_id']: r for r in json.loads(stage_path.read_text())['rows']}
    gaps = []
    for row in inventory:
        stage = stages[row['qa_id']]
        coverage = stage['visible']
        missing_turns = [i for i,(a,b) in enumerate(zip(coverage['fine']['turn_coverage'],
            coverage['listwise_input']['turn_coverage'],strict=True)) if a['full'] and not b['full']]
        if missing_turns:
            linked = set(stage['carrier_ids']['fine']) - set(stage['carrier_ids']['listwise_input'])
            gaps.append(dict(qa_id=row['qa_id'], missing_full_user_turns=missing_turns,
                eligible_donor_count=row['eligible_donor_count'], original_main_bytes=row['original_main_bytes'],
                reference_linked_carriers=[r for r in row['missing_fine'] if r['candidate_id'] in linked]))
    report = dict(status='complete_main_swap_eligibility_inventory', questions=len(inventory),
        rows=inventory, reference_gaps=gaps, provider_calls=0, production_changed=False,
        limitation=__doc__, input_sha256={str(p):sha(p) for p in (
            Path(__file__), Path(__file__).with_name('audit_retained_porter_https.py'),
            args.replay, root/'plan.json', trace_path,stage_path,result_path)})
    with args.output.open('x') as stream:
        json.dump(report,stream,indent=2);stream.write('\n')
    print(json.dumps(dict(questions=len(inventory),reference_gaps=gaps)),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--replay',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args())
