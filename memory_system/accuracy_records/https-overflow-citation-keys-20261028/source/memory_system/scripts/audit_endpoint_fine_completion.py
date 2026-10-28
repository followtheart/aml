"""Audit endpoint-prefix preservation, whole-unit addition, and frozen funnel.

All references are post-selection measurements. This is not an answer test or
a full semantic relevance review of added context.
"""
import argparse
import gc
import json
from pathlib import Path

from audit_fine_direct_lifecycle import read_json, missing
from audit_retained_porter_https import require, same
from inventory_support_carriers import sha
from inventory_visible_source_redundancy import records
from endpoint_fine_completion import without_digest


def run(args):
    result = dict(status='passed', cohorts={}, provider_calls=0, semantic_review_complete=False,
                  https_evaluated=False)
    inputs = [Path(__file__), args.run / 'plan.json']
    for cohort, count in (('regression', 56), ('validation', 61)):
        path = args.run / (cohort + '.json')
        frozen = args.source / ('https-retained-porter-20261025-' + cohort) / 'control'
        data = read_json(path)
        rows = {r['qa_id']: r for r in data['rows']}
        same(len(rows), count, 'Complete cohort')
        identities = {}
        for r in records(frozen / 'completed/data/results/personamem-v2-32k.jsonl'):
            identities[r['search_id']] = str(r['qa_id'])
        seen = set()
        for trace in records(frozen / 'server-search-debug.jsonl'):
            qid = identities[trace['search_id']]
            require(qid not in seen, 'Unique trace')
            seen.add(qid)
            row = rows[qid]
            original, candidate = row['original_packet'], row['candidate_packet']
            same(original, trace['returned'], 'Exact original endpoint')
            same([without_digest(item) for item in original],
                 [without_digest(item) for item in candidate[:len(original)]], 'Exact prefix except digest')
            require(not missing(row['original_catalog'], row['candidate_catalog']), 'All original cards preserved')
            additions = candidate[len(original):]
            same([item['id'] for item in additions], row['proposal']['appended_ids'], 'Appended IDs')
            pool = {r['id']: r for r in trace['ranked']}
            for item in additions:
                source = pool[item['id']]['_packet_item']
                for key in ('content', 'sources', 'memory_type', 'personal_evidence', 'is_constraint'):
                    same(item.get(key, False if key == 'is_constraint' else None),
                         source.get(key, False if key == 'is_constraint' else None), 'Whole-unit field ' + key)
                same(item['score'], pool[item['id']]['_ce_score'], 'Real CE score')
                same(item['score_kind'], 'cross_encoder', 'No promoted judgment score')
            total = sum(len(item['content'].encode()) for item in candidate) + max(0, len(candidate)-1)
            same(total, row['candidate_manifest']['token_upper_bound'], 'Actual byte count')
            require(total <= trace['evidence_token_budget'] and len(candidate) <= trace['top_k'], 'Original caps')
            kept = {item['id'] for item in candidate}
            added = {item['id'] for item in additions}
            for group in row['proposal']['groups']:
                require(not (set(group) & added) or set(group) <= kept, 'Added group complete')
        same(seen, set(rows), 'All traces audited')
        stage_path = frozen / 'funnel/stage-transport.json'
        stages = {str(r['qa_id']): r for r in read_json(stage_path)['rows']}
        details, funnel = [], {}
        names = ('fused', 'coarse', 'fine', 'listwise_input', 'listwise_selected',
                 'overflow_selected', 'packet', 'catalog', 'completion_catalog')
        for name in names:
            masks = [row['coverage']['candidate_catalog'] if name == 'completion_catalog' else
                     [c['full'] for c in stages[qid]['visible'][name]['turn_coverage']]
                     for qid, row in rows.items()]
            funnel[name] = dict(reference_questions=sum(any(mask) for mask in masks),
                                reference_turns=sum(sum(mask) for mask in masks))
        for qid, row in rows.items():
            new = missing(row['candidate_catalog'], row['original_catalog'])
            fine_mask = [c['full'] for c in stages[qid]['visible']['fine']['turn_coverage']]
            final_mask = row['coverage']['candidate_catalog']
            details.append(dict(qa_id=qid, appended_ids=row['proposal']['appended_ids'],
                added_bytes=row['proposal']['added_bytes'], new_or_extended_cards=new,
                fine_to_final_missing=[i for i, (a,b) in enumerate(zip(fine_mask,final_mask,strict=True)) if a and not b]))
        summary = dict(questions=count, exact_endpoint_prefixes=count, full_original_catalogs_preserved=count,
            source_summary=data['summary'], funnel=funnel,
            changed_questions=sum(bool(r['appended_ids']) for r in details),
            additions=sum(len(r['appended_ids']) for r in details),
            new_or_extended_cards=sum(len(r['new_or_extended_cards']) for r in details),
            fine_to_final_gap_questions=sum(bool(r['fine_to_final_missing']) for r in details),
            fine_to_final_gap_turns=sum(len(r['fine_to_final_missing']) for r in details),
            max_final_bytes=max(r['candidate_manifest']['token_upper_bound'] for r in rows.values()))
        with (args.run / (cohort + '-audit.json')).open('x') as stream:
            json.dump(dict(summary=summary, rows=details, limitation=__doc__), stream, indent=2)
            stream.write('\n')
        result['cohorts'][cohort] = summary
        inputs.extend((path, stage_path, frozen / 'server-search-debug.jsonl'))
        print(json.dumps(dict(cohort=cohort, summary=summary)), flush=True)
        del data, rows, stages, details, trace, pool, row
        gc.collect()
    result['input_sha256'] = {str(path.resolve()): sha(path) for path in inputs}
    with (args.run / 'audit.json').open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--source', type=Path, default=Path(__file__).resolve().parents[1] / 'runs')
    run(parser.parse_args())
