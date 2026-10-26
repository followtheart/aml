"""Compare whole-card transport against endpoint and previous direct-fine arm.

Literal containment is a transport check, not a semantic adoption review.
Read one cohort at a time in binary blocks; write detailed differences for review.
"""
import argparse
import gc
import json
from pathlib import Path

from inventory_support_carriers import sha
from audit_retained_porter_https import require, same


def read_json(path):
    data = bytearray()
    with path.open('rb') as stream:
        while block := stream.read(4 * 1024 * 1024):
            data.extend(block)
    return json.loads(data)


def identity(card):
    return (card.get('request_id'), card.get('message_index'), card.get('role'),
            card.get('timestamp'), card['id'][:1], card.get('declared'),
            card.get('legacy_rule', False))


def missing(before, after):
    indexed = {}
    for card in after:
        indexed.setdefault(identity(card), []).append(card['text'])
    return [card for card in before if not any(card['text'] in text
            for text in indexed.get(identity(card), []))]


def run(args):
    report = dict(status='complete_transport_audit', cohorts={}, provider_calls=0,
                  semantic_review_complete=False, runtime_integrated=False, https_evaluated=False)
    inputs = [Path(__file__), args.run / 'plan.json', args.previous / 'plan.json']
    for cohort, count in (('regression', 56), ('validation', 61)):
        path, previous_path = args.run / (cohort + '.json'), args.previous / (cohort + '.json')
        current, previous = read_json(path), read_json(previous_path)
        same(len(current['rows']), count, 'Complete new cohort')
        same(len(previous['rows']), count, 'Complete previous cohort')
        old = {row['qa_id']: row for row in previous['rows']}
        same(set(old), {row['qa_id'] for row in current['rows']}, 'Same questions')
        details = []
        for row in current['rows']:
            baseline = old[row['qa_id']]
            for key in ('query', 'options', 'original_packet', 'original_catalog'):
                same(row[key], baseline[key], 'Unchanged baseline ' + key)
            proposal, manifest = row['proposal'], row['candidate_manifest']
            same(proposal['priority']['original_fine_order'],
                 [mid for mid in baseline['proposal']['selected_ids']
                  if mid not in baseline['proposal']['rule_ids']], 'Original fine order')
            same(set(proposal['selected_ids']), set(baseline['proposal']['selected_ids']), 'Same whole fine units')
            same(proposal['rule_ids'], baseline['proposal']['rule_ids'], 'Same capped rules')
            same(manifest['token_budget'], 32000, 'Original final budget')
            require(manifest['token_upper_bound'] <= 32000 and len(row['candidate_packet']) <= 50,
                    'Final capacity')
            entry = dict(qa_id=row['qa_id'], query=row['query'], options=row['options'],
                         priority=proposal['priority'],
                         missing_expected_constraints=proposal['missing_expected_constraint_hashes'],
                         changed_from_direct=row['candidate_packet'] != baseline['candidate_packet'],
                         candidate_catalog_bytes=row['candidate_catalog_bytes'], comparisons={})
            for name, before, coverage in (
                    ('endpoint', row['original_catalog'], row['coverage']),
                    ('direct', baseline['candidate_catalog'], baseline['coverage'])):
                lost = missing(before, row['candidate_catalog'])
                added = missing(row['candidate_catalog'], before)
                prefix = 'original' if name == 'endpoint' else 'candidate'
                changes = {}
                for view in ('packet', 'catalog'):
                    pairs = list(enumerate(zip(coverage[prefix + '_' + view],
                        row['coverage']['candidate_' + view], strict=True)))
                    changes[view + '_gains'] = [i for i, (a, b) in pairs if b and not a]
                    changes[view + '_losses'] = [i for i, (a, b) in pairs if a and not b]
                entry['comparisons'][name] = dict(missing_cards=lost, added_or_extended_cards=added,
                    missing_constraints=sum(c['id'].startswith('r') for c in lost), **changes)
            details.append(entry)
        summary = dict(questions=count, source_summary=current['summary'],
            changed_from_direct=sum(row['changed_from_direct'] for row in details),
            prioritized_questions=sum(bool(row['priority']['prioritized_ids']) for row in details),
            missing_expected_constraints=sum(len(row['missing_expected_constraints']) for row in details),
            max_final_bytes=max(row['candidate_manifest']['token_upper_bound'] for row in current['rows']),
            comparisons={})
        for name in ('endpoint', 'direct'):
            rows = [row['comparisons'][name] for row in details]
            summary['comparisons'][name] = dict(
                missing_cards=sum(len(row['missing_cards']) for row in rows),
                missing_card_questions=sum(bool(row['missing_cards']) for row in rows),
                missing_constraints=sum(row['missing_constraints'] for row in rows),
                added_or_extended_cards=sum(len(row['added_or_extended_cards']) for row in rows),
                **{key: sum(len(row[key]) for row in rows) for key in
                   ('packet_gains', 'packet_losses', 'catalog_gains', 'catalog_losses')})
        with (args.run / (cohort + '-containment.json')).open('x') as stream:
            json.dump(dict(summary=summary, rows=details, limitation=__doc__), stream, indent=2)
            stream.write('\n')
        report['cohorts'][cohort] = summary
        inputs.extend((path, previous_path))
        print(json.dumps(dict(cohort=cohort, summary=summary)), flush=True)
        del current, previous, old, details, baseline, row, before, rows
        gc.collect()
    report['input_sha256'] = {str(path.resolve()): sha(path) for path in inputs}
    with (args.run / 'transport-audit.json').open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--previous', type=Path, required=True)
    run(parser.parse_args())
