"""Separate full-reference delivery, explicit quotation and observed answer gates.

Offline only. Reference annotations are diagnostic probes, not exhaustive useful
evidence or entailment labels. Quoting a reference source does not establish that
the quoted passage supports the selected answer. Observed gates are not causes.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

import audit_https_funnel as funnel
from audit_residual_reader import citations
from evaluation_model_guard import check_versions


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(args):
    root = args.run.resolve()
    plan = json.loads((root / 'plan.json').read_text())
    if (root / 'exit-code').read_text().strip() != '0':
        raise ValueError('Require a completed successful trial')
    prepared = {r['conversation_id']: r for r in funnel.records(args.prepared)}
    csv.field_size_limit(32 * 1024 * 1024)
    with args.gold_csv.open() as stream:
        gold = list(csv.DictReader(stream))
    arms, hashes = {}, {}
    for arm in ('control', 'treatment'):
        run = root / arm
        paths = {name: run / path for name, path in dict(
            results='completed/data/results/personamem-v2-32k.jsonl',
            memory='server-memory-debug.jsonl', search='server-search-debug.jsonl').items()}
        hashes[arm] = {name: sha(path) for name, path in paths.items()}
        records = {name: list(funnel.records(path)) for name, path in paths.items()}
        results = records['results']
        if (len(results) != plan['questions'] or len(records['search']) != len(results)
                or len({r['qa_id'] for r in results}) != len(results)
                or any(r.get('error_stage') or r.get('fake') or r['transport'] != 'https' for r in results)):
            raise ValueError('Incomplete or invalid cohort')
        by_question = {r['qa_id']: r for r in results}
        rows = funnel.analyze(results, records['memory'], records['search'], prepared, gold)
        stage = json.loads((run / 'funnel/stage-transport.json').read_text())
        if stage['summary']['progress_snapshot']:
            raise ValueError('Stage audit is incomplete')
        stages = {r['qa_id']: r for r in stage['rows']}
        output = []
        for row in rows:
            check_versions(row['server_versions'], arm)
            result = by_question[row['qa_id']]
            covered = row['catalog_turn_coverage'] + row['constraint_turn_coverage']
            delivered = any(turn['full'] for turn in covered)
            if delivered != (stages[row['qa_id']]['visible']['catalog']['full_turns'] > 0):
                raise ValueError('Independent reference transport audits disagree')
            reference_ids = {sid for turn in covered if turn['full'] for sid in turn['source_ids']}
            cards = {c['id']: c for c in result['answer_source_catalog'] + result['answer_constraint_catalog']}
            reference_cards = {sid: cards[sid] for sid in reference_ids}
            uses = [dict(stage=call['stage'], call_status=call.get('status'), **cite)
                    for call in result.get('answer_calls', [])
                    for cite in citations(call.get('response'), reference_cards) if cite['exact_quote']]
            output.append(dict(qa_id=row['qa_id'], score=row['score'], gold=row['gold'],
                prediction=row['prediction'], reference_available=stages[row['qa_id']]['raw_full_turns'] > 0,
                full_reference_delivered=delivered, reference_source_ids=sorted(reference_ids),
                exact_reference_source_quotes=uses, gold_eligible=row['gold'] in row['eligible'],
                gold_kind=row['gold_kind'], gold_valid_citations=row['gold_valid_citations'],
                gold_verified_personal_claims=row['gold_verified_personal_claims'],
                reader_gate=row['reader_gate'], reader_gate_detail=row['reader_gate_detail'],
                reference_transport=row['snippet_transport_stage']))
        wrong = [r for r in output if r['score'] < 1]
        groups = {}
        for name, selected in (
            ('wrong_reference_unavailable', [r for r in wrong if not r['reference_available']]),
            ('wrong_reference_missing', [r for r in wrong if r['reference_available'] and not r['full_reference_delivered']]),
            ('wrong_reference_delivered', [r for r in wrong if r['full_reference_delivered']]),
            ('wrong_reference_delivered_no_explicit_quote', [r for r in wrong if r['full_reference_delivered'] and not r['exact_reference_source_quotes']]),
            ('wrong_reference_delivered_with_explicit_quote', [r for r in wrong if r['full_reference_delivered'] and r['exact_reference_source_quotes']]),
            ('wrong_gold_eligible', [r for r in wrong if r['gold_eligible']])):
            groups[name] = sorted((r['qa_id'] for r in selected), key=int)
        arms[arm] = dict(summary=dict(questions=len(output), score=sum(r['score'] for r in output),
            wrong_reader_gates=dict(Counter(r['reader_gate_detail'] for r in wrong)), groups=groups), rows=output)
    if {r['qa_id'] for r in arms['control']['rows']} != {r['qa_id'] for r in arms['treatment']['rows']}:
        raise ValueError('Arm cohorts differ')
    artifact = dict(arms=arms, input_hashes=hashes, prepared_sha256=sha(args.prepared),
        gold_sha256=sha(args.gold_csv), limitation=__doc__,
        script_hashes={p.name: sha(p) for p in (Path(__file__), Path(funnel.__file__),
            Path(__file__).with_name('audit_residual_reader.py'))})
    with (root / 'reader-transport-audit.json').open('x') as stream:
        json.dump(artifact, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({arm: value['summary'] for arm, value in arms.items()}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--prepared', type=Path, default=Path('data/prepared/personamem-v2-32k.jsonl'))
    parser.add_argument('--gold-csv', type=Path, default=Path('data/raw/personamem-v2.csv'))
    main(parser.parse_args())
