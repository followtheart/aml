"""Compare matched HTTPS questions without conflating replay and end-to-end scores."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re


def load_run(directory, allow_partial=False):
    terminal = directory / 'exit-code'
    exit_code = int(terminal.read_text().strip()) if terminal.exists() else None
    if exit_code is None and not allow_partial:
        raise ValueError(f'Run is still active: {directory}')
    path = directory / 'completed/data/results/personamem-v2-32k.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows or any(r.get('transport') != 'https' or r.get('fake') for r in rows):
        raise ValueError('Expected nonempty, non-fake HTTPS results')
    if len({r['run_id'] for r in rows}) != 1:
        raise ValueError('Mixed experiment IDs')
    indexed = {(r['dataset'], r['conversation_id'], r['qa_id']): r for r in rows}
    if len(indexed) != len(rows):
        raise ValueError('Duplicate question identity')
    planned = re.search(r'^protocol=.*\bqa=(\d+)', (directory / 'output.log').read_text(), re.M)
    expected = int(planned[1]) if planned else None
    complete = exit_code is not None and expected == len(rows)
    if not complete and not allow_partial:
        raise ValueError('Scored questions do not cover the complete planned evaluation')
    audit_path = directory / 'funnel/https-summary.json'
    audit = json.loads(audit_path.read_text()) if audit_path.exists() else {}
    if audit.get('run_id') not in (None, rows[0]['run_id']):
        raise ValueError('Funnel audit belongs to a different experiment')
    # A live result file can advance after its last funnel snapshot. Do not
    # combine current scores with stage counts from an older prefix.
    audit_current = (audit.get('run_id') == rows[0]['run_id']
                     and audit.get('count') == len(rows)
                     and audit.get('score') == sum(r['score'] for r in rows))
    return indexed, dict(directory=str(directory), run_id=rows[0]['run_id'],
                         exit_code=exit_code, expected_qa=expected, observed_qa=len(rows),
                         complete=complete, score=sum(r['score'] for r in rows),
                         recorded_answer_calls=sum(len(r.get('answer_calls', [])) for r in rows),
                         support_repair_calls=sum(c.get('stage', '').startswith('eval.choice_support.repair')
                                                  for r in rows for c in r.get('answer_calls', [])),
                         unresolved_support_rows=sum(bool(r.get('answer_support_validation', {}).get('unresolved_options'))
                                                     for r in rows),
                         errors=dict(Counter(r['error_stage'] for r in rows if r.get('error_stage'))),
                         prepared_sha256=audit.get('prepared_sha256'),
                         audit_current=audit_current,
                         audit_observed_qa=audit.get('count'),
                         wrong_stages=audit.get('wrong_stages') if audit_current else None)


def source_catalog(row):
    # Source IDs and request IDs change between ingestion runs. Preserve actual
    # text, authority and order: their changes can influence the answer.
    return [(s.get('declared') or s['role'], s['text']) for s in row.get('answer_source_catalog', [])]


def compare(control, treatment, allow_partial=False):
    left, a = load_run(control, allow_partial)
    right, b = load_run(treatment, allow_partial)
    if not allow_partial and set(left) != set(right):
        raise ValueError('Runs evaluate different question sets')
    if a['prepared_sha256'] and b['prepared_sha256'] and a['prepared_sha256'] != b['prepared_sha256']:
        raise ValueError('Prepared datasets differ')
    pairs = []
    for key in sorted(set(left) & set(right)):
        x, y = left[key], right[key]
        if x['protocol'] != y['protocol']:
            raise ValueError('Evaluation protocols differ')
        pairs.append(dict(dataset=key[0], conversation_id=key[1], qa_id=key[2],
                          control_score=x['score'], treatment_score=y['score'],
                          control_prediction=x['prediction'], treatment_prediction=y['prediction'],
                          source_catalog_same=source_catalog(x) == source_catalog(y),
                          eligibility_same=x.get('answer_eligible_options') == y.get('answer_eligible_options'),
                          control_unresolved_support=x.get('answer_support_validation', {}).get('unresolved_options', []),
                          treatment_unresolved_support=y.get('answer_support_validation', {}).get('unresolved_options', []),
                          control_recorded_calls=len(x.get('answer_calls', [])),
                          treatment_recorded_calls=len(y.get('answer_calls', [])),
                          control_error=x.get('error_stage'), treatment_error=y.get('error_stage')))
    if not pairs:
        raise ValueError('No shared scored questions')
    return dict(control=a, treatment=b,
                partial=not (a['complete'] and b['complete'] and set(left) == set(right)),
                matched_questions=len(pairs),
                matched_control_score=sum(p['control_score'] for p in pairs),
                matched_treatment_score=sum(p['treatment_score'] for p in pairs),
                wins=sum(p['treatment_score'] > p['control_score'] for p in pairs),
                losses=sum(p['treatment_score'] < p['control_score'] for p in pairs),
                changed_source_catalogs=sum(not p['source_catalog_same'] for p in pairs),
                pairs=pairs,
                caveat='Fresh ingestion and retrieval can change evidence. Use fixed-packet replays '
                       'to isolate downstream policy effects; repeated runs are not independent questions.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control', type=Path, required=True)
    parser.add_argument('--treatment', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args()
    report = compare(args.control, args.treatment, args.allow_partial)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'pairs'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
