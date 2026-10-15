"""Offline audit of incumbent evidence omitted by final-selection review."""
import argparse
import json
from pathlib import Path

from summarize_reader_replay import read_complete


def main(args):
    questions = {(d['conversation_id'], q['id']): q
                 for d in read_complete(args.prepared) for q in d['qa']}
    reports = []
    for path in args.results:
        rows = read_complete(path, args.allow_partial)
        reviewed = []
        for r in rows:
            q = questions[r['conversation_id'], r['qa_id']]
            d = r.get('diagnostics', r)
            review = d.get('answer_selection_review', {})
            if review.get('status') != 'reviewed':
                continue
            first, second = review['first'], review['second']
            if r.get('prediction', r.get('predicted_answer')) != second:
                raise ValueError('Review answer differs from recorded prediction')
            gold = q['gold_labels']
            hints = review.get('context_evidence', [])
            reviewed.append(dict(qa_id=r['qa_id'], enabled=r.get('enabled'),
                first=first, second=second, first_correct=first in gold, second_correct=second in gold,
                changed=first != second, incumbent_has_hint=any(h['letter'] == first for h in hints),
                hint_letters=sorted({h['letter'] for h in hints}),
                personal_options=review.get('personal_options', []),
                context_options=review.get('context_options', [])))
        reports.append(dict(path=str(path), rows=len(rows), reviewed=len(reviewed),
            improved=sum(not r['first_correct'] and r['second_correct'] for r in reviewed),
            regressed=sum(r['first_correct'] and not r['second_correct'] for r in reviewed),
            incumbent_hint_absent=sum(not r['incumbent_has_hint'] for r in reviewed), cases=reviewed))
    result = dict(cohorts=reports, caveat='Offline observed review transitions; gold labels used only for audit. '
        'Omitted hint does not imply absent source: full original context remains in review prompt. '
        'No causal claim that adding incumbent hints will improve accuracy; requires frozen-prefix paired intervention.')
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps([{k:v for k,v in r.items() if k != 'cases'} for r in reports]))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', type=Path, nargs='+', required=True)
    p.add_argument('--prepared', type=Path, default=Path('data/prepared/personamem-v2-32k.jsonl'))
    p.add_argument('--allow-partial', action='store_true')
    p.add_argument('--output', type=Path, required=True)
    main(p.parse_args())
