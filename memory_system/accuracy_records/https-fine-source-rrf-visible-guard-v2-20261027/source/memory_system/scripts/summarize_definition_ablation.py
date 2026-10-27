"""Matched, question-level ablation summaries, including first response labels."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.listwise_contract import State


def first_labels(row):
    replies = row.get('responses', [])
    if not replies or replies[0]['field'] is not None:
        return None
    state = State(len(row['candidate_ids']), keyed=True)
    try:
        state.accept(replies[0]['response'])
    except (ValueError, TypeError, KeyError):
        return None
    if state.ranking is None or state.irrelevant is None:
        return None
    return Counter(v['relevance'] for v in state.payload['candidates'].values())


def main(args):
    data = json.loads(args.input.read_text())
    by = {(r['qa_id'], r['protocol']): r for r in data['rows']}
    arms = ['control', 'definition', 'categories']
    common = sorted({qid for qid, _ in by if all(by[qid, a]['primary_status'] in ('ok','recovered') for a in arms)}, key=int)
    first_common = sorted({qid for qid, _ in by if all(first_labels(by[qid, a]) is not None for a in arms)}, key=int)
    first_counts = {a: dict(sum((first_labels(by[qid, a]) for qid in first_common), Counter())) for a in arms}
    comparisons = {}
    for left, right in [('control','definition'),('definition','categories'),('control','categories')]:
        rows = []
        for qid in common:
            a, b = by[qid, left], by[qid, right]
            rows.append(dict(qa_id=qid, irrelevant_change=len(b['irrelevant_ids'])-len(a['irrelevant_ids']),
                primary_full_turn_change=b['primary_selected_ids_full_turns']-a['primary_selected_ids_full_turns'],
                final_full_turn_change=b['selected_ids_full_turns']-a['selected_ids_full_turns'],
                added_negative_ids=sorted(set(b['irrelevant_ids'])-set(a['irrelevant_ids'])),
                removed_negative_ids=sorted(set(a['irrelevant_ids'])-set(b['irrelevant_ids']))))
        comparisons[left+' -> '+right] = dict(rows=rows,
            irrelevant_reduced_questions=sum(r['irrelevant_change']<0 for r in rows),
            irrelevant_increased_questions=sum(r['irrelevant_change']>0 for r in rows),
            coverage_improved_questions=sum(r['final_full_turn_change']>0 for r in rows),
            coverage_worsened_questions=sum(r['final_full_turn_change']<0 for r in rows))
    report = dict(summary=data['summary'], all_primary_valid=data['both_primary_valid'],
        all_primary_valid_questions=common, first_response_valid_candidate_questions=first_common,
        first_response_labels=first_counts, paired_changes=comparisons, violations=data['violations'],
        limitations=['First-response labels are counted only when all candidate records are valid, independently of group validity.',
            'Single stochastic trial per arm/question; labels are not ground-truth precision.',
            'The categories arm adds evidence-role and duplicate-handling rules together; their separate effects are not isolated.',
            'Literal source coverage does not establish answer correctness or semantic sufficiency.'])
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='paired_changes'},ensure_ascii=False,indent=2))
    for key,val in comparisons.items():
        print(key, {k:v for k,v in val.items() if k!='rows'})

if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
