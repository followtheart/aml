"""Offline counterfactual: skip unfocused lexical-only selection reviews.

No provider calls or runtime changes. Earlier selection is archived, so a skip
has an exact recorded counterfactual; later behavior under different budgets
is not measured. Gold labels are used only to score that saved selection.
"""
import argparse
import json
from pathlib import Path
import sys
from collections import Counter

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import eval_scoring


def skip_review(diagnostics):
    focus=diagnostics.get('answer_context_focus',{})
    review=diagnostics.get('answer_selection_review',{})
    focused={s['source_id'] for s in focus.get('sources',[])}
    candidates=set(review.get('context_options',[]))
    evidence=review.get('context_evidence',[])
    return bool(focus.get('status')=='ok' and focused and review.get('status')=='reviewed'
        and candidates and evidence and not review.get('personal_options')
        and not any(e.get('letter') in candidates and e.get('source_id') in focused for e in evidence))


def main(args):
    questions={}
    for line in args.prepared.read_text().splitlines():
        data=json.loads(line)
        for q in data['qa']:
            if q['id'] in questions:raise ValueError('Ambiguous global QA identity')
            questions[q['id']]=q
    cohorts=[]
    for path in args.results:
        rows=[json.loads(l) for l in path.read_text().splitlines(keepends=True) if l.endswith('\n')]
        changed=[];count=0;original=counterfactual=0;focus_status=Counter()
        for index,r in enumerate(rows):
            d=r.get('diagnostics',r);focus_status[d.get('answer_context_focus',{}).get('status','not_requested')]+=1
            original+=r['score'];score=r['score']
            if skip_review(d):
                first=d['answer_selection_review']['first'];q=questions[r['qa_id']]
                if first not in d['answer_eligible_options']:raise ValueError('Archived first answer was ineligible')
                score=eval_scoring.choice_score(first,q['gold_labels'],q['qa_type']);count+=1
                changed.append(dict(row=index,qa_id=r['qa_id'],enabled=r.get('enabled'),
                    old_prediction=r['prediction'],new_prediction=first,old_score=r['score'],new_score=score,
                    focused=d['answer_context_focus']['sources'],review=d['answer_selection_review']))
            counterfactual+=score
        cohorts.append(dict(path=str(path),rows=len(rows),original=original,counterfactual=counterfactual,
            affected=count,wins=[r['qa_id'] for r in changed if r['new_score']>r['old_score']],
            losses=[r['qa_id'] for r in changed if r['new_score']<r['old_score']],focus_status=dict(focus_status),cases=changed))
    report=dict(cohorts=cohorts,caveat='Counterfactual uses saved initial selection and exact diagnostic trigger; no new model or HTTPS evaluation. Input cohorts may overlap and must not be pooled as independent cases.')
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps([{k:v for k,v in c.items() if k not in ('cases','focus_status')} for c in cohorts]))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,nargs='+',required=True)
    p.add_argument('--prepared',type=Path,default=Path('data/prepared/personamem-v2-32k.jsonl'))
    p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
