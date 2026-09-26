"""Replay archived support proposals and compare first vs reviewed entailment.

No model calls. Candidate-set changes are not new end-to-end scores.
"""
import argparse
import asyncio
import copy
import json
from pathlib import Path
import sys
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import answer_choice as ac, config, llm


async def main(args):
    config.MEMORY_DEBUG_LOG=config.SEARCH_DEBUG_LOG=''
    qa={(c['conversation_id'],q['id']):q for c in map(json.loads,args.prepared.read_text().splitlines()) for q in c['qa']}
    output=[]
    for file in args.results:
        for r in map(json.loads,file.read_text().splitlines()):
            q=qa[r['conversation_id'],r['qa_id']]
            sources={s['id']:s for s in r['answer_source_catalog']}
            recorded=[c for c in r['answer_calls'] if c['stage'].startswith('eval.choice_support')]
            pending=list(recorded)
            async def replay(*a,**kw):
                assert pending, 'unexpected support call'
                call=pending.pop(0)
                assert call['stage']==kw['stage'], (call['stage'],kw['stage'])
                if 'response' not in call: raise ValueError('Recorded provider parse failure')
                return copy.deepcopy(call['response'])
            with patch.object(llm,'complete_json',replay):
                entries=await ac.assess_support('Offline replay',ac.Assessments.model_json_schema(),q['options'],sources,{})
            assert not pending, 'support replay call count mismatch'
            checks=ac.entailment_checks(entries,sources,q['options'])
            first=[c['response'] for c in r['answer_calls'] if c['stage'] in ('eval.choice_entailment','eval.choice_entailment.repair') and c.get('status')=='ok']
            if not first:
                output.append(dict(file=str(file),qa_id=r['qa_id'],skipped='no_valid_first_verdict'));continue
            verdicts=first[-1]
            reviewed=copy.deepcopy(verdicts)
            restored=r.get('answer_entailment_review',{}).get('restored_check_ids',[])
            for v in reviewed['checks']:
                if v['claim_id'] in restored:v['entailed']=True
            baseline=ac.validate_entailments(reviewed,copy.deepcopy(entries),checks)
            assert [(e['letter'],e['status']) for e in baseline]==[(e['letter'],e['status']) for e in r['choice_alignment']], (file,r['qa_id'],'status mismatch')
            candidate=ac.validate_entailments(verdicts,copy.deepcopy(entries),checks)
            blocked=set(r['answer_blocked_options'])
            def eligible(es):
                selected=ac.eligible_choices(es,blocked)
                # v8 excluded generic if any supported/partial option survived.
                if 'v8-' in r['answer_policy']:
                    personal=[e['letter'] for e in es if e['letter'] in selected and e['status'] in ('supported','partial')]
                    return personal or selected
                return selected
            before=eligible(baseline);after=eligible(candidate)
            assert before==r['answer_eligible_options'], (file,r['qa_id'],'eligible mismatch',before,r['answer_eligible_options'])
            output.append(dict(file=str(file),qa_id=r['qa_id'],gold=q['gold_labels'],prediction=r['prediction'],score=r['score'],
                restored=restored,before=before,without_review=after,
                changed=before!=after,gold_was_eligible=bool(set(q['gold_labels'])&set(before)),gold_still_eligible=bool(set(q['gold_labels'])&set(after)),
                new_unique_choice=after[0] if len(after)==1 and before!=after else None))
    summary=dict(rows=len(output),reconstructed=sum('skipped' not in r for r in output),
                 changed=[r for r in output if r.get('changed')],
                 previously_correct_gold_lost=[r for r in output if r.get('score') and not r['gold_still_eligible']],provider_calls=0)
    args.output.write_text(json.dumps(dict(summary=summary,rows=output),ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False))

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--results',nargs='+',type=Path,required=True)
    ap.add_argument('--prepared',type=Path,default=Path('data/prepared/personamem-v2-32k.jsonl'))
    ap.add_argument('--output',type=Path,required=True)
    asyncio.run(main(ap.parse_args()))
