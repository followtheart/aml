"""Exact frozen reader inventory of new advice-scope matches; never calls provider."""
import argparse,asyncio,copy,hashlib,json,sys
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import answer_choice as ac,budget,config,eval_scoring,llm
from probe_question_source_filter import load_cohort,sha
from replay_query_context import recorded_response
from indefinite_advice_scope_v2 import supports

async def main(a):
    rows=[];original=ac.choice_premises.is_suggestion
    for job in json.loads(a.cohorts.read_text()):
        for r,q,packet,s,c in load_cohort(Path(job['run']),Path(job['prepared'])):
            pending=copy.deepcopy(r['answer_calls']);mismatches=[];matched=[]
            def rule(claim,option):
                result=original(claim,option)
                if not result and supports(claim,option):
                    row=dict(claim=claim,option=option)
                    if row not in matched:matched.append(row)
                return result
            async def frozen(prompt,*pos,**kw):
                old=pending.pop(0) if pending else None
                if not old or old['stage']!=kw['stage'] or old['prompt_sha256']!=sha(prompt):mismatches.append(kw['stage']);raise ValueError('Provider forbidden')
                budget.current.get().before_call();return recorded_response(old)
            with patch.multiple(config,**r['versions']['settings']),patch.object(ac.choice_premises,'is_suggestion',rule),patch.object(llm,'complete_json',frozen),budget.scope(seconds=300,calls=40,tokens=200000):
                pred,score,_=await eval_scoring.evaluate(q,copy.deepcopy(packet))
            if pending or mismatches or (pred,score)!=(r['prediction'],r['score']):raise ValueError('Frozen mismatch '+r['qa_id'])
            rows.append(dict(cohort=job['name'],qa_id=r['qa_id'],matched=matched))
    result=dict(questions=len(rows),matches=[r for r in rows if r['matched']],all_baselines_exact=True,provider_calls=0,rule_sha256=hashlib.sha256(Path(__file__).with_name('indefinite_advice_scope_v2.py').read_bytes()).hexdigest(),rows=rows,caveat='Candidate is observed but not applied. Zero matches imply no scope intervention on these archived trajectories, not general semantic accuracy.')
    with a.output.open('x') as f:f.write(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['rows','caveat']}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cohorts',type=Path,required=True);p.add_argument('--output',type=Path,required=True);asyncio.run(main(p.parse_args()))
