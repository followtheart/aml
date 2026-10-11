"""Full-cohort preflight; paired fresh ranking/readers on changed fine selections."""
import argparse,asyncio,copy,hashlib,json,sys
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import budget,config,eval_scoring,llm,cascade_rerank as ranking,evidence_selection as es
from probe_question_source_filter import load_cohort,sha
from probe_visible_topic_reservations import compare
from probe_verifier_assessment import capture
from replay_https_ranking_budget import run_ranking
from replay_query_context import records
from visible_source_dedup import coverage_cache

async def main(a):
    cohort=load_cohort(a.run,a.prepared);traces={s['search_id']:s for s in records(a.run/'server-search-debug.jsonl')};comparisons={};provider=llm.complete_json
    await capture(cohort) # All archived readers and prompt hashes reproduced.
    for r,q,p,s,c in cohort:
        t=traces[r['search_id']];comparisons[r['qa_id']]=compare(t)
        with patch.multiple(config,**r['versions']['settings']):check=await run_ranking(t,t['versions']['settings']['RERANK_MAX_PROMPT_BYTES'],frozen=True)
        if check['packet_hash']!=r['packet_hash'] or check['cascade']['fine']['selected_ids']!=t['cascade']['fine']['selected_ids']:raise ValueError('Frozen ranking mismatch')
    affected=[qid for qid,x in comparisons.items() if x['original']['ids']!=x['visible']['ids']]
    plan=dict(run=str(a.run),prepared=str(a.prepared),prepared_sha256=hashlib.sha256(a.prepared.read_bytes()).hexdigest(),population=[r['qa_id'] for r,*_ in cohort],affected=affected,comparisons=comparisons,baseline_score=sum(r['score'] for r,*_ in cohort),caveat='Complete exact reader/ranking preflight; unchanged fine lists reuse archived result in both arms. Changed lists receive fresh original/candidate ranking and readers with frozen CE/coarse pool. Not fresh HTTPS or a source-sufficiency oracle.')
    a.output.with_suffix('.plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n');rows=[]
    for index,(r,q,p,s,c) in enumerate(cohort):
        for arm in (['original','visible'] if index%2==0 else ['visible','original']):
            row=dict(qa_id=r['qa_id'],arm=arm,live_calls=[],reused=r['qa_id'] not in affected)
            if row['reused']:row.update(prediction=r['prediction'],score=r['score'],packet_hash=r['packet_hash'])
            else:
                async def recorded(prompt,*pos,**kw):
                    call=dict(stage=kw['stage'],prompt=prompt,prompt_sha256=sha(prompt));row['live_calls'].append(call)
                    try:response=await provider(prompt,*pos,**kw);call.update(status='ok',response=response);return response
                    except Exception as exc:call.update(status='error',error_type=type(exc).__name__);raise
                t=traces[r['search_id']];original=ranking.shortlist;calls=0
                def shortlist(*args,**kw):
                    nonlocal calls
                    calls+=1
                    if arm=='visible' and calls==2:
                        expanded=coverage_cache(args[0]);champion=es.option_champions
                        with patch.object(es,'option_champions',lambda ordered,req,visible,limit:champion(ordered,req,expanded,limit)):chosen=original(*args,**kw)
                    else:chosen=original(*args,**kw)
                    if calls==1 and [x['id'] for x in chosen]!=t['cascade']['coarse']['selected_ids']:raise ValueError('Coarse changed')
                    if calls==2 and [x['id'] for x in chosen]!=comparisons[r['qa_id']][arm]['ids']:raise ValueError('Fine not expected')
                    return chosen
                with patch.multiple(config,**r['versions']['settings']),patch.object(llm,'complete_json',recorded),patch.object(ranking,'shortlist',shortlist):
                    ranked=await run_ranking(t,t['versions']['settings']['RERANK_MAX_PROMPT_BYTES'])
                    with budget.scope(seconds=300,calls=40,tokens=200000):prediction,score,diag=await eval_scoring.evaluate(q,copy.deepcopy(ranked['packet']))
                row.update(prediction=prediction,score=score,packet_hash=ranked['packet_hash'],ranking=ranked,diagnostics=diag)
            rows.append(row);print(json.dumps({k:row[k] for k in ['qa_id','arm','prediction','score','reused']}),flush=True)
    report=dict(plan=plan,rows=rows,scores={arm:sum(x['score'] for x in rows if x['arm']==arm) for arm in ['original','visible']},live_calls=sum(len(x['live_calls']) for x in rows))
    with a.output.open('x') as f:f.write(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['scores','live_calls']}))
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ['run','prepared','output']:p.add_argument('--'+n,type=Path,required=True)
    asyncio.run(main(p.parse_args()))
