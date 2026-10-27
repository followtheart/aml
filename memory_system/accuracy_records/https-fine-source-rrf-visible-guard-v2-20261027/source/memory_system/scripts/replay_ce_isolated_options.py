"""Full paired live downstream replay of audited isolated-option CE ranking."""
import argparse,asyncio,hashlib,json,sys
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import budget,config,eval_scoring
from audit_ce_isolated_options import audit
from probe_ce_source_units import records
from probe_question_source_filter import load_cohort
from replay_https_ranking_budget import run_ranking

async def main(a):
    checked=audit(a.probe);plan=json.loads(a.probe.with_suffix('.plan.json').read_text());run=Path(plan['run']);prepared=Path(plan['prepared']);cohort=load_cohort(run,prepared)
    traces={s['search_id']:s for s in records(run/'server-search-debug.jsonl')};probes={p['qa_id']:p for p in records(a.probe)}
    # Complete baseline packet/admission reproduction before ANY live provider call.
    for r,q,packet,s,c in cohort:
        trace=traces[r['search_id']]
        with patch.multiple(config,**trace['versions']['settings']):old=await run_ranking(trace,trace['versions']['settings']['RERANK_MAX_PROMPT_BYTES'],frozen=True)
        assert old['packet_hash']==r['packet_hash']==trace['packet_hash']
        assert old['cascade']['listwise']['candidate_ids']==trace['cascade']['listwise']['candidate_ids']
    receipt={'run':str(run),'prepared':str(prepared),'probe':str(a.probe),'probe_sha256':hashlib.sha256(a.probe.read_bytes()).hexdigest(),'qa_ids':[r['qa_id'] for r,*_ in cohort],'frozen_packets_exact':len(cohort),'upstream_audit':{k:v for k,v in checked.items() if k not in ['rows','caveat']},'caveat':'Original retrieval fixed. Both arms use fresh listwise/recovery/full reader, alternating order. CE scores precomputed; scoring latency/cost omitted from downstream budget. Candidate uses ordinal values solely to implement audited round-robin order in graph cascade; no calibrated probability claim. Not fresh HTTPS performance.'}
    with a.output.with_suffix('.plan.json').open('x') as f:json.dump(receipt,f,indent=2)
    rows=[]
    with a.output.open('x') as out:
        for index,(r,q,packet,s,c) in enumerate(cohort):
            trace=traces[r['search_id']];p=probes[r['qa_id']];ids=p['candidate_ids'];scores={'control':dict(zip(ids,p['arms'][0]['scores'])),'candidate':{cid:len(ids)-p['aggregate_order'].index(cid) for cid in ids}}
            for arm in (['control','candidate'] if index%2==0 else ['candidate','control']):
                with patch.multiple(config,**trace['versions']['settings']):
                    ranking=await run_ranking(trace,config.RERANK_MAX_PROMPT_BYTES,ce_scores=scores[arm])
                    with budget.scope(seconds=300,calls=40,tokens=200000):pred,score,diag=await eval_scoring.evaluate(q,ranking['packet'])
                row={'qa_id':r['qa_id'],'arm':arm,'prediction':pred,'score':score,'ranking':ranking,'diagnostics':diag};rows.append(row);out.write(json.dumps(row,ensure_ascii=False)+'\n');out.flush();print(json.dumps({k:row[k] for k in ['qa_id','arm','prediction','score']}),flush=True)
    summary={'questions':len(cohort),'scores':{arm:sum(r['score'] for r in rows if r['arm']==arm) for arm in ['control','candidate']},'caveat':receipt['caveat']};a.output.with_suffix('.summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--probe',type=Path,required=True);p.add_argument('--output',type=Path,required=True);asyncio.run(main(p.parse_args()))
