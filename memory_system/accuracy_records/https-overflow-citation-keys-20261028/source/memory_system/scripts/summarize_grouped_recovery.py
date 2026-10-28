"""Audit complete grouped-recovery diagnostics; literal anchors are not entailment."""
import argparse,asyncio,hashlib,json
from pathlib import Path
from probe_grouped_recovery import transform,expand,ac,capture,load_cohort,sha

async def audit(path):
    plan=json.loads(path.with_suffix('.plan.json').read_text());prepared=Path(plan['prepared'])
    if hashlib.sha256(prepared.read_bytes()).hexdigest()!=plan['prepared_sha256']:raise ValueError('Input changed')
    if plan['arms']!=['original','claim_only','grouped'] or plan['repetitions']!=2:raise ValueError('Design changed')
    cohort=load_cohort(Path(plan['run']),prepared)
    if plan['full_population']!=[r['qa_id'] for r,*_ in cohort]:raise ValueError('Population changed')
    captured=await capture(cohort);cases=[c for c in captured if any(len(v)>1 for v in transform(c['prompt'],'grouped')[1].values())]
    if cases!=plan['cases']:raise ValueError('Captured prompts/schema/population changed')
    expected=[(c,rep+1,arm) for c in cases for rep in range(2) for arm in (plan['arms'] if rep==0 else list(reversed(plan['arms'])))];rows=[json.loads(x) for x in path.read_text().splitlines()]
    if len(expected)!=len(rows):raise ValueError('Incomplete or extra rows')
    sources={r['qa_id']:s for r,q,p,s,c in cohort};details=[];errors=[]
    for r,(c,rep,arm) in zip(rows,expected):
        prompt,mapping=transform(c['prompt'],arm)
        if (r['qa_id'],r['repetition'],r['arm'],r['prompt_sha256'],r['mapping'])!=(c['qa_id'],rep,arm,sha(prompt),mapping):raise ValueError('Row changed')
        if r['status']=='error':errors.append(r);continue
        proposals=expand(r['response'],mapping)
        if proposals!=r['proposals']:raise ValueError('Expansion changed')
        targets={x['claim_id']:x['text'] for x in json.loads(c['prompt'].split('\nClaims:\n')[1].split('\nOriginal source cards:\n')[0])}
        for p in proposals:
            model=ac.CitationRecovery.model_validate(p)
            citations=[ac._check_citation(ref,targets[model.claim_id],sources[r['qa_id']],model.premise_type) for ref in model.citations]
            details.append(dict(qa_id=r['qa_id'],repetition=rep,arm=arm,claim_id=model.claim_id,premise_type=model.premise_type,supported=model.supported,citations=[dict(source_id=x['source_id'],quote=x['quote'],valid=x['valid'],errors=x['validation_errors']) for x in citations]))
    return dict(status='passed',full_population=len(cohort),duplicate_batches=len(cases),calls=len(rows),errors=errors,details=details,caveat='Proposal diagnostic only; valid literal references do not establish semantic support, source attribution, or final-answer improvement.')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--replay',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=asyncio.run(audit(a.replay));a.output.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k not in ['details','caveat']}))
