"""All duplicate-premise recovery batches, three arms; no answer-score claim."""
import argparse,asyncio,copy,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import answer_choice as ac,budget,llm
from probe_question_source_filter import sha,load_cohort
from probe_recovery_source_order import capture

def transform(prompt,arm):
    prefix,rest=prompt.split('\nClaims:\n');raw,suffix=rest.split('\nOriginal source cards:\n')
    claims=json.loads(raw);groups={};requested=[]
    for c in claims:
        key=c['text'] if arm=='grouped' else c['claim_id']
        if key not in groups:
            groups[key]=[];requested.append(c if arm=='original' else {k:c[k] for k in ['claim_id','text']})
        groups[key].append(c['claim_id'])
    mapping={v[0]:v for v in groups.values()}
    result=prompt if arm=='original' else prefix+'\nClaims:\n'+json.dumps(requested,ensure_ascii=False)+'\nOriginal source cards:\n'+suffix
    return result,mapping

def expand(response,mapping):
    parsed=ac.normalize_citation_recovery_matches(response,list(mapping));out=[]
    for p in parsed:
        for cid in mapping[p.claim_id]:out.append(dict(p.model_dump(),claim_id=cid))
    return out

async def main(a):
    cohort=load_cohort(a.run,a.prepared);cases=await capture(cohort)
    cases=[c for c in cases if any(len(v)>1 for v in transform(c['prompt'],'grouped')[1].values())]
    arms=['original','claim_only','grouped'];plan=dict(run=str(a.run),prepared=str(a.prepared),prepared_sha256=hashlib.sha256(a.prepared.read_bytes()).hexdigest(),full_population=[r['qa_id'] for r,*_ in cohort],cases=cases,arms=arms,repetitions=2,selection='All duplicate exact-text premise batches, regardless of answer outcome; diagnostic only.',script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    a.output.with_suffix('.plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
    with a.output.open('x') as f:
        for c in cases:
            for rep in range(2):
                for arm in (arms if rep==0 else list(reversed(arms))):
                    prompt,mapping=transform(c['prompt'],arm);row=dict(qa_id=c['qa_id'],repetition=rep+1,arm=arm,prompt_sha256=sha(prompt),mapping=mapping)
                    try:
                        with budget.scope(seconds=100,calls=1,tokens=64000):
                            response=await llm.complete_json(prompt,schema=c['schema'],system=c['system'],stage='diagnostic.grouped_recovery.'+arm,attempts=1,timeout=90,max_tokens=c['max_tokens'])
                        row['response']=response;row['proposals']=expand(response,mapping);row['status']='ok'
                    except Exception as e:row.update(status='error',error_type=type(e).__name__)
                    f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush();print(json.dumps({k:row[k] for k in ['qa_id','repetition','arm','status']}),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ['run','prepared','output']:p.add_argument('--'+n,type=Path,required=True)
    asyncio.run(main(p.parse_args()))
