"""Audit paired context ablation identity; summarize outputs without truth labels."""
import argparse,hashlib,itertools,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import integrity

def sha(x):return hashlib.sha256(x.encode()).hexdigest()
def records(p):return [json.loads(l) for l in p.open() if l.strip()]
def audit(path):
    data=records(path);plan=data[0];rows=data[1:]
    source=next(r for r in records(Path(plan['source_log'])) if r['request_id']==plan['request_id'])
    assert sha(json.dumps(source['source_messages'],sort_keys=True))==plan['source_sha256']
    originals={t['segment_index']:t for t in source['extraction_trace']}
    expected=set(itertools.product(range(plan['repeat']),originals,['control','new_only']))
    actual=[(r['repeat'],r['segment'],r['arm']) for r in rows]
    assert len(actual)==len(expected) and set(actual)==expected,'Incomplete/duplicate population'
    messages={s['message_index']:s for s in source['source_messages']};summary=[]
    for r in rows:
        archived=originals[r['segment']];assert r['indices']==archived['message_indices']
        prompt=next(e['prompt'] for e in archived['events'] if e['stage']=='extract_request')
        assert sha(prompt)==next(s['archived_prompt_sha256'] for s in plan['segments'] if s['index']==r['segment'])
        if r['arm']=='new_only':
            a='Conversation summary so far (global context, do not extract from it alone):\n'
            b='\n\nRecent messages (local context):\n';c='\n\nNew messages to extract from:\n'
            prefix,rest=prompt.split(a,1);_,rest=rest.split(b,1);_,rest=rest.split(c,1)
            prompt=prefix+a+'(withheld for diagnostic)'+b+'(withheld for diagnostic)'+c+rest
        events=[e for t in r['extraction_trace'] for e in t['events']]
        requests=[e['prompt'] for e in events if e['stage']=='extract_request'];assert requests==[prompt]
        atomic=[f for f in r['facts'] if f['type']!='episode']
        for f in atomic:
            assert f['evidence']
            for e in f['evidence']:
                assert e['message_index'] in r['indices'] and integrity.quote_in(e['quote'],messages[e['message_index']]['content'])
        summary.append(dict(repeat=r['repeat'],segment=r['segment'],arm=r['arm'],calls=r['calls'],atomic_facts=[f['content'] for f in atomic],source_indices=sorted({i for f in atomic for i in f['_sources']}),grounding_rejections=sum(e['stage']=='grounding_rejected' for e in events)))
    return dict(status='passed',pairs=len(rows)//2,calls=sum(r['calls'] for r in rows),rows=summary,caveat='Exact archived controls, only two context fields changed, complete paired population and original quote containment verified. Semantic judgments are model observations, not independently established truth; fact count is not quality and no downstream accuracy measured.')
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--probe',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();result=audit(a.probe)
    with a.output.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['rows','caveat']}))
