"""Reconcile paired CE result events against actual provider metrics logs."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(directory, log):
    plan=json.loads((directory/'plan.json').read_text())
    report=json.loads((directory/'result.json').read_text())
    replay=json.loads((directory/'audit.json').read_text())
    if replay['status']!='passed' or (directory/'exit-code').read_text().strip()!='0':
        raise ValueError('Require complete evaluation and offline selection audit')
    events=[json.loads(line) for line in (directory/'responses.jsonl').read_text().splitlines() if line.strip()]
    metrics=[];logged_events=[];failures=[]
    for line in log.read_text().splitlines():
        if 'provider_call ' in line:
            fields=dict(re.findall(r'(\w+)=([^\s]+)',line))
            if fields.get('status')!='ok':failures.append(fields)
            if fields.get('kind')!='rerank' or fields.get('model')!=plan['versions']['cross_encoder_model']:
                raise ValueError('Unexpected provider kind/model')
            metrics.append(fields)
        elif line.startswith('{'):
            try:event=json.loads(line)
            except ValueError:continue
            if 'qa_id' in event and 'arm' in event:logged_events.append(event)
    if failures:raise ValueError('Provider failure metrics require explicit accounting')
    if len(logged_events)!=len(events):raise ValueError('Incomplete provider/result event sequence')
    cursor=0;arms={};batches=[]
    jobs={j['qa_id']:j for j in plan['jobs']}
    for event,logged in zip(events,logged_events):
        if logged!={k:event[k] for k in ('qa_id','arm','calls','tokens')}:
            raise ValueError('Provider event order/accounting mismatch')
        calls=event['calls'];chunk=metrics[cursor:cursor+calls];cursor+=calls
        if len(chunk)!=calls or sum(int(m['total_tokens']) for m in chunk)!=event['tokens']:
            raise ValueError('Provider batch token ledger mismatch')
        if sum(int(m['input_count']) for m in chunk)!=len(jobs[event['qa_id']]['candidate_ids']):
            raise ValueError('Provider batch document counts mismatch')
        if any(int(m['attempt'])!=1 or int(m['retries'])!=0 for m in chunk):
            raise ValueError('Unexpected provider retry')
        arm=arms.setdefault(event['arm'],dict(calls=0,tokens=0,seconds=0.0))
        arm['calls']+=calls;arm['tokens']+=event['tokens'];arm['seconds']+=sum(float(m['duration_s']) for m in chunk)
        batches.append(dict(qa_id=event['qa_id'],arm=event['arm'],calls=calls,tokens=event['tokens']))
    if cursor!=len(metrics) or len(metrics)!=report['summary']['calls']:
        raise ValueError('Extra/missing provider calls')
    if sum(v['tokens'] for v in arms.values())!=report['summary']['tokens']:
        raise ValueError('Report token total mismatch')
    paths=[Path(__file__),log,*[directory/name for name in ('plan.json','result.json','audit.json','responses.jsonl','exit-code')]]
    return dict(status='passed',provider_calls=len(metrics),provider_errors=len(failures),questions=plan['preflight']['questions'],
        changed_questions=plan['preflight']['changed'],arms=arms,rows=batches,
        token_delta_fraction=arms['candidate']['tokens']/arms['control']['tokens']-1,
        inputs={str(p.resolve()):sha(p) for p in paths},
        limitation='Provider-reported token ledger; does not attest hidden proxy routing or HTTPS behavior.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--probe',type=Path,required=True)
    p.add_argument('--log',type=Path,required=True)
    args=p.parse_args();result=audit(args.probe,args.log)
    with (args.probe/'provider-ledger-audit.json').open('x') as stream:
        json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','inputs')}))
