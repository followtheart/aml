"""Reconstruct every diagnostic request and revalidate every recorded response."""
import argparse
import asyncio
from collections import Counter
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

from probe_keyed_witness import prepare,decode_keyed,fingerprint,reservation
from app import choice_witness


async def main(path):
    plan_path=path.with_suffix('.plan.json');plan=json.loads(plan_path.read_text())
    result=json.loads(path.read_text());run=Path(plan['run'])
    if hashlib.sha256(plan_path.read_bytes()).hexdigest()!=result['plan_sha256']:raise ValueError('Plan changed')
    for name,digest in plan['source_hashes'].items():
        if hashlib.sha256((run/name).read_bytes()).hexdigest()!=digest:raise ValueError('Source run changed')
    helper=Path(__file__).with_name('probe_keyed_witness.py')
    if hashlib.sha256(helper.read_bytes()).hexdigest()!=plan['script_sha256']:raise ValueError('Diagnostic code changed')
    with patch('socket.socket.connect',side_effect=AssertionError('Offline audit')):
        jobs,_=await prepare(run)
    if jobs!=plan['jobs']:raise ValueError('Frozen requests changed')
    jobs={j['qa_id']:j for j in jobs};seen=set();replayed=[]
    for row in result['rows']:
        key=row['qa_id'],row['arm'],row['repeat']
        if key in seen:raise ValueError('Duplicate response identity')
        seen.add(key);job=jobs[row['qa_id']];request=job['requests'][row['arm']]
        if (row['request_sha256']!=fingerprint(request) or row['packet_hash']!=job['packet_hash']
                or row['reservation']!=reservation(request) or row['calls'] not in (0,1)):
            raise ValueError('Request or budget identity differs')
        detail={};error=None;witnesses={}
        if 'response' in row:
            try:
                witnesses=(decode_keyed(row['response'],job['options'],job['sources'],detail) if row['arm']=='treatment'
                    else choice_witness.validate(row['response'],job['options'],job['sources'],detail))
            except Exception as exc:error=type(exc).__name__
            if error:
                if row.get('failure_phase')!='validation' or row.get('error_type')!=error:
                    raise ValueError('Validation error did not reproduce')
            else:
                if row.get('witnesses')!=witnesses or row['status']!=('partial' if detail.get('rejected_witnesses') else 'ok'):
                    raise ValueError('Validated witnesses differ')
            if row['validation']!=detail:raise ValueError('Validation trace differs')
        elif row.get('failure_phase')!='provider' or row['status']!='error':
            raise ValueError('Response missing without provider failure')
        rejected=detail.get('rejected_witnesses',[])
        replayed.append(dict(qa_id=row['qa_id'],arm=row['arm'],repeat=row['repeat'],status=row['status'],
            failure_phase=row.get('failure_phase'),error_type=row.get('error_type'),
            rejected=dict(Counter(r['reason'] for r in rejected)),
            valid_witnesses=sum(len(v) for v in witnesses.values()),
            first_party_witnesses=sum(w['first_party'] for ws in witnesses.values() for w in ws)))
    expected={(q,arm,repeat) for q in jobs for arm in ('control','treatment') for repeat in range(plan['repeats'])}
    if seen!=expected or len(seen)!=plan['expected_calls']:raise ValueError('Incomplete cohort')
    summary={arm:dict(requests=sum(r['arm']==arm for r in replayed),
        validation_errors=[r['qa_id'] for r in replayed if r['arm']==arm and r['failure_phase']=='validation'],
        provider_errors=[r['qa_id'] for r in replayed if r['arm']==arm and r['failure_phase']=='provider'],
        valid_witnesses=sum(r['valid_witnesses'] for r in replayed if r['arm']==arm),
        first_party_witnesses=sum(r['first_party_witnesses'] for r in replayed if r['arm']==arm),
        rejected=dict(sum((Counter(r['rejected']) for r in replayed if r['arm']==arm),Counter())))
        for arm in ('control','treatment')}
    output=path.with_suffix('.audit.json')
    with output.open('x') as f:
        json.dump(dict(summary=summary,requests_reconstructed=len(jobs)*2,responses_revalidated=len(replayed),
            rows=replayed,diagnostic_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            auditor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),limitation=__doc__),f,indent=2)
        f.write('\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--diagnostic',type=Path,required=True)
    asyncio.run(main(p.parse_args().diagnostic))
