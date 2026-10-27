"""Verify production fine/main selections against the preregistered offline proposal."""
import argparse
import copy
import json
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import fine_evidence,run_metadata
from probe_full_option_ce import prepare,base,sha


def audit(run,proposal,archive):
    report=json.loads(proposal.read_text());frozen=json.loads((archive/'manifest.json').read_text())
    for path,digest in report['inputs'].items():
        current=Path(path)
        if path in frozen:
            current=archive/frozen[path]['file']
            assert frozen[path]['sha256']==digest,path
        assert sha(current)==digest,path
    with patch('socket.socket.connect',side_effect=AssertionError('Offline audit')):
        jobs,_,_=prepare(run)
        expected={r['qa_id']:r for r in report['rows']}
        assert len(jobs)==len(expected)==report['summary']['questions']
        rows=[]
        for job in jobs:
            scores=[c['_ce_score'] for c in job['coarse']];row=expected[job['qa_id']]
            for name,enabled in (('original',False),('candidate',True)):
                current=copy.deepcopy(job)
                current['t']['versions']['settings']['FINE_NEW_COARSE_INSURANCE']=enabled
                observed=[];original=fine_evidence.propose
                def capture(*a,**kw):
                    chosen,trace=original(*a,**kw);observed.append(trace);return chosen,trace
                with patch.object(fine_evidence,'propose',capture):actual=base.replay(current,scores)
                assert len(observed)==1
                for stage in ('fine','main'):
                    assert [c['id'] for c in actual[stage]]==row['variants'][name][stage]['ids'],(job['qa_id'],name,stage)
                assert actual['main_bytes']==row['variants'][name]['main_bytes']
                if enabled:
                    adaptive=copy.deepcopy(row['adaptive'])
                    adaptive['new_coarse_insurance']=adaptive.pop('semantic_reserve')
                    assert observed[0]['adaptive']==adaptive,job['qa_id']
            rows.append(dict(qa_id=job['qa_id'],both_fine_and_main_exact=True,
                eligible=row['adaptive']['semantic_reserve']['eligible_ids'],used=row['adaptive']['semantic_reserve']['used']))
    return dict(status='passed',questions=len(rows),rows=rows,provider_calls=0,
        production_versions=run_metadata.versions(),inputs={str(p.resolve()):sha(p) for p in (
            proposal,archive/'manifest.json',Path(__file__),Path(base.__file__),
            Path(__file__).with_name('probe_source_ce.py'))})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--proposal',type=Path,required=True)
    p.add_argument('--archive',type=Path,default=Path('runs/new-coarse-insurance-20261025-proposal-source'))
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=audit(a.run,a.proposal,a.archive)
    with a.output.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','inputs','production_versions')}))
