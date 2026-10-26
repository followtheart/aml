"""Compare both production missing-only modes to the archived offline proposal."""
import argparse
import copy
import json
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import fine_evidence,run_metadata
from probe_full_option_ce import prepare,base,sha


def main(args):
    report=json.loads(args.proposal.read_text());frozen=json.loads((args.archive/'manifest.json').read_text())
    for path,digest in report['inputs'].items():
        p=Path(path)
        if path in frozen:
            p=args.archive/frozen[path]['file'];assert frozen[path]['sha256']==digest
        assert sha(p)==digest,path
    with patch('socket.socket.connect',side_effect=AssertionError('Offline')):
        jobs,_,_=prepare(args.run);expected={r['qa_id']:r for r in report['rows']}
        assert len(jobs)==len(expected)==report['summary']['questions']
        for job in jobs:
            scores=[c['_ce_score'] for c in job['coarse']];row=expected[job['qa_id']]
            for name,enabled in (('baseline',False),('candidate',True)):
                revised=copy.deepcopy(job)
                revised['t']['versions']['settings'].update(FINE_NEW_COARSE_INSURANCE=False,FINE_NEW_COARSE_MISSING_ONLY=enabled)
                traces=[];original=fine_evidence.propose
                def capture(*a,**kw):
                    chosen,trace=original(*a,**kw);traces.append(trace);return chosen,trace
                with patch.object(fine_evidence,'propose',capture):actual=base.replay(revised,scores)
                assert len(traces)==1
                for stage in ('fine','main'):assert [c['id'] for c in actual[stage]]==row['variants'][name][stage]['ids']
                assert actual['main_bytes']==row['variants'][name]['main_bytes']
                if enabled:
                    adaptive=copy.deepcopy(row['adaptive'])
                    adaptive['new_coarse_insurance'].update(missing_only=True,baseline_fine_ids=row['variants']['baseline']['fine']['ids'])
                    assert traces[0]['adaptive']==adaptive,job['qa_id']
    result=dict(status='passed',questions=len(jobs),provider_calls=0,variants=report['summary']['variants'],
        production_versions=run_metadata.versions(),inputs={str(p.resolve()):sha(p) for p in (
            args.proposal,args.archive/'manifest.json',Path(__file__),Path(__file__).with_name('probe_source_ce.py'))})
    with args.output.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('variants','inputs','production_versions')}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--proposal',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--archive',type=Path,default=Path('runs/missing-new-coarse-20261025-proposal-source'))
    main(p.parse_args())
