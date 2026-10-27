"""Extend the frozen selection audit to recorded partial CE scores.

Missing scores stay None. Replays the exact fine-selection block extracted from
the frozen production cascade, including failed-score rescue and core quotas.
Does not call providers, omit questions, retry an evaluation or alter a service.
"""
import argparse
import ast
import copy
import hashlib
import inspect
import json
import math
from pathlib import Path
import sys
from unittest.mock import patch


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def assignment(node,name):
    return isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in node.targets)


def main(args):
    root=args.run.resolve();project=root/'source/memory_system'
    sys.path.insert(0,str(project));sys.path.insert(0,str(project/'scripts'))
    from app import cascade_rerank,config,schemas
    import audit_main_prompt_https as reader
    import audit_new_coarse_https as auditor
    import probe_full_option_ce as probe
    from probe_coarse_fine_core import admit
    plan=json.loads((root/'plan.json').read_text());fixture=json.loads((root/'fixture.json').read_text())
    for name,digest in plan['source_files'].items():assert sha(root/'source'/name)==digest,name
    # Keep every original identity/plan/source check. Broaden only CE status.
    source=inspect.getsource(reader.progress_inputs)
    needle="assert t['cascade']['cross_encoder']['status']=='ok'"
    assert source.count(needle)==1
    source=source.replace(needle,"assert t['cascade']['cross_encoder']['status'] in ('ok','partial')")
    namespace=dict(vars(reader));exec(compile(source,'<partial-ce-inputs>','exec'),namespace)
    read_inputs=namespace['progress_inputs']
    parsed=ast.parse(inspect.getsource(cascade_rerank.rank));body=parsed.body[0].body
    starts=[i for i,n in enumerate(body) if assignment(n,'failed')]
    stops=[i for i,n in enumerate(body) if assignment(n,'fine_ids')]
    assert len(starts)==len(stops)==1 and starts[0]<stops[0]
    fragment=ast.Module(body=body[starts[0]:stops[0]],type_ignores=[])
    assert not any(isinstance(n,(ast.Await,ast.AsyncFor,ast.AsyncWith)) for n in ast.walk(fragment))
    code=compile(ast.fix_missing_locations(fragment),'<frozen-production-fine-selection>','exec')
    partial={};replayed={}

    def validate_scores(job,scores):
        t=job['t'];ce=t['cascade']['cross_encoder']
        assert ce['status'] in ('ok','partial')
        assert len(scores)==len(job['coarse'])
        assert all(v is None or type(v) in (int,float) and math.isfinite(v) for v in scores)
        scored={c['id'] for c,v in zip(job['coarse'],scores) if v is not None}
        succeeded={mid for b in ce['batches'] if b['status']=='ok' for mid in b['candidate_ids']}
        assert scored==succeeded and len(scored)==ce['scored_count']>0
        if ce['status']=='partial':
            assert len(scored)<ce['input_count']
            partial[t['search_id']]=dict(qa_id=job['qa_id'],search_id=t['search_id'],scored=len(scored),
                missing_ids=[c['id'] for c in job['coarse'] if c['id'] not in scored],errors=ce['errors'])
        else:assert len(scored)==ce['input_count']

    def replay(job,scores):
        validate_scores(job,scores)
        t=job['t'];coarse=copy.deepcopy(job['coarse']);scored={}
        for c,value in zip(coarse,scores):
            c['_ce_score']=value
            if value is None:c.pop('_final',None)
            else:c['_final']=value;scored[c['id']]=value
        replay_plan=copy.deepcopy(t['plan']);replay_plan['_query_specs']=copy.deepcopy(t['query_specs'])
        env=dict(vars(cascade_rerank),coarse=coarse,scores=scored,trace=copy.deepcopy(t['cascade']),
            req=schemas.SearchRequest(user_id=t['user_id'],query=t['query'],options=t['options']),plan=replay_plan,
            requirements=t['coverage_manifest']['coverage']['requirements'])
        with patch.multiple(config,**t['versions']['settings']):
            exec(code,env)
            assert 'adaptive' in env['fine_selection']
            main,size=admit(t,env['fine'],env['fine_selection'])
        replayed[t['search_id']]=job['qa_id']
        return dict(fine=env['fine'],main=main,main_bytes=size)

    def prepare(run,**unused):
        jobs,manifest,conversations,_,_=read_inputs(run,plan,fixture)
        assert manifest['client_exit_code']==manifest['audit_exit_code']==0
        assert 'server_exit_code' in manifest
        assert len(jobs)==plan['questions']
        return jobs,manifest,conversations

    # This process owns these module objects; the running services are untouched.
    with patch('socket.socket.connect',side_effect=AssertionError('Offline partial CE audit')),\
         patch.object(reader,'progress_inputs',read_inputs),patch.object(probe,'prepare',prepare),\
         patch.object(probe.base,'replay',replay):
        auditor.main(args)
    result=json.loads(args.output.read_text())
    result['partial_ce_support']=dict(policy='recorded-successful-scores-and-production-rescue',
        partial_searches=list(partial.values()),all_unique_searches_replayed=len(replayed),provider_calls=0,
        production_selection_sha256=hashlib.sha256(ast.dump(fragment).encode()).hexdigest(),
        frozen_cascade_sha256=sha(Path(cascade_rerank.__file__)),frozen_auditor_sha256=sha(Path(auditor.__file__)),
        wrapper_sha256=sha(Path(__file__)),limitation=__doc__)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['partial_ce_support']))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gold-csv',type=Path,required=True);p.add_argument('--progress',action='store_true')
    main(p.parse_args())
