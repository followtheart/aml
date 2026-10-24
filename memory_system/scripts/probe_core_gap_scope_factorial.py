"""Fixed-packet 2x2 reader diagnostic: complete core-gap context and cited scope.

Select every question whose archived packet changes under the existing core-gap
production function, before looking at outcomes. Compare original/augmented
packets and cited-scope off/on with fresh full readers and balanced arm order.
This selected-case diagnostic is not an HTTPS or whole-cohort accuracy result.
"""
import argparse
import asyncio
import copy
import hashlib
import json
import logging
from pathlib import Path
from unittest.mock import patch

from probe_source_rank_residual import records
from replay_query_context import recorded_response
from app import answer_choice, budget, config, core_gap_context, eval_scoring, evidence_packet, llm, search_pipeline
from evaluation_model_guard import check_runtime


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized(value):
    return json.loads(json.dumps(value,ensure_ascii=False))


async def replay(case, settings):
    pending=copy.deepcopy(case['original']['answer_calls'])
    calls=[]
    async def recorded(prompt,*a,**kw):
        if not pending:
            raise ValueError('Unexpected historical call')
        call=pending.pop(0)
        if call['stage']!=kw['stage'] or call['prompt_sha256']!=hashlib.sha256(prompt.encode()).hexdigest():
            raise ValueError('Historical call identity differs')
        budget.current.get().before_call()
        calls.append(call)
        return recorded_response(call)
    with patch.multiple(config,**settings),patch.object(llm,'complete_json',recorded),patch('socket.socket.connect',side_effect=AssertionError('Offline reader verification')):
        prediction,score,diagnostics=await eval_scoring.evaluate(case['qa'],copy.deepcopy(case['packets']['original']))
    diagnostics=normalized(diagnostics)
    if pending or (prediction,score)!=(case['original']['prediction'],case['original']['score']):
        raise ValueError('Historical reader answer differs')
    for key in ('answer_calls','answer_source_catalog','answer_constraint_catalog','choice_alignment','answer_eligible_options'):
        if diagnostics.get(key)!=case['original'].get(key):
            raise ValueError('Historical reader differs: '+key)
    return len(calls)


async def main(args):
    check_runtime()
    if config.FAKE or config.CE_MODEL!='qwen3-rerank':
        raise ValueError('Require unchanged real models')
    root=args.run.resolve()
    traces=records(root/'search/server-search-debug.jsonl')
    settings=json.loads((root/'control/reader-versions.json').read_text())['settings']
    if settings['CHOICE_CITED_SCOPE']:
        raise ValueError('Require original scope-off control')
    cases=[]
    with patch('socket.socket.connect',side_effect=AssertionError('Offline packet proposal')):
        for t in traces:
            with patch.multiple(config,**t['versions']['settings']):
                if evidence_packet.digest(t['returned'])!=t['packet_hash']:
                    raise ValueError('Original packet hash differs')
                pool={c['id']:c for c in search_pipeline._foresight_filter(copy.deepcopy(t['ranked']),copy.deepcopy(t['plan']))
                    if c.get('view_status')!='stale' and '_packet_item' in c}
                nominees,nomination=core_gap_context.nominate(pool,t['returned'],
                    fine_ids=t['cascade']['fine']['selected_ids'],coarse_ids=t['cascade']['coarse']['selected_ids'],
                    selection=t['cascade']['fine']['selection'],count=config.CASCADE_FINE_LIMIT)
                added,digest,manifest=core_gap_context.attach(copy.deepcopy(t['returned']),t['packet_hash'],
                    copy.deepcopy(t['coverage_manifest']),nominees,nomination,user_id=t['user_id'],top_k=t['top_k'])
                detail=manifest['conversation_context']['core_gap']
                if not detail['appended_ids']:
                    continue
                if [dict(p,packet_hash=t['packet_hash']) for p in added[:len(t['returned'])]]!=t['returned']:
                    raise ValueError('Original prefix changed')
                cases.append(dict(search_id=t['search_id'],user_id=t['user_id'],
                    packets=dict(original=t['returned'],augmented=added),added=detail,
                    hashes=dict(original=t['packet_hash'],augmented=digest)))
    # Outcomes and question answers are accessed only after the case set is fixed.
    inputs={j['payload']['search_id']:j for j in records(root/'http-reader-inputs.jsonl')}
    original={r['search_id']:r for r in records(root/'control/completed/data/results/personamem-v2-32k.jsonl')}
    for case in cases:
        case['qa']=inputs[case['search_id']]['qa']
        case['original']=original[case['search_id']]
        case['verified_calls']=await replay(case,settings)
    args.output.mkdir(parents=True,exist_ok=False)
    plan=dict(source_run=str(root),questions=len(cases),repeats=args.repeat,
        expected_readers=len(cases)*args.repeat*4,settings=settings,llm_model=config.LLM_MODEL,
        source_trace_sha256=sha(root/'search/server-search-debug.jsonl'),script_sha256=sha(Path(__file__)),
        cases=[{k:v for k,v in c.items() if k!='original'} for c in cases],protocol=__doc__)
    (args.output/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(event='plan',questions=[c['qa']['id'] for c in cases],readers=plan['expected_readers'],
        verified_calls=sum(c['verified_calls'] for c in cases))),flush=True)
    handler=logging.FileHandler(args.output/'provider.log',mode='x')
    logging.getLogger('aml.metrics').addHandler(handler)
    logging.getLogger('aml.metrics').setLevel(logging.INFO)
    provider=llm.complete_json
    rows=[]
    for repetition in range(args.repeat):
        for index,case in enumerate(cases):
            arms=[('original',False),('augmented',True),('original',True),('augmented',False)]
            if (repetition+index)%2:
                arms.reverse()
            for packet_mode,scope in arms:
                requests=[]
                usages=[]
                async def capture(prompt,*a,**kw):
                    requests.append(dict(stage=kw['stage'],prompt=prompt,
                        **{k:kw[k] for k in ('schema','system','max_tokens') if k in kw}))
                    usages.append(budget.current.get())
                    return await provider(prompt,*a,**kw)
                with patch.multiple(config,**dict(settings,CHOICE_CITED_SCOPE=scope)),patch.object(llm,'complete_json',capture):
                    prediction,score,diagnostics=await eval_scoring.evaluate(case['qa'],copy.deepcopy(case['packets'][packet_mode]))
                usage=usages[-1] if usages else None
                row=dict(qa_id=case['qa']['id'],repetition=repetition+1,packet_mode=packet_mode,
                    cited_scope=scope,prediction=prediction,score=score,packet_hash=case['hashes'][packet_mode],
                    calls=usage.calls if usage else 0,tokens=usage.tokens if usage else 0,
                    requests=requests,diagnostics=normalized(diagnostics))
                rows.append(row)
                with (args.output/'results.jsonl').open('a') as out:
                    out.write(json.dumps(row,ensure_ascii=False)+'\n')
                print(json.dumps({k:v for k,v in row.items() if k not in ('diagnostics','requests','packet_hash')}),flush=True)
    summary={}
    for packet_mode in ('original','augmented'):
        for scope in (False,True):
            rs=[r for r in rows if r['packet_mode']==packet_mode and r['cited_scope']==scope]
            summary[f'{packet_mode}_scope_{int(scope)}']=dict(readers=len(rs),score=sum(r['score'] for r in rs),
                predictions={c['qa']['id']:[r['prediction'] for r in rs if r['qa_id']==c['qa']['id']] for c in cases},
                terminal_errors=sum(bool(r['diagnostics'].get('error_stage')) for r in rs))
    result=dict(summary=summary,calls=sum(r['calls'] for r in rows),tokens=sum(r['tokens'] for r in rows),
        readers=len(rows),plan_sha256=sha(args.output/'plan.json'),results_sha256=sha(args.output/'results.jsonl'),
        limitation=__doc__)
    if len(rows)!=plan['expected_readers']:
        raise ValueError('Incomplete factorial')
    (args.output/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    (args.output/'exit-code').write_text('0\n')
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--repeat',type=int,choices=(1,2,3),default=2)
    asyncio.run(main(p.parse_args()))
