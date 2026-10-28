"""Check scope-normalization code against archived full-reader trajectories.

No live provider calls are permitted. Provider-error rows are reported as
unverified rather than replaced or treated as successful validation.
"""
import argparse
import asyncio
import copy
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

from replay_query_context import recorded_response
from summarize_reader_replay import read_complete
from app import budget, config, eval_scoring, evidence_packet, llm


async def main(args):
    plan=json.loads(args.replay.with_suffix('.plan.json').read_text())
    setting = plan['setting']
    if setting not in ('CHOICE_CONDITIONAL_LIST_SCOPE', 'CHOICE_TEMPLATE_PATH_SCOPE',
                       'CHOICE_EXISTING_STATE_GUARD', 'CHOICE_REPAIR_VALID_CLAIMS', 'CHOICE_CITED_SCOPE') or plan['probe']:
        raise ValueError('Require original-packet scope-normalization experiment')
    run=Path(plan['run'])
    results={r['qa_id']:r for r in read_complete(run/'completed/data/results/personamem-v2-32k.jsonl')}
    traces={t['search_id']:t for t in read_complete(run/'server-search-debug.jsonl')}
    questions={(d['conversation_id'],q['id']):q for d in read_complete(args.prepared) for q in d['qa']}
    verified=[];unverified=[]
    for r in read_complete(args.replay,args.allow_partial):
        replay_enabled = not r['enabled'] if args.invert_flag else r['enabled']
        key=dict(qa_id=r['qa_id'],enabled=r['enabled'],replay_enabled=replay_enabled,repetition=r['repetition'])
        error_stages = [c['stage'] for c in r['diagnostics'].get('answer_calls', [])
                        if c.get('status') == 'error']
        if r['diagnostics'].get('answer_status')=='error' or error_stages:
            unverified.append(dict(**key,reason='archived_call_error',stages=error_stages));continue
        original=results[r['qa_id']];packet=traces[original['search_id']]['returned']
        if evidence_packet.digest(packet)!=r['packet_hash'] or r['packet_hash']!=original['packet_hash']:
            raise ValueError('Packet identity changed')
        pending=copy.deepcopy(r['diagnostics']['answer_calls']);mismatches=[]
        async def provider(prompt,*positional,**kwargs):
            if not pending:
                mismatches.append('unexpected_call');raise ValueError('Live calls forbidden')
            old=pending.pop(0)
            if old['stage']!=kwargs['stage'] or old['prompt_sha256']!=hashlib.sha256(prompt.encode()).hexdigest():
                mismatches.append(kwargs['stage']);raise ValueError('Prompt changed')
            budget.current.get().before_call()
            return recorded_response(old)
        with patch.multiple(config,**{setting:replay_enabled,'CHOICE_QUERY_CONTEXT':plan['query_context']}),patch.object(llm,'complete_json',provider):
            prediction,score,diagnostics=await eval_scoring.evaluate(questions[original['conversation_id'],r['qa_id']],copy.deepcopy(packet))
        if pending or mismatches or (prediction,score)!=(r['prediction'],r['score']):
            raise ValueError(dict(**key,pending=len(pending),mismatches=mismatches,prediction=prediction,score=score))
        verified.append(key)
    total=len(verified)+len(unverified);expected=len(plan['questions'])*plan['repeat']*2
    if total!=expected and not args.allow_partial:
        raise ValueError('Incomplete experiment')
    report=dict(partial=total!=expected,rows=total,expected_rows=expected,verified=verified,unverified=unverified,
                invert_flag=args.invert_flag, setting=setting,
                current_code_sha256=hashlib.sha256(Path('app/answer_choice.py').read_bytes()).hexdigest(),
                caveat='Exact prompt sequence and prediction/score equivalence on recorded non-error trajectories only; no network. '
                       'Does not prove equivalence on future model outputs or reproduce real token/time consumption.')
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(partial=report['partial'],rows=total,verified=len(verified),unverified=unverified)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--replay',type=Path,required=True)
    p.add_argument('--prepared',type=Path,default=Path('data/prepared/personamem-v2-32k.jsonl'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--allow-partial',action='store_true')
    p.add_argument('--invert-flag', action='store_true',
                   help='Require identical prompts and outcomes with the opposite conditional-scope flag; abort on divergence.')
    asyncio.run(main(p.parse_args()))
