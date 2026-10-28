"""Replay complete factorial readers and audit literal citation/neighbor usage."""
import argparse
import asyncio
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

from probe_core_gap_scope_factorial import records,normalized,sha
from replay_query_context import recorded_response
from app import budget,config,eval_scoring,llm


def response(call):
    if ('response' not in call and call.get('error_type')=='LLMError'
            and call.get('failure_phase') is None):
        # Transport diagnostics retain the exception type, but not its message
        # or cause. Reproduce the recorded observable failure, not a payload.
        raise llm.LLMError(call.get('error_detail', 'Archived provider transport failure'))
    return recorded_response(call)


async def replay(row,case,settings):
    pending=copy.deepcopy(row['diagnostics']['answer_calls'])
    seen=[]
    async def recorded(prompt,*a,**kw):
        if not pending:
            raise ValueError('Unexpected replay call')
        old=pending.pop(0)
        if old['stage']!=kw['stage'] or old['prompt_sha256']!=hashlib.sha256(prompt.encode()).hexdigest():
            raise ValueError('Replay request changed')
        budget.current.get().before_call()
        seen.append(old)
        return response(old)
    with patch.multiple(config,**dict(settings,CHOICE_CITED_SCOPE=row['cited_scope'])),patch.object(llm,'complete_json',recorded),patch('socket.socket.connect',side_effect=AssertionError('Offline replay')):
        prediction,score,diagnostics=await eval_scoring.evaluate(case['qa'],copy.deepcopy(case['packets'][row['packet_mode']]))
    diagnostics=normalized(diagnostics)
    if pending or (prediction,score)!=(row['prediction'],row['score']):
        raise ValueError('Answer replay differs')
    for key in ('answer_calls','answer_source_catalog','answer_constraint_catalog','choice_alignment',
                'answer_eligible_options','answer_premise_scope','answer_validation'):
        if diagnostics.get(key)!=row['diagnostics'].get(key):
            details=[]
            if key=='answer_calls':
                for i,(a,b) in enumerate(zip(diagnostics[key],row['diagnostics'][key])):
                    if a!=b:
                        details.append(dict(index=i,fields=[k for k in a.keys()|b.keys() if a.get(k)!=b.get(k)],
                            actual_status=a.get('status'),expected_status=b.get('status')))
            raise ValueError('Replay diagnostics differ: '+json.dumps(dict(key=key,qa_id=row['qa_id'],
                repetition=row['repetition'],packet_mode=row['packet_mode'],scope=row['cited_scope'],details=details)))
    return len(seen)


async def main(args):
    root=args.run
    plan=json.loads((root/'plan.json').read_text())
    summary=json.loads((root/'summary.json').read_text())
    if sha(root/'plan.json')!=summary['plan_sha256'] or sha(root/'results.jsonl')!=summary['results_sha256']:
        raise ValueError('Factorial inputs changed')
    cases={c['qa']['id']:c for c in plan['cases']}
    rows=records(root/'results.jsonl')
    checked=[]
    for row in rows:
        count=await replay(row,cases[row['qa_id']],plan['settings'])
        checks=[]
        for request in row['requests']:
            if request['stage']=='eval.choice_entailment':
                checks=json.loads(request['prompt'].split('<checks>',1)[1].split('</checks>',1)[0])
        checked.append(dict(qa_id=row['qa_id'],repetition=row['repetition'],packet_mode=row['packet_mode'],
            cited_scope=row['cited_scope'],prediction=row['prediction'],score=row['score'],calls_verified=count,
            entailment_checks=checks,
            internal_errors=[c for c in row['diagnostics']['answer_calls'] if c['status']=='error']))
    if len(checked)!=plan['expected_readers']:
        raise ValueError('Incomplete factorial')
    events=[dict(p.split('=',1) for p in line.split() if '=' in p)
        for line in (root/'provider.log').read_text().splitlines() if 'provider_call ' in line]
    if len(events)!=summary['calls'] or sum(int(e['total_tokens']) for e in events)!=summary['tokens']:
        raise ValueError('Provider ledger mismatch')
    if any(e['model']!='gpt-4o-mini' for e in events):
        raise ValueError('LLM changed')
    result=dict(readers_replayed=len(checked),calls_replayed=sum(r['calls_verified'] for r in checked),
        rows=checked,provider_calls=len(events),recorded_tokens=summary['tokens'],
        provider_statuses=dict(Counter(e['status'] for e in events)),
        provider_error_events=[e for e in events if e['status']!='ok'],
        error_usage_note='Zero tokens recorded for a failed provider request do not establish its actual billing.',
        plan_sha256=sha(root/'plan.json'),results_sha256=sha(root/'results.jsonl'),auditor_sha256=sha(Path(__file__)))
    with args.output.open('x') as f:
        json.dump(result,f,ensure_ascii=False,indent=2)
        f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    asyncio.run(main(p.parse_args()))
