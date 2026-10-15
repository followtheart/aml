"""Recover exact premise-scope inputs by replaying archived provider responses."""
import argparse
import asyncio
import copy
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

from replay_query_context import recorded_response, records
from app import budget, config, eval_scoring, evidence_packet, llm


async def main(args):
    rows = {r['qa_id']:r for r in records(args.run/'completed/data/results/personamem-v2-32k.jsonl')}
    traces = {r['search_id']:r for r in records(args.run/'server-search-debug.jsonl')}
    questions = {(d['conversation_id'],q['id']):q for d in records(args.prepared) for q in d['qa']}
    reports = []
    for qid in args.questions.split(','):
        row = rows[qid]
        if any(c.get('status') == 'error' for c in row['answer_calls']):
            raise ValueError('Cannot verify provider-error trajectory')
        packet = traces[row['search_id']]['returned']
        if evidence_packet.digest(packet) != row['packet_hash']:
            raise ValueError('Packet changed')
        pending = copy.deepcopy(row['answer_calls'])
        captured = []
        failures = []
        async def provider(prompt, *a, **kw):
            if not pending:
                failures.append('unexpected_call')
                raise ValueError('No live calls permitted')
            call = pending.pop(0)
            if call['stage'] != kw['stage'] or call['prompt_sha256'] != hashlib.sha256(prompt.encode()).hexdigest():
                failures.append(kw['stage'])
                raise ValueError('Archived prompt changed')
            if kw['stage'] == 'eval.choice_premise_scope':
                targets = json.loads(prompt.rsplit('\n',1)[1])
                captured.append(dict(prompt_sha256=call['prompt_sha256'],cases=targets))
            budget.current.get().before_call()
            return recorded_response(call)
        with patch.multiple(config, **row['versions']['settings']), patch.object(llm,'complete_json',provider):
            pred, score, _ = await eval_scoring.evaluate(questions[row['conversation_id'],qid], copy.deepcopy(packet))
        if pending or failures or (pred,score)!=(row['prediction'],row['score']) or len(captured)!=1:
            raise ValueError(dict(question=qid,pending=len(pending),failures=failures,captured=len(captured)))
        reports.append(dict(qa_id=qid,verified=True,**captured[0]))
    with args.output.open('x') as out:
        json.dump(dict(run=str(args.run),groups=reports,
                       caveat='Exact prompts and outcomes reproduced without network; captures contain no answer labels.'),out,indent=2)
        out.write('\n')
    print(json.dumps(dict(verified_questions=[r['qa_id'] for r in reports],case_counts=[len(r['cases']) for r in reports])))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--prepared',type=Path,required=True)
    p.add_argument('--questions',required=True)
    p.add_argument('--output',type=Path,required=True)
    asyncio.run(main(p.parse_args()))
