"""Factorial diagnostic: JSON layout versus explicit conversation-link metadata.

Source string values, order, ownership, schema and validators are invariant.
Original-layout linked requests may exceed production prompt caps; only the
compact-layout variant is a deployment candidate if the actual cap permits it.
"""
import argparse
import asyncio
import copy
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import budget,config,listwise_recovery as recovery,llm,recovery_spans as spans,schemas
from conversation_link_proposal import conversation_links
from evaluation_model_guard import REQUIRED_MODEL,check_runtime,check_versions

GUIDANCE = (' The conversation_links field records consecutive original user and assistant messages, '
            'including displayed excerpts: assistant_source is the reply immediately following user_source. '
            'Read linked text together for conversational context while preserving each speaker and all '
            'question, quotation, fictional, negation and hypothetical framing. A reply does not establish '
            'the truth of its premises or turn assistant material into user facts. These structural links '
            'do not indicate usefulness and grant no additional citation ownership. Judge each candidate '
            'under the original relevance and citation rules.')


def variant(request,built,*,compact,linked):
    result=copy.deepcopy(request)
    prefix,raw=result['prompt'].rsplit('\n',1)
    data=json.loads(raw)
    if linked:
        data['conversation_links']=conversation_links(built)
        result['system']+=GUIDANCE
    result['prompt']=prefix+'\n'+json.dumps(data,ensure_ascii=False,**({'separators':(',',':')} if compact else {}))
    result['reservation']+=(len(result['prompt'].encode())-len(request['prompt'].encode())
                            +len(result['system'].encode())-len(request['system'].encode()))
    recovered=json.loads(result['prompt'].rsplit('\n',1)[1]);recovered.pop('conversation_links',None)
    if recovered!=json.loads(raw) or result['schema']!=request['schema']:
        raise ValueError('Source data or contract changed')
    return result


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


async def main(args):
    check_runtime()
    plan_path=args.output.with_suffix('.plan.json');live_path=args.output.with_suffix('.live.jsonl')
    if any(p.exists() for p in (args.output,plan_path,live_path)):
        raise FileExistsError(args.output)
    if (args.run/'exit-code').read_text().strip()!='0':raise ValueError('Require completed arm')
    qas=set(args.qas.split(','));jobs=[];rows=[]
    traces={t['search_id']:t for t in records(args.run/'server-search-debug.jsonl')}
    for result in records(args.run/'completed/data/results/personamem-v2-32k.jsonl'):
        if result['qa_id'] not in qas:continue
        t=traces[result['search_id']];check_versions(t['versions'],'source trace')
        req=schemas.SearchRequest(**{k:t[k] for k in ('user_id','query','options','top_k','reference_time','evidence_token_budget')})
        v=t['cascade']['listwise']['recovery']['batches'][0];cards=v['sources']
        with patch.multiple(config,**t['versions']['settings']):
            original=recovery.request_for(req,cards)
            assert hashlib.sha256(original['prompt'].encode()).hexdigest()==v['prompt_sha256']
            assert original['reservation']==v['request_bound']
            built=spans.build(req,cards)
            variants={f'compact_{int(compact)}_linked_{int(linked)}':variant(original,built,compact=compact,linked=linked)
                      for compact in (False,True) for linked in (False,True)}
        assert variants['compact_0_linked_0']==original
        if not conversation_links(built):raise ValueError('Selected batch has no links')
        jobs.append((dict(qa_id=result['qa_id'],batch=0,stage='main',prompt_limit=v['prompt_limit_bytes'],
            sizes={name:len(r['prompt'].encode()) for name,r in variants.items()}),variants,built))
    if qas!={m['qa_id'] for m,_,_ in jobs}:raise ValueError('Missing question')
    plan=dict(model=REQUIRED_MODEL,run=str(args.run),repeats=args.repeats,expected_calls=len(jobs)*4*args.repeats,
              jobs=[m for m,_,_ in jobs],limitation=__doc__)
    plan_path.write_text(json.dumps(plan,indent=2)+'\n')
    if config.FAKE:raise ValueError('Real provider required')
    for meta,variants,built in jobs:
        names=list(variants)
        for repeat in range(args.repeats):
            ordered=names[repeat%4:]+names[:repeat%4]
            for name in ordered:
                check_runtime()
                request=copy.deepcopy(variants[name]);reservation=request.pop('reservation')
                row=dict(**meta,variant=name,repeat=repeat,request_bound=reservation)
                with budget.scope(seconds=45,calls=1,tokens=reservation) as limits:
                    try:
                        async with asyncio.timeout(45):
                            payload=await llm.complete_json(**request,attempts=1,timeout=45,stage='diagnostic.link_encoding.'+name)
                        selected,judgments=spans.decode_partial(payload,built)
                        row.update(selected_ids=sorted(selected),judgments=judgments,payload=payload)
                    except Exception as exc:
                        row.update(error=type(exc).__name__,detail=str(exc))
                row.update(calls=limits.calls,tokens=limits.tokens);rows.append(row)
                with live_path.open('a') as out:out.write(json.dumps(row,ensure_ascii=False)+'\n')
                print(json.dumps({k:v for k,v in row.items() if k not in ('judgments','payload','detail','sizes')}),flush=True)
    args.output.write_text(json.dumps(dict(plan=plan,live=rows),ensure_ascii=False,indent=2)+'\n')
    if len(rows)!=plan['expected_calls'] or any('error' in r for r in rows):raise RuntimeError('Incomplete diagnostic')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--qas',default='20,22,23')
    p.add_argument('--repeats',type=int,default=3)
    args=p.parse_args()
    if args.repeats<1:p.error('Positive repeats required')
    asyncio.run(main(args))
