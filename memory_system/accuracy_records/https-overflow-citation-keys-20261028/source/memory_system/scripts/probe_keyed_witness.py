"""Paired real-provider witness-shape diagnostic on identical frozen catalogs.

All 56 development questions are scheduled before calls, in alternating arm
order. Only output structure changes. Exact quote/role/option validators remain
strict. This is not HTTPS Search or a new answer-accuracy measurement; historical
answers are never replaced. No gold, old answer or selected option reaches LLM.
"""
import argparse
import asyncio
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import answer_choice,budget,choice_witness,config,evidence_packet,llm
from evaluation_model_guard import check_runtime,check_versions


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True).encode()).hexdigest()


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def keyed_request(request,options):
    letters=list(choice_witness._options(options))
    result=copy.deepcopy(request)
    schema=result['schema'];defs=schema['$defs']
    entry=copy.deepcopy(defs['OptionWitnesses'])
    entry['properties'].pop('letter');entry['required'].remove('letter');entry.pop('title',None)
    schema=dict(type='object',additionalProperties=False,
        properties={letter:copy.deepcopy(entry) for letter in letters},required=letters,
        **{'$defs':{'Witness':copy.deepcopy(defs['Witness'])}})
    before,separator,shape=result['prompt'].rpartition('Call emit_json_result with this shape:\n')
    if not separator or not shape.strip().startswith('{"options":'):
        raise ValueError('Unknown historical output-shape instruction')
    example={letter:dict(witnesses=[]) for letter in letters}
    example[letters[0]]['witnesses']=[dict(source_id='s0',quote='exact original text')]
    result['prompt']=(before+'Call emit_json_result with exactly these option-letter keys: '+', '.join(letters)+'.\n'
        'Each value contains only witnesses. Include every requested key; no additional keys, '
        'letter fields, or options array. Use an empty witnesses list for an option without useful sources.\n'
        'Shape example (choose actual source IDs and exact quotations):\n'+json.dumps(example,ensure_ascii=False)+'\n')
    result['schema']=schema
    # Everything before the output contract, including full catalogs, is identical.
    assert result['prompt'].startswith(before)
    assert result['system']==request['system'] and result['max_tokens']==request['max_tokens']
    return result


def decode_keyed(payload,options,sources,trace):
    letters=list(choice_witness._options(options))
    if type(payload) is not dict or set(payload)!=set(letters):
        raise ValueError('Return every requested option key exactly once, with no extra keys')
    compiled=[]
    for letter in letters:
        value=payload[letter]
        if type(value) is not dict or set(value)!={'witnesses'}:
            raise ValueError('Each option value must contain only witnesses')
        compiled.append(dict(letter=letter,witnesses=value['witnesses']))
    return choice_witness.validate(dict(options=compiled),options,sources,trace)


async def prepare(run):
    if (run/'exit-code').read_text().strip()!='0':raise ValueError('Require complete successful arm')
    rows=records(run/'completed/data/results/personamem-v2-32k.jsonl')
    traces={t['search_id']:t for t in records(run/'server-search-debug.jsonl')}
    jobs=[];settings=None
    for row in sorted(rows,key=lambda r:int(r['qa_id'])):
        t=traces[row['search_id']];check_versions(t['versions'],'service trace')
        cfg=t['versions']['settings']
        if settings is None:settings=cfg
        if settings!=cfg:raise ValueError('Mixed settings')
        if cfg['CHOICE_WITNESS_SPAN_REFS']:raise ValueError('Require original quotation format')
        assert evidence_packet.digest(t['returned'])==row['packet_hash']==t['packet_hash']
        sources,constraints=answer_choice.build_catalog(t['returned'])
        assert list(sources.values())==row['answer_source_catalog']
        assert list(constraints.values())==row['answer_constraint_catalog']
        old=next(c for c in row['answer_calls'] if c['stage']=='eval.choice_witness')
        captured=[]
        async def capture(prompt,**kwargs):
            captured.append(dict(prompt=prompt,**kwargs))
            return old.get('response',{})
        with patch.multiple(config,**cfg),patch.object(llm,'complete_json',capture),\
                patch('socket.socket.connect',side_effect=AssertionError('Offline request preparation')):
            await choice_witness.select(t['query'],t['options'],sources,{})
        if len(captured)!=1:raise ValueError('Witness request not reconstructed')
        request=captured[0]
        assert hashlib.sha256(request['prompt'].encode()).hexdigest()==old['prompt_sha256']
        request.pop('stage')
        revised=keyed_request(request,t['options'])
        jobs.append(dict(qa_id=row['qa_id'],packet_hash=row['packet_hash'],options=t['options'],sources=sources,
            requests=dict(control=request,treatment=revised),historical_status=old['status'],
            historical_error_type=old.get('error_type')))
    if len(jobs)!=56:raise ValueError('Require the full fixed 56-question cohort')
    return jobs,settings


def reservation(request):
    return len(request['prompt'].encode())+len(json.dumps(request['schema']).encode())+len(request['system'].encode())+1024+request['max_tokens']


async def main(args):
    check_runtime()
    plan_path=args.output.with_suffix('.plan.json');live_path=args.output.with_suffix('.live.jsonl')
    if any(p.exists() for p in (args.output,plan_path,live_path)):raise FileExistsError(args.output)
    jobs,settings=await prepare(args.run)
    plan=dict(run=str(args.run),model=config.LLM_MODEL,questions=len(jobs),repeats=args.repeats,
        expected_calls=len(jobs)*args.repeats*2,concurrency=args.concurrency,jobs=jobs,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_hashes={n:hashlib.sha256((args.run/n).read_bytes()).hexdigest() for n in
            ('server-search-debug.jsonl','completed/data/results/personamem-v2-32k.jsonl')},limitation=__doc__)
    plan_path.write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
    if args.prepare_only:
        print(json.dumps(dict(prepared=len(jobs),calls=0)));return
    if config.FAKE:raise ValueError('Real providers required')
    results=[];semaphore=asyncio.Semaphore(args.concurrency)
    async def paired(index,job):
        async with semaphore:
            for repeat in range(args.repeats):
                order=('control','treatment') if (index+repeat)%2==0 else ('treatment','control')
                for arm in order:
                    request=copy.deepcopy(job['requests'][arm]);bound=reservation(request)
                    row=dict(qa_id=job['qa_id'],arm=arm,repeat=repeat,request_sha256=fingerprint(request),
                        packet_hash=job['packet_hash'],reservation=bound)
                    phase='provider';detail={}
                    with budget.scope(seconds=choice_witness.TIMEOUT_SECONDS+1,calls=1,tokens=bound) as limits:
                        try:
                            async with asyncio.timeout(choice_witness.TIMEOUT_SECONDS):
                                payload=await llm.complete_json(**request,stage='diagnostic.keyed_witness.'+arm)
                            row['response']=payload;phase='validation'
                            witnesses=(decode_keyed(payload,job['options'],job['sources'],detail) if arm=='treatment'
                                else choice_witness.validate(payload,job['options'],job['sources'],detail))
                            row.update(status='partial' if detail.get('rejected_witnesses') else 'ok',
                                witnesses=witnesses,validation=detail)
                        except Exception as exc:
                            chain=[];cause=exc
                            while cause is not None and len(chain)<8:
                                chain.append(type(cause).__name__);cause=cause.__cause__
                            row.update(status='error',failure_phase=phase,error_type=type(exc).__name__,cause_types=chain,
                                validation=detail)
                            if phase=='validation':row['validation_detail']=str(exc)[:1000]
                    row.update(calls=limits.calls,tokens=limits.tokens);results.append(row)
                    with live_path.open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
                    print(json.dumps({k:v for k,v in row.items() if k not in ('response','witnesses','validation','validation_detail')}),flush=True)
    with patch.multiple(config,**settings):
        check_runtime()
        await asyncio.gather(*(paired(i,j) for i,j in enumerate(jobs)))
    summaries={}
    for arm in ('control','treatment'):
        rs=[r for r in results if r['arm']==arm]
        summaries[arm]=dict(calls=sum(r['calls'] for r in rs),tokens=sum(r['tokens'] for r in rs),
            statuses=dict(Counter(r['status'] for r in rs)),
            errors=dict(Counter((r.get('failure_phase','')+':'+r.get('error_type','')) for r in rs if r['status']=='error')),
            valid_witnesses=sum(len(v) for r in rs for v in r.get('witnesses',{}).values()),
            first_party_witnesses=sum(w['first_party'] for r in rs for ws in r.get('witnesses',{}).values() for w in ws))
    assert len(results)==plan['expected_calls']
    result=dict(summary=summaries,rows=results,plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),limitation=__doc__)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(summaries),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--repeats',type=int,default=1)
    p.add_argument('--concurrency',type=int,default=2);p.add_argument('--prepare-only',action='store_true')
    a=p.parse_args()
    if a.repeats<1 or a.concurrency<1:p.error('Positive repeats and concurrency required')
    asyncio.run(main(a))
