"""Paired extraction context ablation, all segments of one archived request.
No persistence, no gold inputs, normal original-source validation retained.
"""
import argparse, asyncio, copy, hashlib, json, sys, time
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import add_pipeline,budget,config,memory_debug,schemas,prompts

def digest(x):return hashlib.sha256(x.encode()).hexdigest()

async def main(a):
    if config.FAKE:raise ValueError('Real extraction required')
    row=next(json.loads(l) for l in a.memory_log.open() if json.loads(l).get('request_id')==a.request_id)
    messages=[]
    for i,s in enumerate(sorted(row['source_messages'],key=lambda s:s['message_index'])):
        assert s['message_index']==i
        fields={k:v for k,v in s.items() if k in schemas.Message.model_fields}
        if isinstance(fields.get('trust_scope'),str):fields['trust_scope']=json.loads(fields['trust_scope'])
        messages.append(schemas.Message(**fields))
    req=schemas.AddRequest(request_id=row['request_id'],user_id=row['user_id'],session_id=row['session_id'],messages=messages)
    segments=[]
    for t in row['extraction_trace']:
        original=next(e['prompt'] for e in t['events'] if e['stage']=='extract_request')
        summary=original.split('Conversation summary so far (global context, do not extract from it alone):\n',1)[1].split('\n\nRecent messages (local context):',1)[0]
        segments.append((t['segment_index'],t['message_indices'],original,summary))
    assert len({s[0] for s in segments})==len(segments)
    render=prompts.render
    def intervention(name,**values):
        if name=='01_extract_amu.txt':values=dict(values,session_summary='(withheld for diagnostic)',recent_messages='(withheld for diagnostic)')
        return render(name,**values)
    plan=dict(event='plan',source_log=str(a.memory_log),request_id=a.request_id,segments=[dict(index=s[0],indices=s[1],archived_prompt_sha256=digest(s[2])) for s in segments],repeat=a.repeat,source_sha256=digest(json.dumps(row['source_messages'],sort_keys=True)),intervention='Only session_summary and recent_messages in extraction prompt withheld; all new messages and normal original-source grounding/semantic checks unchanged. No DB writes.',caveat='Selected development request. Context ablation may lose coreference and attribution; no retrieval or accuracy claim.')
    with a.output.open('x') as out,patch.object(config,'MEMORY_DEBUG_LOG','diagnostic-capture-only'):
        out.write(json.dumps(plan)+'\n');out.flush()
        for repeat in range(a.repeat):
            for idx,indices,original,summary in segments:
                for arm in (['control','new_only'] if repeat%2==0 else ['new_only','control']):
                    seen=[]
                    def checked(name,**values):
                        prompt=render(name,**values)
                        if name=='01_extract_amu.txt':
                            assert prompt+add_pipeline._DOCUMENT_ATTRIBUTION_NOTE==original,'Archived prompt mismatch'
                            seen.append(digest(prompt))
                            if arm=='new_only':prompt=intervention(name,**values)
                        return prompt
                    started=time.monotonic()
                    with patch.object(prompts,'render',checked),budget.scope(seconds=240,calls=30,tokens=240000) as usage,memory_debug.extraction_scope():
                        facts=await add_pipeline._extract_segment(req,summary,idx,indices)
                        trace=copy.deepcopy(memory_debug._extraction.get())
                    assert seen,'No extraction request'
                    result=dict(event='result',repeat=repeat,segment=idx,indices=indices,arm=arm,facts=facts,extraction_trace=trace,calls=usage.calls,tokens=usage.tokens,elapsed_seconds=time.monotonic()-started)
                    out.write(json.dumps(result,ensure_ascii=False)+'\n');out.flush()
                    print(json.dumps({k:result[k] for k in ['repeat','segment','arm','calls','elapsed_seconds']}),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--memory-log',type=Path,required=True);p.add_argument('--request-id',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--repeat',type=int,default=2);a=p.parse_args()
    if a.repeat<1:p.error('positive repeat required')
    asyncio.run(main(a))
