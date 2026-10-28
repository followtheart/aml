"""Freeze initial selection; test including incumbent original-source context in review."""
import argparse
import asyncio
import copy
import hashlib
import json
from pathlib import Path
import re
import sys
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import answer_choice, budget, config, eval_scoring, evidence_packet, llm
from replay_query_context import records, recorded_response
from replay_source_vector_route import source_route


def clean_scaffolding(text):
    text=re.sub(r"\b(?:i|you|we|they|he|she|it)['’](?:m|re|ve|ll|d|s)\b",' ',text,flags=re.I)
    return clean_request_scaffolding(text)


def clean_request_scaffolding(text):
    return re.sub(r'\b(?:look(?:ing|ed|s)?|search(?:ing|ed|es)?|seek(?:ing|s)?)\s+for\b',' ',text,flags=re.I)


async def main(args):
    if config.FAKE:raise ValueError('Real live review required')
    archived=next(r for r in records(args.replay) if r['qa_id']==args.question and r['enabled'])
    baseline=next(r for r in records(args.run/'completed/data/results/personamem-v2-32k.jsonl') if r['qa_id']==args.question)
    trace=next(r for r in records(args.run/'server-search-debug.jsonl') if r.get('search_id')==baseline['search_id'])
    packet=trace['returned']
    if args.probe:
        probe=json.loads(args.probe.read_text());question=next(q for q in probe['questions'] if q['qa_id']==args.question)
        packet,_=source_route(probe,question,packet,trace['evidence_token_budget'])
    if evidence_packet.digest(packet)!=archived['packet_hash']:raise ValueError('Packet mismatch')
    qa=next(q for d in records(args.prepared) if d['conversation_id']==baseline['conversation_id'] for q in d['qa'] if q['id']==args.question)
    calls=archived['diagnostics']['answer_calls']
    boundary=next((i for i,c in enumerate(calls) if c['stage']=='eval.choice_select_review'),len(calls))
    original=llm.complete_json;old_prompt=answer_choice.selection_review_prompt
    sources,_=answer_choice.build_catalog(packet)
    letters=answer_choice.option_map(qa['options'])
    eligible=archived['diagnostics']['answer_eligible_options']
    option_terms={k:answer_choice.contrastive_context_terms(letters[k]) for k in eligible}
    weights={t:1+len(option_terms)-sum(t in terms for terms in option_terms.values())
             for terms in option_terms.values() for t in terms}
    def incumbent_prompt(prompt,evidence,focus,selected):
        evidence=copy.deepcopy(evidence)
        if not any(e['letter']==selected for e in evidence):
            candidates=[]
            for source in sources.values():
                if source.get('role')!='user' and source.get('declared')!='persona':continue
                if source.get('declared')!='persona' and source['text'].rstrip().endswith('?') and not answer_choice._SELF.search(source['text']):continue
                match=answer_choice.local_context_match(source['text'],option_terms[selected],weights)
                if len(match['shared_terms'])>=2:
                    candidates.append((sum(weights[t] for t in match['shared_terms']),source,match))
            if candidates:
                _,source,match=max(candidates,key=lambda row:(row[0],-len(row[2]['excerpt'])))
                evidence.append(dict(letter=selected,source_id=source['id'],role=source.get('declared') or source.get('role'),**match))
        return old_prompt(prompt,evidence,focus,selected)
    async def run(enabled,frozen=False):
        pending=copy.deepcopy(calls if frozen else calls[:boundary]);mismatches=[];live=[]
        async def provider(prompt,*positional,**kwargs):
            if pending:
                old=pending.pop(0)
                if old['stage']!=kwargs['stage'] or old['prompt_sha256']!=hashlib.sha256(prompt.encode()).hexdigest():
                    mismatches.append(kwargs['stage']);raise ValueError('Frozen stage/prompt mismatch')
                budget.current.get().before_call();return recorded_response(old)
            if frozen or kwargs['stage']!='eval.choice_select_review':
                mismatches.append('unexpected_live');raise ValueError('Unexpected live stage')
            record=dict(stage=kwargs['stage'],prompt=prompt);live.append(record)
            response=await original(prompt,*positional,**kwargs);record['response']=response;return response
        with patch.object(config,'CHOICE_QUERY_CONTEXT',archived.get('query_context',archived.get('context_enabled',False))),patch.object(answer_choice,'selection_review_prompt',incumbent_prompt if enabled else old_prompt),patch.object(llm,'complete_json',provider):
            prediction,score,diagnostics=await eval_scoring.evaluate(qa,copy.deepcopy(packet))
        if pending or mismatches:raise ValueError(dict(pending=len(pending),mismatches=mismatches))
        return dict(qa_id=args.question,enabled=enabled,prediction=prediction,score=score,diagnostics=diagnostics,
                    live_calls=live,packet_hash=archived['packet_hash'])
    checked=await run(False,frozen=True)
    if (checked['prediction'],checked['score'])!=(archived['prediction'],archived['score']):raise ValueError('Frozen outcome mismatch')
    with args.output.open('x') as out:
        for repetition in range(args.repeat):
            for enabled in ([False,True] if (repetition+int(args.reverse))%2==0 else [True,False]):
                row=await run(enabled);row.update(repetition=repetition+1,frozen_verified=True,
                    intervention='incumbent_original_context',
                    caveat='Frozen all calls through first selection; only incumbent source hint varies; trigger and eligible options unchanged. Selected case, not full reader or HTTPS accuracy.')
                out.write(json.dumps(row,ensure_ascii=False)+'\n');out.flush()
                print(json.dumps({k:row[k] for k in ['qa_id','repetition','enabled','prediction','score']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('run','replay','prepared','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--probe',type=Path);p.add_argument('--question',required=True)
    p.add_argument('--repeat',type=int,choices=(1,2,3),default=2)
    p.add_argument('--reverse',action='store_true')
    p.add_argument('--request-only',action='store_true')
    asyncio.run(main(p.parse_args()))
