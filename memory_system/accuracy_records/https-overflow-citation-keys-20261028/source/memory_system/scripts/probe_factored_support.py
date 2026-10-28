"""Frozen-packet diagnostic: source-independent premise inventory before support."""
import argparse
import asyncio
import copy
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import budget, config, eval_scoring, evidence_packet, llm
from app import personal_evidence
from replay_query_context import records, recorded_response
import support_reference_probe

INSTRUCTION = (
    'Identify existing personal premises in every option, without assessing support or choosing an answer. '
    'You have no memory evidence. Copy minimal exact contiguous substrings of the option, one personal '
    'fact per claim. Read each substring in its complete option context, preserving subject, negation, '
    'time and frequency. Separate existing personal attributes or experiences from suggested new actions '
    'and their proposed benefits. A recommendation can presuppose an existing personal fact: retain '
    'that fact without treating the proposed action as already experienced. A hypothetical condition '
    'does not establish the condition. Do not omit unsupported-looking premises: support is unknown. '
    'Do not invent words, paraphrase, or convert generic suggestions into history. Return every letter '
    'once; claims=[] only if the option asserts no distinguishing personal premise. '
    'Do not split a phrase if doing so loses a shared negation or other essential qualification.\n')


async def main(args):
    if config.FAKE:
        raise ValueError('Real provider required')
    result = next(r for r in records(args.run / 'completed/data/results/personamem-v2-32k.jsonl')
                  if r['qa_id'] == args.question)
    trace = next(t for t in records(args.run / 'server-search-debug.jsonl')
                 if t.get('search_id') == result['search_id'])
    qa = next(q for d in records(args.prepared) if d['conversation_id'] == result['conversation_id']
              for q in d['qa'] if q['id'] == args.question)
    packet = trace['returned']
    if evidence_packet.digest(packet) != result['packet_hash'] or trace['packet_hash'] != result['packet_hash']:
        raise ValueError('Packet mismatch')
    options = {chr(65+i): o for i,o in enumerate(qa['options'])}
    schema = dict(type='object', additionalProperties=False, required=['options'], properties={
        'options':dict(type='array', items=dict(type='object', additionalProperties=False,
            required=['letter','claims'], properties=dict(letter=dict(type='string', enum=list(options)),
            claims=dict(type='array', items=dict(type='string')))))})
    provider = llm.complete_json
    target_stage = 'eval.choice_support.repair' if args.mode in ('quote-repair','quote-refs') else 'eval.choice_citation_recovery'
    boundary = next((i for i,c in enumerate(result['answer_calls']) if c['stage']==target_stage), None)
    if args.mode!='inventory' and boundary is None:
        raise ValueError('This question has no recorded intervention boundary')

    async def run(enabled, frozen=False):
        upstream = copy.deepcopy(result['answer_calls'] if frozen else
            result['answer_calls'][:boundary] if args.mode!='inventory' else
            [c for c in result['answer_calls'] if c['stage']=='eval.choice_witness'])
        live, mismatches = [], []
        inventory = None

        async def call(prompt, *positional, **kwargs):
            nonlocal inventory
            if upstream:
                old = upstream.pop(0)
                if old['stage'] != kwargs['stage'] or hashlib.sha256(prompt.encode()).hexdigest() != old['prompt_sha256']:
                    mismatches.append(kwargs['stage'])
                    raise ValueError('Frozen reader prompt mismatch')
                budget.current.get().before_call()
                return recorded_response(old)
            if frozen:
                mismatches.append('unexpected_call')
                raise ValueError('Unexpected call during frozen preflight')
            if enabled and args.mode=='inventory' and kwargs['stage']=='eval.choice_support':
                inventory_prompt = INSTRUCTION + json.dumps(dict(question=qa['question'],options=options),ensure_ascii=False)
                inventory = await provider(inventory_prompt, schema=schema, stage='probe.premise_inventory',
                                           attempts=1, timeout=60, max_tokens=2048)
                live.append(dict(stage='probe.premise_inventory',prompt=inventory_prompt,response=inventory))
                entries=inventory['options']
                if len(entries)!=len(options) or {e['letter'] for e in entries}!=set(options):
                    raise ValueError('Incomplete/duplicate inventory')
                for e in entries:
                    if len(e['claims'])!=len(set(e['claims'])) or any(not c or c not in options[e['letter']] for c in e['claims']):
                        raise ValueError('Inventory claim is not unique exact option text')
                prompt += ('\nSource-independent personal premise inventory:\n'+json.dumps(inventory,ensure_ascii=False)
                           +'\nAssess these premises against all visible original sources. The inventory is '
                           'a decomposition aid, never evidence. Keep any additional personal premises that '
                           'the complete option actually asserts. Never remove a premise because support '
                           'is missing. Suggested actions are not past personal experiences.')
            if enabled and args.mode=='source-routes' and kwargs['stage']=='eval.choice_citation_recovery':
                claim_json, source_json = prompt.split('\nClaims:\n',1)[1].split('\nOriginal source cards:\n',1)
                claims, sources = json.loads(claim_json), json.loads(source_json)
                inventory=[]
                for claim in claims:
                    ranked=[]
                    for source in sources:
                        score=personal_evidence.source_score(
                            dict(content=source['text'],role=source['role']),claim['text'])
                        if score[2]>0:
                            ranked.append((score,source['id']))
                    ranked.sort(key=lambda x:(x[0],x[1]),reverse=True)
                    inventory.append(dict(claim_id=claim['claim_id'],source_ids=[sid for _,sid in ranked[:3]]))
                prompt += ('\nPer-premise source lookup hints (lexical relevance only, never evidence of truth; '
                           'read complete original cards and retain all support requirements):\n'
                           +json.dumps(inventory,ensure_ascii=False))
            if (enabled and args.mode=='quote-repair' and kwargs['stage']=='eval.choice_support.repair'
                    and 'quote_not_in_source' in prompt):
                prompt += ('\nThe prior quotation failed literal validation. Do not copy it from the '
                           'failed answer or feedback: those are not source evidence. Never summarize '
                           'a quotation with ellipses or invented list entries. Find an exact contiguous '
                           'quote in the original card. If no valid supporting quotation can be found, '
                           'retain the original personal premise with status=unsupported, reason=no_source '
                           'and citations=[]. Do not discard an actual personal premise, call it generic '
                           'because its quote failed, or keep supported without a valid quotation.')
            registry = None
            if enabled and args.mode=='quote-refs' and kwargs['stage']=='eval.choice_support.repair':
                prompt, kwargs['schema'], registry = support_reference_probe.request(prompt,kwargs['schema'])
            row=dict(stage=kwargs['stage'],prompt=prompt)
            live.append(row)
            response=await provider(prompt,*positional,**kwargs)
            row['response']=response
            if registry is not None:
                row['reference_registry']=registry
                response=support_reference_probe.decode(response,registry)
                row['decoded_response']=response
            return response

        with patch.object(llm,'complete_json',call), budget.scope(seconds=300,calls=21,tokens=160000) as usage:
            prediction, score, diagnostics=await eval_scoring.evaluate(qa,copy.deepcopy(packet))
        if upstream or mismatches:
            raise ValueError(dict(pending=len(upstream),mismatches=mismatches))
        return dict(qa_id=args.question,enabled=enabled,mode=args.mode,prediction=prediction,score=score,
                    inventory=inventory,diagnostics=diagnostics,live_calls=live,calls=usage.calls,
                    tokens=usage.tokens,packet_hash=result['packet_hash'])

    checked=await run(False,frozen=True)
    if checked['prediction']!=result['prediction'] or checked['score']!=result['score']:
        raise ValueError('Frozen answer mismatch')
    print(json.dumps(dict(qa_id=args.question,frozen_verified=True)),flush=True)
    with args.output.open('x') as out:
        for rep in range(args.repeat):
            for enabled in ([False,True] if rep%2==0 else [True,False]):
                row=await run(enabled)
                row.update(repetition=rep+1,frozen_verified=True,
                    caveat='Frozen packet and all reader calls before intervention boundary. Inventory mode '
                    'adds a model call before support; source-routes mode adds lexical source IDs at citation '
                    'recovery without changing prior claims. Expanded diagnostic budget; upstream time/tokens '
                    'omitted. No runtime change or HTTPS claim.')
                out.write(json.dumps(row,ensure_ascii=False)+'\n');out.flush()
                print(json.dumps({k:row[k] for k in ['qa_id','enabled','repetition','prediction','score']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['run','prepared','output']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--question',required=True)
    p.add_argument('--mode',choices=('inventory','source-routes','quote-repair','quote-refs'),default='inventory')
    p.add_argument('--repeat',type=int,choices=(1,2,3),default=2)
    asyncio.run(main(p.parse_args()))
