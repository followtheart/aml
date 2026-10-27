"""Fixed-prefix diagnostic: verify early recovered evidence before retaining scope."""
import argparse
import asyncio
import copy
import hashlib
import json
from pathlib import Path
from unittest.mock import patch
from replay_source_vector_route import source_route
from replay_query_context import records, recorded_response
from app import answer_choice as ac, budget, config, eval_scoring, evidence_packet, llm, prompts




async def verify_recovered(original_entries, recovered, sources, qa, diagnostics):
    originals = {f"{e['letter']}:{i}": c for e in original_entries for i, c in enumerate(e['claims'])}
    targets = {f"{e['letter']}:{i}": c for e in recovered for i, c in enumerate(e['claims'])
               if c.get('citation_recovery') == 'awaiting_entailment' and c != originals[f"{e['letter']}:{i}"]}
    if not targets:
        return recovered
    options = ac.option_map(qa['options'])
    checks = [c for c in ac.entailment_checks(recovered, sources, qa['options']) if c['claim_id'] in targets]
    if {c['claim_id'] for c in checks} != set(targets):
        raise ValueError('Recovered claim missing verification check')
    for check in checks:
        check.pop('proposed_status', None)
        check['option_context'] = options[check['claim_id'].split(':')[0]]
    prompt = prompts.render('15_choice_entailment.txt', question=qa['question'],
        question_date=qa.get('question_date', ''), checks=json.dumps(checks, ensure_ascii=False))
    prompt += ('\nOutput verdict instead of entailed. For each extracted premise, choose '
        'personal_supported only when it asserts a prior personal fact supported by these sources; '
        'no_personal_premise when it is advice, a general-world statement, or hypothetical; '
        'unsupported_personal when a prior personal fact lacks support. Interpret the phrase '
        'inside option_context, preserving its subject, time and negation. Classify actual '
        'meaning, not the proposed premise_type. Return every claim_id exactly once.')
    verdicts = await ac._judge(prompt, ac.TypedEntailments.model_json_schema(),
        lambda result: ac._typed_verdicts(result, checks), 'diagnostic.early_recovery_entailment', diagnostics)
    keep = {v['claim_id'] for v in verdicts['checks'] if v['entailed']}
    restored = []
    for entry in recovered:
        for i, claim in enumerate(entry['claims']):
            cid = f"{entry['letter']}:{i}"
            if cid in targets and cid not in keep:
                entry['claims'][i] = copy.deepcopy(originals[cid])
                restored.append(cid)
    diagnostics['diagnostic_early_recovery_validation'] = dict(kept=sorted(keep), restored=restored)
    return recovered


async def main(args):
    if config.FAKE:raise ValueError('Real provider required')
    archived=next(r for r in records(args.replay) if r['qa_id']==args.question and r['arm']=='window')
    probe=json.loads(args.probe.read_text());q=next(q for q in probe['questions'] if q['qa_id']==args.question)
    trace=next(t for t in records(args.run/'server-search-debug.jsonl') if t['search_id']==q['search_id'])
    packet,_=source_route(probe,q,trace['returned'],trace['evidence_token_budget'])
    if evidence_packet.digest(packet)!=archived['packet_hash']:raise ValueError('Packet changed')
    qa=next(qa for d in records(args.prepared) for qa in d['qa'] if qa['id']==args.question)
    calls=archived['diagnostics']['answer_calls'];boundary=next(i for i,c in enumerate(calls) if c['stage']=='eval.choice_premise_scope')
    original=llm.complete_json
    recover=ac.recover_missing_citations

    async def run(enabled,frozen=False):
        pending=copy.deepcopy(calls if frozen else calls[:boundary]);live=[];violations=[]
        async def provider(prompt,*positional,**kwargs):
            if pending:
                old=pending.pop(0)
                if old['stage']!=kwargs['stage'] or old['prompt_sha256']!=hashlib.sha256(prompt.encode()).hexdigest():
                    violations.append(kwargs['stage']);raise ValueError('Frozen prompt mismatch')
                budget.current.get().before_call()
                return recorded_response(old)
            if frozen:
                violations.append('unexpected_live_call');raise ValueError('Network forbidden during reproduction')
            actual=prompt
            call=dict(stage=kwargs['stage'],prompt=actual,prompt_sha256=hashlib.sha256(actual.encode()).hexdigest());live.append(call)
            response=await original(actual,*positional,**kwargs);call['response']=response
            return response
        async def verified_recover(entries, sources, question, diagnostics):
            before = copy.deepcopy(entries)
            recovered = await recover(entries, sources, question, diagnostics)
            return await verify_recovered(before, recovered, sources, question, diagnostics) if enabled else recovered

        with patch.object(config,'CHOICE_QUERY_CONTEXT',False), \
             patch.object(config,'CHOICE_RECOVERY_BEFORE_SCOPE',enabled), \
             patch.object(llm,'complete_json',provider), \
             patch.object(ac,'recover_missing_citations',verified_recover):
            prediction,score,diagnostics=await eval_scoring.evaluate(qa,copy.deepcopy(packet))
        if pending or violations:raise ValueError('Frozen trajectory mismatch')
        return dict(qa_id=args.question,enabled=enabled,prediction=prediction,score=score,
                    diagnostics=diagnostics,live_calls=live,packet_hash=archived['packet_hash'])

    check=await run(False,True)
    if (check['prediction'],check['score'])!=(archived['prediction'],archived['score']):
        raise ValueError('Frozen outcome mismatch')
    args.output.with_suffix('.plan.json').write_text(json.dumps(dict(qa_id=args.question,
        frozen_verified=True,intervention='Recover citations before scope pruning, verify recovered personal premises independently, restore original uncited claims when evidence fails. Final verifier unchanged.',
        caveat='Selected diagnostic, fixed prefix and packet, live recovery/downstream. No general accuracy inference.'),indent=2)+'\n')
    with args.output.open('x') as stream:
        for rep in range(getattr(args, 'repetitions', 2)):
            for enabled in ([False,True] if (rep + getattr(args, 'order_offset', 0)) % 2 == 0 else [True,False]):
                row=await run(enabled);row['repetition']=rep+1
                stream.write(json.dumps(row,ensure_ascii=False)+'\n');stream.flush()
                print(json.dumps({k:row[k] for k in ('repetition','enabled','prediction','score')}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('run','probe','replay','prepared','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--question',required=True)
    p.add_argument('--repetitions',type=int,default=2)
    p.add_argument('--order-offset',type=int,default=0)
    asyncio.run(main(p.parse_args()))
