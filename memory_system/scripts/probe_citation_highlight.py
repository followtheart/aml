"""Verifier-only ablation: remove redundant quote highlights, preserve full sources."""
import argparse,asyncio,copy,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import answer_choice as ac,budget,config,llm
from probe_question_source_filter import sha
from summarize_typed_entailment_prompt import summarize


def strip_highlights(prompt):
    before,rest=prompt.split('<checks>\n',1);raw,after=rest.split('\n</checks>',1)
    checks=json.loads(raw);changed=copy.deepcopy(checks);removed=0
    for c in changed:
        for source in c.get('sources',[]):
            if 'quote' not in source:continue
            if not isinstance(source.get('context'),str) or not isinstance(source['quote'],str) or source['quote'] not in source['context']:
                raise ValueError('Quote is not contained in preserved full context')
            del source['quote'];removed+=1
    return before+'<checks>\n'+json.dumps(changed,ensure_ascii=False)+'\n</checks>'+after,removed

async def main(args):
    if config.FAKE:raise ValueError('Real provider required')
    checked=summarize(args.replay)
    if checked['partial']:raise ValueError('Complete audited replay required')
    rows=[json.loads(l) for l in args.replay.read_text().splitlines()];cases=[]
    for r in rows:
        if r['arm']!='baseline':continue
        call=next(c for c in r['live_calls'] if c['stage']=='eval.choice_entailment')
        alternate,removed=strip_highlights(call['prompt'])
        cases.append(dict(qa_id=r['qa_id'],baseline_prompt=call['prompt'],full_context_prompt=alternate,removed=removed))
    plan=dict(replay=str(args.replay),replay_sha256=hashlib.sha256(args.replay.read_bytes()).hexdigest(),questions=[c['qa_id'] for c in cases],arms=['baseline','full_context'],script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),caveat='Complete22 archived verifier batches; all context, roles, checks and surrounding instructions unchanged. Only repeated quote fields removed. Verdict diagnostic, no final-answer accuracy claim.')
    with args.output.open('x') as out:
        with args.output.with_suffix('.plan.json').open('x') as f:f.write(json.dumps(plan,indent=2)+'\n')
        for i,case in enumerate(cases):
            expected={c['claim_id'] for c in json.loads(case['baseline_prompt'].split('<checks>\n')[1].split('\n</checks>')[0])}
            for arm in (['baseline','full_context'] if i%2==0 else ['full_context','baseline']):
                prompt=case[arm+'_prompt'];row=dict(qa_id=case['qa_id'],arm=arm,prompt_sha256=sha(prompt),removed_quotes=case['removed'] if arm=='full_context' else 0)
                try:
                    with budget.scope(seconds=100,calls=1,tokens=64000) as usage:
                        response=await llm.complete_json(prompt,schema=ac.TypedEntailments.model_json_schema(),stage='diagnostic.citation_highlight.'+arm,attempts=1,timeout=90,max_tokens=4096)
                    row.update(response=response,calls=usage.calls,tokens=usage.tokens)
                    parsed=ac.TypedEntailments.model_validate(response)
                    if len(parsed.checks)!=len(expected) or {c.claim_id for c in parsed.checks}!=expected:raise ValueError('Missing/extra/duplicate checks')
                    row['status']='ok'
                except Exception as exc:row.update(status='error',error_type=type(exc).__name__,error=str(exc)[:500])
                out.write(json.dumps(row,ensure_ascii=False)+'\n');out.flush();print(json.dumps({k:v for k,v in row.items() if k!='response'}),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['replay','output']:p.add_argument('--'+n,type=Path,required=True)
    asyncio.run(main(p.parse_args()))
