"""Compare mixed-option and per-option scope batches using identical schemas."""
import argparse
import asyncio
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import answer_choice as ac, budget, config, llm
from probe_verification_scope import scope_prefix


async def main(args):
    if config.FAKE:
        raise ValueError('Real provider required')
    groups = json.loads(args.cases.read_text())['groups']
    if len(groups)!=1:
        raise ValueError('One original scope call per invocation')
    group=groups[0]; cases=group['cases']
    args.output.with_suffix('.plan.json').write_text(json.dumps(dict(
        qa_id=group['qa_id'], cases=cases, repeat=2, arms=['mixed','per_option'],
        caveat='Selected scope diagnostic; no labels supplied and no downstream accuracy inferred.'),indent=2)+'\n')
    with args.output.open('x') as out:
        for rep in range(2):
            ordered=cases if rep==0 else list(reversed(cases))
            for enabled in ([False,True] if rep==0 else [True,False]):
                by_option={}
                for c in ordered:
                    by_option.setdefault(c['option'],[]).append(c)
                batches=list(by_option.values()) if enabled else [ordered]
                calls=[]; decisions={}; complete=True
                for batch in batches:
                    prompt=scope_prefix()+'\n'+json.dumps(batch,ensure_ascii=False)
                    with budget.scope(seconds=100,calls=1,tokens=30000):
                        response=await llm.complete_json(prompt,schema=ac.Scopes.model_json_schema(),
                            stage='diagnostic.scope_option_groups',attempts=1,timeout=90,max_tokens=3000)
                    parsed=ac.Scopes.model_validate(response)
                    got={c.claim_id:c.personal_fact for c in parsed.claims}
                    valid=(len(parsed.claims)==len(batch) and set(got)=={c['claim_id'] for c in batch})
                    complete=complete and valid
                    decisions.update(got)
                    calls.append(dict(prompt=prompt,response=response,complete=valid))
                row=dict(qa_id=group['qa_id'],repetition=rep+1,enabled=enabled,
                         complete=complete,cases=cases,calls=calls,decisions=decisions)
                out.write(json.dumps(row,ensure_ascii=False)+'\n');out.flush()
                print(json.dumps(dict(qa_id=group['qa_id'],repetition=rep+1,enabled=enabled,
                                      complete=complete,calls=len(calls))),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cases',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    asyncio.run(main(p.parse_args()))
