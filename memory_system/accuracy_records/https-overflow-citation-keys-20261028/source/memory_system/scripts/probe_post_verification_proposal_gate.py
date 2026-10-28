"""Fixed discovery proposals: compare support-gated and literal-only source attachment."""
import argparse,asyncio,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import answer_choice as ac,budget,config,llm
from probe_post_verification_citations import make_cases,attach
from summarize_post_verification_citations import summarize
from probe_question_source_filter import sha

async def main(args):
    if config.FAKE:raise ValueError('Real verifier required')
    checked=summarize(args.proposals)
    if checked['partial'] or checked['errors']:raise ValueError('Complete valid discovery diagnostic required')
    previous=json.loads(args.proposals.with_suffix('.plan.json').read_text())
    cases={c['qa_id']:c for c in make_cases(Path(previous['run']),Path(previous['prepared']),Path(previous['replay']))}
    proposals=[r for r in map(json.loads,args.proposals.read_text().splitlines()) if r['arm']=='supplemented']
    plan=dict(proposals=str(args.proposals),proposals_sha256=hashlib.sha256(args.proposals.read_bytes()).hexdigest(),rows=28,arms=['gated','literal_only'],scope='Same14 frozen discovery responses; no new discovery, gold or final answers. Both arms live independent verification. Only whether proposal.supported can veto exact first-party citations changes; no support automatically granted.',script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with args.output.open('x') as out:
        with args.output.with_suffix('.plan.json').open('x') as f:f.write(json.dumps(plan,indent=2)+'\n')
        for i,old in enumerate(proposals):
            case=cases[old['qa_id']];response=old['calls'][0]['response']
            for arm in (plan['arms'] if i%2==0 else list(reversed(plan['arms']))):
                prompt,added,rejected=attach(case,response,use_proposal_verdict=arm=='gated')
                row=dict(qa_id=old['qa_id'],repetition=old['repetition'],arm=arm,proposal_sha256=sha(json.dumps(response,sort_keys=True)),prompt=prompt,prompt_sha256=sha(prompt),added=added,rejected=rejected)
                try:
                    with budget.scope(seconds=100,calls=1,tokens=64000) as usage:
                        answer=await llm.complete_json(prompt,schema=ac.TypedEntailments.model_json_schema(),stage='diagnostic.proposal_gate.'+arm,attempts=1,timeout=90,max_tokens=4096)
                    row.update(response=answer,calls=usage.calls,tokens=usage.tokens)
                    parsed=ac.TypedEntailments.model_validate(answer)
                    if len(parsed.checks)!=len(case['checks']) or {c.claim_id for c in parsed.checks}!={c['claim_id'] for c in case['checks']}:raise ValueError('Incomplete verifier checks')
                    row.update(status='ok',verdicts={c.claim_id:c.verdict for c in parsed.checks if c.claim_id in case['targets']})
                except Exception as exc:row.update(status='error',error_type=type(exc).__name__,error=str(exc)[:500])
                out.write(json.dumps(row,ensure_ascii=False)+'\n');out.flush();print(json.dumps({k:v for k,v in row.items() if k not in ['prompt','response','added','rejected']}),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['proposals','output']:p.add_argument('--'+n,type=Path,required=True)
    asyncio.run(main(p.parse_args()))
