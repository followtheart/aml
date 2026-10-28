"""Isolate evidence-reading instructions from note output and retrieval changes."""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import answer_choice,budget,config,direct_catalog_reader,eval_scoring,evidence_packet,llm,run_metadata
from replay_query_context import records

GUIDANCE = '''
Compare every option against the question and the supplied original sources.
Prefer useful personalization when its distinguishing personal premise is supported and relevant to this request. Do not replace an explicitly supported, relevant preference with unrelated biography or generic advice merely because the latter is broadly safe.
Distinguish proposed activities and their intended benefits from claims about existing user facts. Suggested activities need not have happened before. Possessions, diagnoses, frequencies, past events, and specific causes of distress need evidence for that particular detail; topical questions alone do not establish them.
Keep the speaker and described person separate. A friend or colleague's experience does not become the user's. A forwarded message addressed to the user can report an event involving them, but preserve its actual strength and qualifications. Assistant advice or rewrites alone do not establish personal history. User-authored first-person drafts can contain user facts.
Apply the supplied forget constraints and later corrections. Do not treat a forgotten fact as current personalization. When evidence does not support personalization, compare the relevance of the generic alternatives. Do not add personal details that only appear in an option.
Return only the chosen answer letter in the required structured field; do not output notes.
'''

async def main(args):
 if config.FAKE:raise ValueError('Real provider required')
 if (args.run/'exit-code').read_text().strip()!='0':raise ValueError('Require successful complete HTTPS')
 archived=records(args.run/'completed/data/results/personamem-v2-32k.jsonl')
 questions={(c['conversation_id'],q['id']):q for c in records(args.prepared) for q in c['qa']}
 ids=[(r['conversation_id'],r['qa_id']) for r in archived]
 if len(ids)!=len(set(ids)) or set(ids)!=set(questions):raise ValueError('Require full unique cohort')
 traces={r['search_id']:r for r in records(args.run/'server-search-debug.jsonl')}
 plan=dict(run=str(args.run),prepared=str(args.prepared),prepared_sha256=hashlib.sha256(args.prepared.read_bytes()).hexdigest(),questions=[r['qa_id'] for r in archived],arms=['direct','guided'],guidance=GUIDANCE,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),versions=run_metadata.versions(),basis='LongMemEval notes13: isolate reading from retrieval. Previous note experiment changed both instructions and schema; this changes instructions only.',caveat='One live call per arm, same packet, original catalogs and answer-only schema. No hard entailment guarantees.')
 args.output.with_suffix('.plan.json').write_text(json.dumps(plan,indent=2)+'\n')
 with args.output.open('x') as out:
  for index,r in enumerate(archived):
   packet=traces[r['search_id']]['returned']
   if evidence_packet.digest(packet)!=r['packet_hash']:raise ValueError('Packet changed')
   qa=questions[r['conversation_id'],r['qa_id']];sources={s['id']:s for s in r['answer_source_catalog']};constraints=r['answer_constraint_catalog']
   base,context=direct_catalog_reader.make_prompt(qa,packet,sources,constraints)
   letters=list(answer_choice.option_map(qa['options']));schema=dict(type='object',additionalProperties=False,required=['answer'],properties=dict(answer=dict(type='string',enum=letters)))
   for arm in (['direct','guided'] if index%2==0 else ['guided','direct']):
    prompt=base+(GUIDANCE if arm=='guided' else '')
    row=dict(qa_id=r['qa_id'],conversation_id=r['conversation_id'],arm=arm,packet_hash=r['packet_hash'],source_sha256=hashlib.sha256(context.encode()).hexdigest(),prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest())
    try:
     with budget.scope(seconds=95,calls=1,tokens=64000) as usage:
      response=await llm.complete_json(prompt,schema=schema,stage='diagnostic.reader_guidance.'+arm,attempts=1,timeout=90,max_tokens=128)
     row.update(response=response,calls=usage.calls,tokens=usage.tokens)
     if not isinstance(response,dict) or set(response)!={'answer'} or response['answer'] not in letters:raise ValueError('Invalid answer')
     row.update(status='ok',prediction=response['answer'],score=eval_scoring.choice_score(response['answer'],qa['gold_labels'],qa['qa_type']))
    except Exception as exc:row.update(status='error',error_type=type(exc).__name__,error=str(exc)[:500],score=0)
    out.write(json.dumps(row,ensure_ascii=False)+'\n');out.flush();print(json.dumps(row),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 for name in ['run','prepared','output']:p.add_argument('--'+name,type=Path,required=True)
 asyncio.run(main(p.parse_args()))
