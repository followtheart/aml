"""Audit full-cohort guidance-only reading pairs against frozen source archives."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import direct_catalog_reader,evidence_packet
from summarize_reader_replay import read_complete


def summarize(path,allow_partial=False):
 plan=json.loads(path.with_suffix('.plan.json').read_text());prepared=Path(plan['prepared'])
 if hashlib.sha256(prepared.read_bytes()).hexdigest()!=plan['prepared_sha256']:raise ValueError('Prepared input changed')
 qa={(c['conversation_id'],q['id']):q for c in read_complete(prepared) for q in c['qa']}
 run=Path(plan['run']);base=read_complete(run/'completed/data/results/personamem-v2-32k.jsonl');traces={r['search_id']:r for r in read_complete(run/'server-search-debug.jsonl')}
 if [r['qa_id'] for r in base]!=plan['questions'] or {(r['conversation_id'],r['qa_id']) for r in base}!=set(qa):raise ValueError('Incomplete/changed cohort')
 expected={}
 for r in base:
  packet=traces[r['search_id']]['returned'];q=qa[r['conversation_id'],r['qa_id']]
  if evidence_packet.digest(packet)!=r['packet_hash']:raise ValueError('Packet changed')
  if q['qa_type']!='single_choice' or len(q['gold_labels'])!=1:raise ValueError('Single-choice only')
  prompt,context=direct_catalog_reader.make_prompt(q,packet,{s['id']:s for s in r['answer_source_catalog']},r['answer_constraint_catalog'])
  expected[r['qa_id']]=dict(conversation_id=r['conversation_id'],packet_hash=r['packet_hash'],source_sha256=hashlib.sha256(context.encode()).hexdigest(),prompts={a:hashlib.sha256((prompt+(plan['guidance'] if a=='guided' else '')).encode()).hexdigest() for a in ['direct','guided']},gold=q['gold_labels'][0])
 grouped={};errors=[]
 for r in read_complete(path,allow_partial):
  qid,arm=r['qa_id'],r['arm']
  if qid not in expected or arm not in ['direct','guided']:raise ValueError('Unexpected question/arm')
  e=expected[qid]
  if any(r[k]!=e[k] for k in ['conversation_id','packet_hash','source_sha256']) or r['prompt_sha256']!=e['prompts'][arm]:raise ValueError('Context/prompt differs from plan')
  arms=grouped.setdefault(qid,{})
  if arm in arms:raise ValueError('Duplicate arm')
  if r['status']=='ok':
   if r['prediction']!=r['response']['answer'] or r['score']!=float(r['prediction']==e['gold']):raise ValueError('Score mismatch')
  else:
   if r['score']!=0:raise ValueError('Failed row must score zero')
   errors.append(dict(qa_id=qid,arm=arm,error_type=r.get('error_type')))
  arms[arm]=r
 missing=[qid for qid in plan['questions'] if set(grouped.get(qid,{}))!={'direct','guided'}]
 if missing and not allow_partial:raise ValueError('Incomplete comparison')
 pairs=[dict(qa_id=qid,**{a:dict(prediction=grouped[qid][a].get('prediction'),score=grouped[qid][a]['score'],status=grouped[qid][a]['status']) for a in ['direct','guided']}) for qid in plan['questions'] if qid not in missing]
 return dict(partial=bool(missing),expected_pairs=len(plan['questions']),complete_pairs=len(pairs),missing=missing,errors=errors,direct=sum(p['direct']['score'] for p in pairs),guided=sum(p['guided']['score'] for p in pairs),wins=[p['qa_id'] for p in pairs if p['guided']['score']>p['direct']['score']],losses=[p['qa_id'] for p in pairs if p['guided']['score']<p['direct']['score']],pairs=pairs,caveat='Same archived sources/schema and instruction-only prompt difference verified. One draw per arm; answer score is not evidence of semantic grounding. Not a gated-reader or HTTPS comparison.')

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--replay',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--allow-partial',action='store_true');a=p.parse_args();r=summarize(a.replay,a.allow_partial);a.output.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k not in ['pairs','missing','caveat']}))
