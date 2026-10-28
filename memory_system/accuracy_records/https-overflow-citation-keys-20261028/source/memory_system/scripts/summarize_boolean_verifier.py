import argparse,hashlib,json
from pathlib import Path
from probe_boolean_verifier import SUFFIX,schema,validate
from probe_verifier_assessment import checks_in
from probe_question_source_filter import sha
from typed_entailment_prompt import normalize
from summarize_reader_replay import read_complete
from summarize_verifier_assessment import summarize as audit_source

def summarize(path,allow_partial=False):
 p=json.loads(path.with_suffix('.plan.json').read_text());src=Path(p['source'])
 if hashlib.sha256(src.read_bytes()).hexdigest()!=p['source_sha256'] or hashlib.sha256(src.with_suffix('.plan.json').read_bytes()).hexdigest()!=p['source_plan_sha256']:raise ValueError('Source changed')
 if audit_source(src)['partial']:raise ValueError('Incomplete source')
 original=json.loads(src.with_suffix('.plan.json').read_text())['cases']
 for c in original:c['normalized_prompt']=normalize(c['prompt'])
 if p['cases']!=original or p['suffix']!=SUFFIX or p['arms']!=['labels','booleans']:raise ValueError('Cases/policy changed')
 expected=[(c,a) for i,c in enumerate(original) for a in (p['arms'] if i%2==0 else list(reversed(p['arms'])))];rows=read_complete(path,allow_partial)
 if len(rows)>len(expected) or (not allow_partial and len(rows)!=len(expected)):raise ValueError('Incomplete/extra rows')
 for r,(c,a) in zip(rows,expected):
  checks=checks_in(c['prompt']);prompt=c['normalized_prompt']+(SUFFIX if a=='booleans' else '');contract=schema(checks) if a=='booleans' else c['schema']
  if (r['qa_id'],r['arm'],r['prompt_sha256'],r['schema_sha256'])!=(c['qa_id'],a,sha(prompt),sha(json.dumps(contract,sort_keys=True))):raise ValueError('Identity/prompt/schema changed')
  if r['calls'] not in [0,1]:raise ValueError('Extra calls')
  if r['status']=='ok':
   if validate(r['response'],checks,a)!=r['verdicts']:raise ValueError('Mapped verdict changed')
  elif r['status']=='error':
   if not r.get('error_type') or r.get('verdicts'):raise ValueError('Failure disguised')
   if 'response' in r:
    try:validate(r['response'],checks,a)
    except ValueError:pass
    else:raise ValueError('Valid response mislabeled')
  else:raise ValueError('Unknown status')
 changes=[]
 for i in range(0,len(rows)-1,2):
  pair={r['arm']:r for r in rows[i:i+2]}
  if all(r['status']=='ok' for r in pair.values()):
   maps={a:{c['claim_id']:c['verdict'] for c in r['verdicts']} for a,r in pair.items()}
   changes.extend(dict(qa_id=rows[i]['qa_id'],claim_id=k,labels=v,booleans=maps['booleans'][k]) for k,v in maps['labels'].items() if v!=maps['booleans'][k])
 return dict(partial=len(rows)!=len(expected),expected_pairs=len(original),complete_pairs=len(rows)//2,errors=[{k:r.get(k) for k in ['qa_id','arm','error_type']} for r in rows if r['status']=='error'],changes=changes,change_count=len(changes),calls={a:sum(r['calls'] for r in rows if r['arm']==a) for a in p['arms']},caveat=p['caveat'])
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--replay',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--allow-partial',action='store_true');a=p.parse_args();r=summarize(a.replay,a.allow_partial);a.output.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k not in ['changes','caveat']}))
