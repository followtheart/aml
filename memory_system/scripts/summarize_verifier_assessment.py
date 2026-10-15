"""Audit complete verifier-assessment pairs against archived prompt identities."""
import argparse,hashlib,json
from pathlib import Path
from probe_verifier_assessment import SUFFIX,assessment_schema,checks_in,validate
from probe_question_source_filter import load_cohort,sha
from summarize_reader_replay import read_complete

def summarize(path,allow_partial=False):
    plan=json.loads(path.with_suffix('.plan.json').read_text());prepared=Path(plan['prepared'])
    if hashlib.sha256(prepared.read_bytes()).hexdigest()!=plan['prepared_sha256']:raise ValueError('Input changed')
    if plan['suffix']!=SUFFIX or plan['arms']!=['labels','assessment']:raise ValueError('Policy changed')
    cohort=load_cohort(Path(plan['run']),prepared)
    if [r['qa_id'] for r,*_ in cohort]!=plan['all_questions']:raise ValueError('Incomplete population')
    archived=[(r['qa_id'],c['prompt_sha256']) for r,*_ in cohort for c in r['answer_calls'] if c['stage']=='eval.choice_entailment']
    if archived!=[(c['qa_id'],sha(c['prompt'])) for c in plan['cases']]:raise ValueError('Verifier cases changed')
    expected=[(c,a) for i,c in enumerate(plan['cases']) for a in (plan['arms'] if i%2==0 else list(reversed(plan['arms'])))];rows=read_complete(path,allow_partial)
    if len(rows)>len(expected) or (not allow_partial and len(rows)!=len(expected)):raise ValueError('Incomplete/extra rows')
    for row,(case,arm) in zip(rows,expected):
        prompt=case['prompt']+(SUFFIX if arm=='assessment' else '');checks=checks_in(case['prompt']);schema=assessment_schema(checks) if arm=='assessment' else case['schema']
        if (row['qa_id'],row['arm'],row['prompt_sha256'],row['schema_sha256'])!=(case['qa_id'],arm,sha(prompt),sha(json.dumps(schema,sort_keys=True))):raise ValueError('Identity/prompt/schema changed')
        if row['status']=='ok':
            if row['verdicts']!=validate(row['response'],checks,arm):raise ValueError('Verdict changed')
        elif row['status']=='error':
            if not row.get('error_type') or row.get('verdicts'):raise ValueError('Failure disguised')
            if 'response' in row:
                try:validate(row['response'],checks,arm)
                except ValueError:pass
                else:raise ValueError('Valid response marked error')
        else:raise ValueError('Unknown status')
    changes=[];pairs=[]
    for i in range(0,len(rows)-1,2):
        by={r['arm']:r for r in rows[i:i+2]};pairs.append(by)
        if all(r['status']=='ok' for r in by.values()):
            maps={a:{x['claim_id']:x['verdict'] for x in r['verdicts']} for a,r in by.items()}
            changes.extend(dict(qa_id=rows[i]['qa_id'],claim_id=k,labels=v,assessment=maps['assessment'][k]) for k,v in maps['labels'].items() if v!=maps['assessment'][k])
    return dict(partial=len(rows)!=len(expected),expected_pairs=len(plan['cases']),complete_pairs=len(pairs),errors=[{k:r.get(k) for k in ['qa_id','arm','error_type']} for r in rows if r['status']=='error'],changes=changes,change_count=len(changes),calls={a:sum(r['calls'] for r in rows if r['arm']==a) for a in plan['arms']},caveat=plan['caveat'])
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--replay',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--allow-partial',action='store_true');a=p.parse_args();r=summarize(a.replay,a.allow_partial);a.output.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k not in ['changes','caveat']}))
