"""Audit full-cohort quote-highlight verifier pairs and report verdict changes."""
import argparse,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app.answer_choice import TypedEntailments
from probe_citation_highlight import strip_highlights
from probe_question_source_filter import sha
from summarize_reader_replay import read_complete

def summarize(path,partial=False):
    plan=json.loads(path.with_suffix('.plan.json').read_text());source=Path(plan['replay'])
    if hashlib.sha256(source.read_bytes()).hexdigest()!=plan['replay_sha256']:raise ValueError('Source replay changed')
    cases=[]
    for r in read_complete(source):
        if r['arm']=='baseline':
            c=next(c for c in r['live_calls'] if c['stage']=='eval.choice_entailment')
            p=c['prompt'];q,n=strip_highlights(p);cases.append(dict(qa_id=r['qa_id'],baseline=p,full_context=q,removed=n))
    if [c['qa_id'] for c in cases]!=plan['questions'] or plan['arms']!=['baseline','full_context']:raise ValueError('Plan changed')
    expected=[(c,a) for i,c in enumerate(cases) for a in (plan['arms'] if i%2==0 else list(reversed(plan['arms'])))]
    rows=read_complete(path,partial)
    if len(rows)>len(expected) or (not partial and len(rows)!=len(expected)):raise ValueError('Incomplete/extra rows')
    errors=[]
    for row,(case,arm) in zip(rows,expected):
        if row['qa_id']!=case['qa_id'] or row['arm']!=arm or row['prompt_sha256']!=sha(case[arm]) or row['removed_quotes']!=(case['removed'] if arm=='full_context' else 0):raise ValueError('Identity/prompt changed')
        if row['status']=='error':errors.append(dict(qa_id=row['qa_id'],arm=arm,error=row.get('error')));continue
        if row['status']!='ok':raise ValueError('Unknown row status')
        parsed=TypedEntailments.model_validate(row['response']);checks=json.loads(case[arm].split('<checks>\n')[1].split('\n</checks>')[0])
        if len(parsed.checks)!=len(checks) or {c.claim_id for c in parsed.checks}!={c['claim_id'] for c in checks}:raise ValueError('Check identities changed')
    changes=[];complete=0
    for i in range(0,len(rows)-1,2):
        arms={r['arm']:r for r in rows[i:i+2]};complete+=1
        if any(r['status']!='ok' for r in arms.values()):continue
        a,b=({c['claim_id']:c['verdict'] for c in arms[k]['response']['checks']} for k in plan['arms'])
        changes.extend(dict(qa_id=rows[i]['qa_id'],claim_id=k,baseline=a[k],full_context=b[k]) for k in a if a[k]!=b[k])
    return dict(partial=len(rows)!=len(expected),expected_pairs=len(cases),complete_pairs=complete,errors=errors,changed_verdicts=len(changes),changes=changes,caveat=plan['caveat'])
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--replay',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--allow-partial',action='store_true');a=p.parse_args();r=summarize(a.replay,a.allow_partial);a.output.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k not in ['changes','caveat']}))
