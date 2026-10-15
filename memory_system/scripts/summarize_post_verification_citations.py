"""Reconstruct exact source augmentation and audit all post-verification pairs."""
import argparse,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import answer_choice as ac
from probe_post_verification_citations import make_cases,attach,INSTRUCTION
from probe_question_source_filter import sha
from summarize_reader_replay import read_complete

def summarize(path,partial=False):
    plan=json.loads(path.with_suffix('.plan.json').read_text())
    for key in ['prepared','replay']:
        if hashlib.sha256(Path(plan[key]).read_bytes()).hexdigest()!=plan[key+'_sha256']:raise ValueError('Input changed')
    if plan['instruction']!=INSTRUCTION or plan['repetitions']!=2:raise ValueError('Policy changed')
    cases=make_cases(Path(plan['run']),Path(plan['prepared']),Path(plan['replay']))
    if plan['cases']!=[dict(qa_id=c['qa_id'],target_ids=list(c['targets'])) for c in cases]:raise ValueError('Target selection changed')
    expected=[(c,rep+1,a) for rep in range(2) for i,c in enumerate(cases) for a in (['baseline','supplemented'] if (i+rep)%2==0 else ['supplemented','baseline'])]
    rows=read_complete(path,partial)
    if len(rows)>len(expected) or (not partial and len(rows)!=len(expected)):raise ValueError('Incomplete/extra rows')
    errors=[]
    for row,(c,rep,arm) in zip(rows,expected):
        if (row['qa_id'],row['repetition'],row['arm'],row['target_ids'])!=(c['qa_id'],rep,arm,list(c['targets'])):raise ValueError('Identity/order changed')
        prompt=c['prompt'];offset=0;failed=False
        for call in row['calls']:
            if sha(call['prompt'])!=call['prompt_sha256']:raise ValueError('Prompt hash changed')
        if arm=='supplemented':
            request=INSTRUCTION+json.dumps(dict(claims=[dict(claim_id=cid,text=t['text'],existing_citations=t['citations']) for cid,t in c['targets'].items()],original_sources=ac._cards({k:s for k,s in c['sources'].items() if (s.get('declared') or s['role']) in ['user','persona']}),forget_constraints=c['constraints']),ensure_ascii=False)
            call=row['calls'][0];offset=1
            if call['stage']!='diagnostic.post_verify.find' or call['prompt']!=request:raise ValueError('Source discovery input changed')
            if call['status']=='error':failed=True
            else:
                try:prompt,added,rejected=attach(c,call['response'])
                except (ValueError,KeyError,TypeError):failed=True
                if not failed and (row['added']!=added or row['rejected']!=rejected):raise ValueError('Source augmentation changed')
        elif row['added'] or row['rejected']:raise ValueError('Baseline source mutation')
        if failed:
            if len(row['calls'])!=offset:raise ValueError('Failure bypassed')
        else:
            if len(row['calls'])!=offset+1:raise ValueError('Missing/extra verifier')
            call=row['calls'][offset]
            if call['stage']!='diagnostic.post_verify.verify' or call['prompt']!=prompt:raise ValueError('Verifier sources changed')
            if call['status']=='error':failed=True
            else:
                try:
                    parsed=ac.TypedEntailments.model_validate(call['response'])
                    if len(parsed.checks)!=len(c['checks']) or {x.claim_id for x in parsed.checks}!={x['claim_id'] for x in c['checks']}:raise ValueError('Invalid check identities')
                except (ValueError,KeyError,TypeError):failed=True
                if not failed and row.get('verdicts')!={x.claim_id:x.verdict for x in parsed.checks if x.claim_id in c['targets']}:raise ValueError('Verdicts changed')
        if row['status']!=('error' if failed else 'ok'):raise ValueError('Row status inconsistent')
        if failed:errors.append(dict(qa_id=c['qa_id'],repetition=rep,arm=arm,error=row.get('error')))
    transitions=[]
    for i in range(0,len(rows)-1,2):
        a={r['arm']:r for r in rows[i:i+2]}
        if any(r['status']!='ok' for r in a.values()):continue
        for cid,v in a['baseline']['verdicts'].items():
            transitions.append(dict(qa_id=rows[i]['qa_id'],repetition=rows[i]['repetition'],claim_id=cid,baseline=v,supplemented=a['supplemented']['verdicts'][cid],added_source_ids=[x['source_id'] for x in a['supplemented']['added'] if x['claim_id']==cid]))
    return dict(partial=len(rows)!=len(expected),expected_rows=len(expected),rows=len(rows),errors=errors,transitions=transitions,changed=sum(x['baseline']!=x['supplemented'] for x in transitions),caveat=plan['caveat'])
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--replay',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--allow-partial',action='store_true');a=p.parse_args();r=summarize(a.replay,a.allow_partial);a.output.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k not in ['transitions','caveat']}))
