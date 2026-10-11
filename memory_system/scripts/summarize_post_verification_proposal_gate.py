"""Audit exact frozen proposal and full-context verifier gate ablation."""
import argparse,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import answer_choice as ac
from probe_post_verification_citations import make_cases,attach
from probe_question_source_filter import sha
from summarize_reader_replay import read_complete

def summarize(path,partial=False):
    plan=json.loads(path.with_suffix('.plan.json').read_text());source=Path(plan['proposals'])
    if hashlib.sha256(source.read_bytes()).hexdigest()!=plan['proposals_sha256']:raise ValueError('Proposals changed')
    if plan['arms']!=['gated','literal_only'] or plan['rows']!=28:raise ValueError('Plan changed')
    oldplan=json.loads(source.with_suffix('.plan.json').read_text());cases={c['qa_id']:c for c in make_cases(Path(oldplan['run']),Path(oldplan['prepared']),Path(oldplan['replay']))}
    proposals=[r for r in read_complete(source) if r['arm']=='supplemented']
    expected=[(old,a) for i,old in enumerate(proposals) for a in (plan['arms'] if i%2==0 else list(reversed(plan['arms'])))];rows=read_complete(path,partial)
    if len(rows)>len(expected) or (not partial and len(rows)!=len(expected)):raise ValueError('Incomplete/extra rows')
    errors=[]
    for row,(old,arm) in zip(rows,expected):
        c=cases[old['qa_id']];proposal=old['calls'][0]['response'];prompt,added,rejected=attach(c,proposal,use_proposal_verdict=arm=='gated')
        if (row['qa_id'],row['repetition'],row['arm'])!=(old['qa_id'],old['repetition'],arm):raise ValueError('Row identity/order changed')
        if row['prompt']!=prompt or row['prompt_sha256']!=sha(prompt) or row['added']!=added or row['rejected']!=rejected or row['proposal_sha256']!=sha(json.dumps(proposal,sort_keys=True)):raise ValueError('Prompt/source/proposal changed')
        if row['status']=='error':errors.append(dict(qa_id=row['qa_id'],repetition=row['repetition'],arm=arm));continue
        parsed=ac.TypedEntailments.model_validate(row['response'])
        if len(parsed.checks)!=len(c['checks']) or {x.claim_id for x in parsed.checks}!={x['claim_id'] for x in c['checks']}:raise ValueError('Check identities changed')
        if row['verdicts']!={x.claim_id:x.verdict for x in parsed.checks if x.claim_id in c['targets']}:raise ValueError('Verdicts changed')
    transitions=[]
    for i in range(0,len(rows)-1,2):
        arms={r['arm']:r for r in rows[i:i+2]}
        if any(r['status']!='ok' for r in arms.values()):continue
        for cid,v in arms['gated']['verdicts'].items():transitions.append(dict(qa_id=rows[i]['qa_id'],repetition=rows[i]['repetition'],claim_id=cid,gated=v,literal_only=arms['literal_only']['verdicts'][cid],added_ids=[x['source_id'] for x in arms['literal_only']['added'] if x['claim_id']==cid]))
    return dict(partial=len(rows)!=len(expected),rows=len(rows),expected_rows=len(expected),errors=errors,changed=sum(t['gated']!=t['literal_only'] for t in transitions),transitions=transitions,caveat=plan['scope'])
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--replay',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--allow-partial',action='store_true');a=p.parse_args();r=summarize(a.replay,a.allow_partial);a.output.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k not in ['transitions','caveat']}))
