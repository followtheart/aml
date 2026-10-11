"""Audit source-order-only recovery proposals against the complete HTTPS archive."""
import argparse,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import answer_choice as ac
from probe_question_source_filter import load_cohort,sha
from probe_recovery_source_order import reverse_cards
from summarize_reader_replay import read_complete

def summarize(path,partial=False):
    plan=json.loads(path.with_suffix('.plan.json').read_text());prepared=Path(plan['prepared'])
    if hashlib.sha256(prepared.read_bytes()).hexdigest()!=plan['prepared_sha256'] or plan['arms']!=['original','reversed']:raise ValueError('Plan/input changed')
    cohort=load_cohort(Path(plan['run']),prepared);archived={r['qa_id']:r for r,*_ in cohort};sources={r['qa_id']:s for r,q,p,s,c in cohort}
    if plan['all_questions']!=list(archived):raise ValueError('Incomplete preflight cohort')
    expected_ids=[qid for qid,r in archived.items() if any(c['stage']=='eval.choice_citation_recovery' for c in r['answer_calls'])]
    if [c['qa_id'] for c in plan['cases']]!=expected_ids:raise ValueError('Recovery population changed')
    for c in plan['cases']:
        old=next(x for x in archived[c['qa_id']]['answer_calls'] if x['stage']=='eval.choice_citation_recovery')
        if sha(c['prompt'])!=old['prompt_sha256']:raise ValueError('Original recovery prompt changed')
    expected=[(c,a) for i,c in enumerate(plan['cases']) for a in (plan['arms'] if i%2==0 else list(reversed(plan['arms'])))];rows=read_complete(path,partial)
    if len(rows)>len(expected) or (not partial and len(rows)!=len(expected)):raise ValueError('Incomplete/extra rows')
    errors=[];details=[]
    for row,(c,arm) in zip(rows,expected):
        prompt=c['prompt'] if arm=='original' else reverse_cards(c['prompt'])
        if (row['qa_id'],row['arm'],row['prompt_sha256'])!=(c['qa_id'],arm,sha(prompt)):raise ValueError('Identity/order/prompt changed')
        if row['status']=='error':errors.append(dict(qa_id=row['qa_id'],arm=arm,error=row.get('error')));continue
        targets={x['claim_id']:x for x in json.loads(c['prompt'].split('\nClaims:\n')[1].split('\nOriginal source cards:\n')[0])}
        parsed=ac.normalize_citation_recovery_matches(row['response'],list(targets))
        if row['proposals']!=[x.model_dump() for x in parsed]:raise ValueError('Proposals changed')
        for p in parsed:
            checks=[ac._check_citation(ref,targets[p.claim_id]['text'],sources[c['qa_id']],p.premise_type) for ref in p.citations]
            details.append(dict(qa_id=c['qa_id'],arm=arm,claim_id=p.claim_id,supported=p.supported,source_ids=[x.source_id for x in p.citations],valid_anchor_ids=[x['source_id'] for x in checks if x['valid'] and x.get('anchor') and not x.get('strength_gap')],invalid_quotes=sum('quote_not_in_source' in x['validation_errors'] for x in checks)))
    grouped={}
    for d in details:grouped.setdefault((d['qa_id'],d['claim_id']),{})[d['arm']]=d
    changes=[dict(qa_id=k[0],claim_id=k[1],**v) for k,v in grouped.items() if set(v)==set(plan['arms']) and any(v['original'][f]!=v['reversed'][f] for f in ['supported','source_ids','valid_anchor_ids'])]
    return dict(partial=len(rows)!=len(expected),all_questions=len(cohort),expected_pairs=len(plan['cases']),complete_pairs=len(rows)//2,errors=errors,changed_proposals=len(changes),changes=changes,details=details,caveat=plan['caveat'])
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--replay',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--allow-partial',action='store_true');a=p.parse_args();r=summarize(a.replay,a.allow_partial);a.output.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k not in ['details','changes','caveat']}))
