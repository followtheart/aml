"""Rebuild full-history paired prompts and audit complete outcomes."""
import argparse,hashlib,json
from pathlib import Path
from probe_full_history_reader import GUIDANCE,histories,make_prompt
from probe_question_source_filter import load_cohort,sha
from summarize_reader_replay import read_complete
from app import answer_choice,eval_scoring

def summarize(path,allow_partial=False):
    plan=json.loads(path.with_suffix('.plan.json').read_text());prepared=Path(plan['prepared'])
    if hashlib.sha256(prepared.read_bytes()).hexdigest()!=plan['prepared_sha256']:raise ValueError('Input changed')
    if plan['arms']!=['retrieved','full_history'] or plan['guidance']!=GUIDANCE:raise ValueError('Policy changed')
    cohort=load_cohort(Path(plan['run']),prepared);history=histories(prepared)
    if plan['questions']!=[r['qa_id'] for r,*_ in cohort]:raise ValueError('Population changed')
    if plan['history']!={cid:dict(messages=len(h),sha256=sha(json.dumps(h,ensure_ascii=False))) for cid,h in history.items()}:raise ValueError('History changed')
    expected=[(data,arm) for i,data in enumerate(cohort) for arm in (plan['arms'] if i%2==0 else list(reversed(plan['arms'])))]
    rows=read_complete(path,allow_partial)
    if len(rows)>len(expected) or (not allow_partial and len(rows)!=len(expected)):raise ValueError('Incomplete/extra rows')
    for row,(data,arm) in zip(rows,expected):
        r,qa,packet,sources,constraints=data
        if tuple(row[k] for k in ['qa_id','conversation_id','packet_hash','arm'])!=(r['qa_id'],r['conversation_id'],r['packet_hash'],arm):raise ValueError('Identity/order changed')
        ss,cc=answer_choice.build_catalog(packet)
        if ss!=sources or list(cc.values())!=constraints:raise ValueError('Catalog changed')
        prompt,context=make_prompt(qa,packet,sources,constraints,history[r['conversation_id']] if arm=='full_history' else [])
        if (row['prompt_sha256'],row['context_sha256'],row['prompt_bytes'])!=(sha(prompt),sha(context),len(prompt.encode())):raise ValueError('Prompt changed')
        response=row.get('response');valid=isinstance(response,dict) and set(response)=={'answer'} and isinstance(response['answer'],str) and response['answer'] in answer_choice.option_map(qa['options'])
        if valid:
            if row['status']!='ok' or row.get('prediction')!=response['answer'] or row['score']!=eval_scoring.choice_score(response['answer'],qa['gold_labels'],qa['qa_type']):raise ValueError('Answer/score changed')
        elif row['status']!='error' or row['score']!=0 or not row.get('error_type'):raise ValueError('Failure must score zero')
        if row['calls'] not in (0,1):raise ValueError('Extra calls')
    pairs=[]
    for i in range(0,len(rows)-1,2):
        by={r['arm']:r for r in rows[i:i+2]}
        pairs.append(dict(qa_id=rows[i]['qa_id'],**{a:{k:by[a].get(k) for k in ['status','prediction','score']} for a in plan['arms']}))
    return dict(partial=len(rows)!=len(expected),complete_pairs=len(pairs),expected_pairs=len(cohort),scores={a:sum(p[a]['score'] for p in pairs) for a in plan['arms']},wins=[p['qa_id'] for p in pairs if p['full_history']['score']>p['retrieved']['score']],losses=[p['qa_id'] for p in pairs if p['full_history']['score']<p['retrieved']['score']],errors=[{k:r.get(k) for k in ['qa_id','arm','error_type']} for r in rows if r['status']=='error'],recorded_calls={a:sum(r['calls'] for r in rows if r['arm']==a) for a in plan['arms']},pairs=pairs,caveat=plan['caveat'])

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--replay',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--allow-partial',action='store_true');a=p.parse_args()
    result=summarize(a.replay,a.allow_partial);a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['pairs','caveat']}))
