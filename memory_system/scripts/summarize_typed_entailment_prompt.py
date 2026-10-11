"""Audit frozen-prefix typed-prompt full-reader comparisons."""
import argparse,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import eval_scoring
from probe_question_source_filter import load_cohort,sha
from summarize_reader_replay import read_complete
from typed_entailment_prompt import normalize,REPLACEMENTS

def summarize(path,partial=False):
    plan=json.loads(path.with_suffix('.plan.json').read_text());prepared=Path(plan['prepared'])
    if hashlib.sha256(prepared.read_bytes()).hexdigest()!=plan['prepared_sha256']:raise ValueError('Input changed')
    if plan['replacements']!=[list(x) for x in REPLACEMENTS] or plan['arms']!=['baseline','typed']:raise ValueError('Policy changed')
    cohort=load_cohort(Path(plan['run']),prepared)
    if plan['questions']!=[r['qa_id'] for r,*_ in cohort]:raise ValueError('Incomplete plan')
    expected=[(data,arm) for i,data in enumerate(cohort) for arm in (['baseline','typed'] if i%2==0 else ['typed','baseline'])]
    rows=read_complete(path,partial)
    if len(rows)>len(expected) or (not partial and len(rows)!=len(expected)):raise ValueError('Incomplete/extra rows')
    errors=[]
    for row,(data,arm) in zip(rows,expected):
        old,qa,packet,sources,constraints=data
        if (row['qa_id'],row['conversation_id'],row['arm'],row['packet_hash'])!=(old['qa_id'],old['conversation_id'],arm,old['packet_hash']) or row['frozen_verified'] is not True:raise ValueError('Identity/order changed')
        d=row['diagnostics']
        if d['answer_source_catalog']!=list(sources.values()) or d['answer_constraint_catalog']!=constraints:raise ValueError('Source catalog changed')
        if row['score']!=eval_scoring.choice_score(row['prediction'],qa['gold_labels'],qa['qa_type']):raise ValueError('Score changed')
        boundary=next(i for i,c in enumerate(old['answer_calls']) if c['stage']=='eval.choice_entailment')
        if row['fixed_prefix_calls']!=boundary:raise ValueError('Prefix boundary changed')
        prefix=d['answer_calls'][:boundary]
        if len(prefix)!=boundary or any((a['stage'],a['prompt_sha256'],a.get('response'))!=(b['stage'],b['prompt_sha256'],b.get('response')) for a,b in zip(prefix,old['answer_calls'][:boundary])):raise ValueError('Frozen responses changed')
        first=True
        for c in row['live_calls']:
            if sha(c['prompt'])!=c['prompt_sha256']:raise ValueError('Prompt hash changed')
            if c['stage']=='eval.choice_entailment':
                original=c['original_prompt']
                if sha(original)!=c['original_prompt_sha256'] or c['prompt']!=(normalize(original) if arm=='typed' else original):raise ValueError('Verifier changed more than contract')
                if first:
                    cap=plan['captured'][row['qa_id']]
                    if sha(original)!=cap['prompt_sha256'] or sha(normalize(original))!=cap['typed_prompt_sha256']:raise ValueError('Initial verifier inputs changed')
                    first=False
            elif c['prompt_sha256']!=c['original_prompt_sha256']:raise ValueError('Non-verifier prompt modified')
        if first:raise ValueError('Missing verifier call')
        if d.get('answer_status')=='error':errors.append(dict(qa_id=row['qa_id'],arm=arm))
    pairs=[]
    for i in range(0,len(rows)-1,2):
        arms={r['arm']:r for r in rows[i:i+2]}
        pairs.append(dict(qa_id=rows[i]['qa_id'],**{a:dict(prediction=arms[a]['prediction'],score=arms[a]['score']) for a in plan['arms']}))
    return dict(partial=len(rows)!=len(expected),expected_pairs=len(cohort),complete_pairs=len(pairs),errors=errors,baseline=sum(p['baseline']['score'] for p in pairs),typed=sum(p['typed']['score'] for p in pairs),wins=[p['qa_id'] for p in pairs if p['typed']['score']>p['baseline']['score']],losses=[p['qa_id'] for p in pairs if p['typed']['score']<p['baseline']['score']],live_attempts={a:sum(len(r['live_calls']) for r in rows if r['arm']==a) for a in plan['arms']},pairs=pairs,caveat=plan['caveat'])
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--replay',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--allow-partial',action='store_true');a=p.parse_args();r=summarize(a.replay,a.allow_partial);a.output.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k not in ['pairs','caveat']}))
