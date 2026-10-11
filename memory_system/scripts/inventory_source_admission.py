"""Offline exact-user-turn admission inventory; never a semantic-support verdict."""
import argparse,ast,csv,hashlib,json
from collections import Counter
from pathlib import Path
from audit_https_funnel import norm
from inventory_snippet_provenance import parse_snippet
from probe_question_source_filter import load_cohort
from replay_query_context import records

def carrier_stage(candidate,trace):
    cid=candidate['id']
    if cid not in trace['coarse']['selected_ids']:return 'coarse_limit'
    if cid not in trace['fine']['selected_ids']:return 'fine_limit'
    if cid not in trace['listwise']['candidate_ids']:return 'listwise_admission'
    if cid not in trace['listwise']['selected_ids']:return 'listwise_selection'
    return 'packet_or_catalog'

ORDER=['coarse_limit','fine_limit','listwise_admission','listwise_selection','packet_or_catalog']
def classify(turns,originals,search,sources,constraints):
    valid=[t for t in turns if t and any(t in norm(m['content']) for m in originals if m['role']=='user')]
    if not valid:return dict(stage='no_original_user_turn',carriers=[])
    if any(t in norm(s['text']) for t in valid for s in sources.values() if (s.get('declared') or s['role']) in ['user','persona']):return dict(stage='delivered_full_user_turn',carriers=[])
    if any(t in norm(s['text']) for t in valid for s in constraints):return dict(stage='delivered_constraint',carriers=[])
    carriers=[];render_missing=[];render_missing_details=[]
    for c in search['ranked']:
        if not any(t in norm(c.get('content','')) or t in norm(c.get('_rank_text','')) for t in valid):continue
        if not any(t in norm(c.get('_rank_text','')) for t in valid):
            render_missing.append(c['id'])
            fine=next((d for d in search['cascade']['fine']['selection']['decisions'] if d['id']==c['id']),{})
            render_missing_details.append(dict(id=c['id'],stage=carrier_stage(c,search['cascade']),fine_rank=fine.get('rank'),fine_reason=fine.get('reason')))
            continue
        stage=carrier_stage(c,search['cascade']);fine=next((d for d in search['cascade']['fine']['selection']['decisions'] if d['id']==c['id']),{})
        carriers.append(dict(id=c['id'],stage=stage,fine_rank=fine.get('rank'),fine_score=fine.get('score'),fine_reason=fine.get('reason')))
    stage=max((c['stage'] for c in carriers),key=ORDER.index) if carriers else ('source_rendering' if render_missing else 'no_full_turn_in_ranked_pool')
    return dict(stage=stage,carriers=carriers,render_missing=render_missing,render_missing_details=render_missing_details)

def inventory(jobs,gold_path):
    gold=list(csv.DictReader(gold_path.open()));rows=[]
    for job in jobs:
        run=Path(job['run']);prepared=Path(job['prepared']);convs={c['conversation_id']:c for c in records(prepared)};searches={s['search_id']:s for s in records(run/'server-search-debug.jsonl')}
        for r,q,p,s,c in load_cohort(run,prepared):
            meta=gold[int(r['qa_id'])];conv=convs[r['conversation_id']]
            if norm(ast.literal_eval(meta['user_query'])['content'])!=norm(q['question']) or meta['correct_answer']!=q['answer'] or conv['source_history'] not in [meta['chat_history_32k_link'],meta['chat_history_128k_link']]:raise ValueError('Metadata identity mismatch')
            turns=[norm(m['content']) for m in parse_snippet(meta['related_conversation_snippet']) if m['role']=='user'];search=searches[r['search_id']]
            result=classify(turns,[m for session in conv['sessions'] for m in session],search,s,c)
            rows.append(dict(cohort=job['name'],qa_id=r['qa_id'],score=r['score'],search_id=r['search_id'],ce_status=search['cascade']['cross_encoder']['status'],**result))
    return dict(questions=len(rows),stages=dict(Counter(r['stage'] for r in rows)),wrong_stages=dict(Counter(r['stage'] for r in rows if not r['score'])),rows=rows,gold_sha256=hashlib.sha256(gold_path.read_bytes()).hexdigest(),caveat='Complete literal user-turn containment only; fragmented turns, absent full rank text and attribution ambiguity need separate review. Most advanced full-turn carrier is reported. render_missing_details separately follows downstream ranking of raw carriers whose visible rank text lacks a complete turn; useful fragments may remain, so source_rendering is not proof of semantic loss. Fine limits include reservation effects; listwise admission may involve byte/cap budgets. These stages describe transport, not whether gold is entailed or changing admission fixes answers. Offline labels never enter runtime.')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cohorts',type=Path,required=True);p.add_argument('--gold-csv',type=Path,default=Path('data/raw/personamem-v2.csv'));p.add_argument('--output',type=Path,required=True);a=p.parse_args();result=inventory(json.loads(a.cohorts.read_text()),a.gold_csv)
    with a.output.open('x') as f:f.write(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['rows','caveat']}))
