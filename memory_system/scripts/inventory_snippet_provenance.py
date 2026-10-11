"""Offline role/format inventory of benchmark snippets; never infer fact support."""
import argparse,ast,csv,hashlib,json,re,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from probe_question_source_filter import load_cohort
from audit_https_funnel import norm

VERSION='snippet-role-format-v1'
def parse_snippet(value):
    try:result=json.loads(value)
    except ValueError:result=ast.literal_eval(value)
    if not isinstance(result,list) or any(not isinstance(m,dict) or not isinstance(m.get('content'),str) or m.get('role') not in ['user','assistant','system'] for m in result):raise ValueError('Invalid snippet')
    return result

def format_features(snippet):
    users=[m['content'] for m in snippet if m['role']=='user'];assistants=[m['content'] for m in snippet if m['role']=='assistant']
    short_requests=bool(users) and all(len(t)<=180 and '\n' not in t.strip() and re.search(r'\b(?:refine|polish|smoother|sound better|improve|translate|edit|rewrite|wording)\b',t,re.I) for t in users)
    draft=any(re.search(r'\boriginal.{0,40}\b(?:email|paragraph|text|message)\b',t.replace('_',' ').replace('*',''),re.I) for t in assistants)
    return dict(user_turns=len(users),assistant_turns=len(assistants),user_chars=sum(map(len,users)),assistant_chars=sum(map(len,assistants)),short_editing_requests_only=bool(short_requests),assistant_original_draft_marker=draft,review_candidate=bool(short_requests and draft))

def main(args):
    jobs=json.loads(args.cohorts.read_text());gold=list(csv.DictReader(args.gold_csv.open()));seen=set();rows=[]
    for job in jobs:
        for result,qa,packet,sources,constraints in load_cohort(Path(job['run']),Path(job['prepared'])):
            key=(result['conversation_id'],result['qa_id'])
            if key in seen:raise ValueError('Duplicate QA across cohorts')
            seen.add(key);metadata=gold[int(result['qa_id'])]
            prepared=next(c for c in map(json.loads,Path(job['prepared']).read_text().splitlines()) if c['conversation_id']==result['conversation_id'])
            query=ast.literal_eval(metadata['user_query'])['content']
            if norm(query)!=norm(qa['question']) or metadata['correct_answer']!=qa['answer'] or prepared['source_history'] not in [metadata['chat_history_32k_link'],metadata['chat_history_128k_link']]:raise ValueError('Gold identity mismatch')
            snippet=parse_snippet(metadata['related_conversation_snippet']);features=format_features(snippet)
            rows.append(dict(cohort=job['name'],qa_id=result['qa_id'],conversation_id=result['conversation_id'],score=result['score'],question=qa['question'],preference=metadata['preference'],**features,user_snippet_texts=[m['content'] for m in snippet if m['role']=='user'] if features['review_candidate'] else None))
    cohorts=[]
    for job in jobs:
        subset=[r for r in rows if r['cohort']==job['name']];flagged=[r for r in subset if r['review_candidate']]
        cohorts.append(dict(name=job['name'],questions=len(subset),short_editing_requests=sum(r['short_editing_requests_only'] for r in subset),assistant_original_drafts=sum(r['assistant_original_draft_marker'] for r in subset),review_candidates=len(flagged),candidate_ids=[r['qa_id'] for r in flagged],candidate_score=sum(r['score'] for r in flagged)))
    report=dict(version=VERSION,cohorts=cohorts,questions=len(rows),review_candidates=sum(r['review_candidate'] for r in rows),gold_sha256=hashlib.sha256(args.gold_csv.read_bytes()).hexdigest(),cohort_plan_sha256=hashlib.sha256(args.cohorts.read_bytes()).hexdigest(),criteria='Flag only short single-line user editing requests plus explicit assistant Original-document marker. Surface-format screen, not semantic classification.',caveat='Does not establish an accuracy ceiling, absence of corroboration elsewhere, or whether assistant draft is true. Role and authorship need manual review. Metadata remains offline and never enters answering.',rows=rows)
    with args.output.open('x') as f:f.write(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k in ['cohorts','questions','review_candidates']}))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cohorts',type=Path,required=True);p.add_argument('--gold-csv',type=Path,default=Path('data/raw/personamem-v2.csv'));p.add_argument('--output',type=Path,required=True);main(p.parse_args())
