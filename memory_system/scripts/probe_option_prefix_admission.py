"""Offline fine-reservation probe: initial personal condition versus full premise."""
import argparse,copy,csv,json,re,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import cascade_rerank as ranking
from audit_https_funnel import norm
from inventory_snippet_provenance import parse_snippet
from probe_question_source_filter import load_cohort
from replay_query_context import records

def prefix_requirements(requirements):
    result=copy.deepcopy(requirements)
    for r in result:
        text=r['text'];prefix=text.split(',',1)[0]
        if r['id'].startswith('option:') and ',' in text and re.match(r'^(since|given|because|as)\b',text,re.I) and len(prefix.split())>=5:r['text']=prefix
    return result

def compare(search):
    trace=search['cascade'];settings=search['versions']['settings'];by={c['id']:c for c in search['ranked']};coarse=[by[cid] for cid in trace['coarse']['selected_ids']]
    if trace['cross_encoder']['status']!='ok' or any(c.get('_ce_score') is None for c in coarse):raise ValueError('Require complete CE')
    req=[{k:r[k] for k in ['id','text','origin']} for r in search['coverage_manifest']['coverage']['requirements']]
    fused={r['candidate_id'] for r in trace['fine']['selection']['reservations'] if 'fused_head' in r['reasons']}
    arms={}
    for arm,requirements in [('original',req),('prefix',prefix_requirements(req))]:
        selection={};chosen=ranking.shortlist(coarse,settings['CASCADE_FINE_LIMIT'],requirements,lambda c:(c['_ce_score'],c.get('_fused',0)),fused_ids=fused,trace=selection)
        arms[arm]=dict(ids=[c['id'] for c in chosen],reservations=selection['reservations'])
    if arms['original']['ids']!=trace['fine']['selected_ids']:raise ValueError('Baseline not reproduced')
    return dict(**arms,changed_requirements=[dict(id=a['id'],original=a['text'],prefix=b['text']) for a,b in zip(req,prefix_requirements(req)) if a!=b])

def main(args):
    jobs=json.loads(args.cohorts.read_text());gold=list(csv.DictReader(Path('data/raw/personamem-v2.csv').open()));rows=[]
    for job in jobs:
        searches={s['search_id']:s for s in records(Path(job['run'])/'server-search-debug.jsonl')}
        for r,q,p,s,c in load_cohort(Path(job['run']),Path(job['prepared'])):
            search=searches[r['search_id']];result=compare(search)
            # Gold is used only after both deterministic selections, for offline transport measurement.
            turns=[norm(m['content']) for m in parse_snippet(gold[int(r['qa_id'])]['related_conversation_snippet']) if m['role']=='user']
            carriers=[c['id'] for c in search['ranked'] if any(t and t in norm(c.get('_rank_text','')) for t in turns)]
            rows.append(dict(cohort=job['name'],qa_id=r['qa_id'],carriers=carriers,**result,transport={a:bool(set(carriers)&set(result[a]['ids'])) for a in ['original','prefix']}))
    report=dict(questions=len(rows),all_baselines_exact=True,changed_selections=sum(r['original']['ids']!=r['prefix']['ids'] for r in rows),transport={a:sum(r['transport'][a] for r in rows) for a in ['original','prefix']},wins=[r['qa_id'] for r in rows if r['transport']['prefix'] and not r['transport']['original']],losses=[r['qa_id'] for r in rows if r['transport']['original'] and not r['transport']['prefix']],rows=rows,caveat='Offline fine selection only. CE scores, coarse pool, reservations and caps frozen; only leading personal-condition text used for option topic reservation. Whole user snippet containment is a proxy, not role-verified truth or final answer accuracy. No listwise/packet/reader calls; no default changes.')
    with args.output.open('x') as f:f.write(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['rows','caveat']}))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cohorts',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
