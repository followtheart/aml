"""Audit isolated CE queries and reproduce actual fine selection with reservations."""
import argparse,ast,copy,csv,hashlib,json,math,sys
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import cascade_rerank,config
from probe_ce_isolated_options import queries_for,round_robin
from probe_ce_source_units import records
from audit_https_funnel import norm
from inventory_snippet_provenance import parse_snippet

def audit(probe):
    plan=json.loads(probe.with_suffix('.plan.json').read_text());prepared=Path(plan['prepared']);run=Path(plan['run'])
    assert hashlib.sha256(prepared.read_bytes()).hexdigest()==plan['prepared_sha256']
    assert hashlib.sha256(Path('scripts/probe_ce_isolated_options.py').read_bytes()).hexdigest()==plan['script_sha256']
    probes=list(records(probe));assert [p['qa_id'] for p in probes]==plan['qa_ids']
    by_search={p['search_id']:p for p in probes};assert len(by_search)==len(probes)
    convs=list(records(prepared));qa={str(q['id']):q for c in convs for q in c['qa']};original_users=[norm(m['content']) for c in convs for session in c['sessions'] for m in session if m['role']=='user']
    assert set(qa)==set(plan['qa_ids'])
    gold=list(csv.DictReader(Path('data/raw/personamem-v2.csv').open()));rows=[]
    for s in records(run/'server-search-debug.jsonl'):
        if s.get('search_id') not in by_search:continue
        p=by_search[s['search_id']];ids=s['cascade']['coarse']['selected_ids'];pool={c['id']:c for c in s['ranked']};coarse=[pool[i] for i in ids]
        assert s['query']==qa[p['qa_id']]['question'] and s['options']==qa[p['qa_id']]['options']
        assert ids==p['candidate_ids'];assert p['document_hashes']==[hashlib.sha256(c.get('_rank_text',c['content']).encode()).hexdigest() for c in coarse]
        expected_queries=queries_for(s);assert {a['name']:a['query'] for a in p['arms']}==expected_queries and len(p['arms'])==len(expected_queries)
        assert [a['name'] for a in p['arms']]==list(expected_queries),'Arm order changed'
        for a in p['arms']:
            assert len(a['scores'])==len(ids) and all(type(v) in [int,float] and math.isfinite(v) for v in a['scores'])
            assert a['order']==[ids[i] for i in sorted(range(len(ids)),key=lambda i:-a['scores'][i])]
        orders=[a['order'] for a in p['arms'][1:]];offset=plan['qa_ids'].index(p['qa_id'])%len(orders);assert p['option_offset']==offset
        assert p['aggregate_order']==round_robin(orders[offset:]+orders[:offset])
        requirements=[{k:r[k] for k in ['id','text','origin']} for r in s['coverage_manifest']['coverage']['requirements']]
        settings=s['versions']['settings'];assert s['cascade']['cross_encoder']['status']=='ok'
        fused={r['candidate_id'] for r in s['cascade']['fine']['selection']['reservations'] if 'fused_head' in r['reasons']}
        values={'archived':[c['_ce_score'] for c in coarse],'fresh_control':p['arms'][0]['scores'],'isolated':[len(ids)-p['aggregate_order'].index(i) for i in ids]};selections={}
        for name,scores in values.items():
            candidates=copy.deepcopy(coarse)
            for c,value in zip(candidates,scores):c['_ce_score']=value;c['_final']=value
            selection={}
            with patch.multiple(config,**settings):chosen=cascade_rerank.shortlist(candidates,settings['CASCADE_FINE_LIMIT'],requirements,lambda c:(c['_ce_score'],c.get('_fused',0)),fused_ids=fused,trace=selection)
            selections[name]={'ids':[c['id'] for c in chosen],'selection':selection}
        assert selections['archived']['ids']==s['cascade']['fine']['selected_ids'],'Archived fine selection mismatch'
        meta=gold[int(p['qa_id'])];assert norm(ast.literal_eval(meta['user_query'])['content'])==norm(s['query'])
        turns=[norm(m['content']) for m in parse_snippet(meta['related_conversation_snippet']) if m['role']=='user'];turns=[t for t in turns if t and any(t in original for original in original_users)]
        carriers=[c['id'] for c in coarse if any(t in norm(c.get('_rank_text',c['content'])) for t in turns)]
        rows.append({'qa_id':p['qa_id'],'carriers':carriers,'arms':selections,'transport':{name:bool(set(carriers)&set(a['ids'])) for name,a in selections.items()}})
    assert len(rows)==len(probes)
    return {'status':'passed','questions':len(rows),'calls':sum(a['calls'] for p in probes for a in p['arms']),'coverage':{name:sum(r['transport'][name] for r in rows) for name in ['archived','fresh_control','isolated']},'gains':[r['qa_id'] for r in rows if r['transport']['isolated'] and not r['transport']['fresh_control']],'losses':[r['qa_id'] for r in rows if r['transport']['fresh_control'] and not r['transport']['isolated']],'rows':rows,'caveat':'Full query/document identity and complete scores validated; archived fine IDs reproduced. Fresh single-query control compared with rotated round-robin isolated-option rankings through actual reservations/capacity. Whole original user-turn containment is only a transport proxy, not option truth. Expanded CE budget and no listwise/reader/HTTPS accuracy measured.'}
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--probe',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=audit(a.probe)
    with a.output.open('x') as f:json.dump(r,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in r.items() if k not in ['rows','caveat']}))
