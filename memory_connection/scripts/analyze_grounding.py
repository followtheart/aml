"""Offline audit of frozen retrieval x expansion x reader experiments."""
import argparse
import collections
import importlib.util
import json
import math
import random
import sqlite3
import statistics
from datetime import datetime,timezone
from pathlib import Path
from .benchmark_metrics import file_hash,call_cost,percentile
from .grounding_benchmark import rows,ROOT
from memory_connection.app.event_builder import valid_spans
from memory_connection.app.schemas import Span


def temporal(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def paired(left,right):
    ids=sorted(set(left)&set(right));ds=[left[i]-right[i] for i in ids]
    wins=sum(x>0 for x in ds);losses=sum(x<0 for x in ds);n=wins+losses
    rng=random.Random(20261027);samples=[statistics.mean(rng.choices(ds,k=len(ds))) for _ in range(4000)] if ds else []
    return {'pairs':len(ds),'delta':statistics.mean(ds) if ds else None,'wins':wins,'losses':losses,
            'ties':len(ds)-n,'exact_paired_sign_p':min(1.,2*sum(math.comb(n,k) for k in range(min(wins,losses)+1))/2**n) if n else 1.,
            'history_bootstrap_95':[percentile(samples,.025),percentile(samples,.975)],
            'history_differences':dict(zip(ids,ds)),'small_sample':True}


def analyze(root):
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'));plan=json.loads((root/'plan.json').read_text(encoding='utf-8'))
    results=rows(root/'results.jsonl');evidence=rows(root/'evidence.jsonl');calls=rows(root/'calls.jsonl');prices=json.loads((root/'prices.snapshot.json').read_text())
    by_key={(r['arm'],r['reader'],r['conversation_id']):r for r in results};ekeys={(r['arm'],r['conversation_id']) for r in evidence}
    expected={(v+'/'+a,rd,c['conversation_id']) for c in plan for v in ('before','after') for a in ('C00','C10','C01','C11') for rd in ('v4','v5')}
    archive=root/'frozen-sources';source_base=archive if archive.exists() else Path('.')
    integrity={'expected_results':len(expected),'results':len(results),'evidence':len(evidence),
               'missing_results':sorted(expected-set(by_key)),'unexpected_results':sorted(set(by_key)-expected),
               'duplicate_results':len(results)-len(by_key),'duplicate_evidence':len(evidence)-len(ekeys),
               'missing_searches':sorted({(arm,cid) for arm,_,cid in expected}-ekeys),
               'changed_sources':[p for p,h in manifest['source_hashes'].items() if file_hash(source_base/p)!=h],
               'changed_before':[p for p,h in json.loads((root/'before-hashes.json').read_text()).items() if file_hash(root/'before'/p)!=h],
               'dataset_changed':file_hash('memory_connection/data/longmemeval_s_cleaned.json')!=manifest['dataset']['sha256'],
               'invalid_originals':[],'invalid_annotations':[],'invalid_reader_sources':[],
               'different_query_views':[],'different_coarse_orders':[],'lossless_adapter_errors':[]}
    modules={'before':temporal(root/'before/app/temporal.py','before_temporal'),
             'after':temporal(source_base/'memory_connection/app/temporal.py','after_temporal')}
    report={'status':json.loads((root/'status.json').read_text()),'integrity':integrity,
            'recorded_calls':len(calls),'actual_cost':call_cost(calls,prices),
            'failed_calls_by_phase':dict(collections.Counter(c.get('phase') for c in calls if c.get('status')!='ok')),
            'missing_usage_calls':sum(not c.get('usage_reported') for c in calls),'index_counts':{},'expansion_audit':{},
            'search_arms':{},'answers':{},'paired_end_to_end':{},'retrieval_with_fixed_reader':{},'reader_with_fixed_retrieval':{},
            'dual_comparisons':{},'nondegraded_comparisons':{},'question_pipeline':[],'case_changes':[]}
    report['resumptions']=rows(root/'resumptions.jsonl')
    if any(r.get('interrupted_inflight_usage_unknown') for r in report['resumptions']):
        report['actual_cost']['cost_complete']=False
    lookup={};expansion=collections.Counter()
    for path in root.glob('*-C[01].db'):
        conn=sqlite3.connect(f'file:{path.resolve().as_posix()}?mode=ro',uri=True)
        try:events=[json.loads(p) for (p,) in conn.execute('SELECT payload FROM events')]
        finally:conn.close()
        lookup.update({e['event_id']:e for e in events});report['index_counts'][path.stem]=len(events)
        if path.stem.endswith('C1'):
            expansion['events']+=len(events);expansion['derived_views']+=sum(len(e['views'])-1 for e in events)
            expansion['failed']+=sum(e['expansion_status']=='failed' for e in events)
            expansion['zero_derived']+=sum(len(e['views'])==1 for e in events)
    report['expansion_audit']=dict(expansion);convs={c['conversation_id']:c for c in plan}
    report['marked_message_recall_eligible']=sum(bool(c['qa'][0]['evidence']) for c in plan)
    integrity['incomplete_indices']=[c['conversation_id']+'-'+f for c in plan for f in ('C0','C1')
        if report['index_counts'].get(c['conversation_id']+'-'+f)!=sum(map(len,c['sessions']))]
    report['same_day_clock_anomalies']=[]
    for c in plan:
        qa=c['qa'][0];qdate=datetime.fromisoformat(qa['question_date'])
        later=[]
        for sid,session in zip(c['session_ids'],c['sessions']):
            date=datetime.fromtimestamp(session[0]['timestamp']/1000,timezone.utc)
            if date>qdate:later.append({'session_id':sid,'stated_at':date.isoformat(),
                'same_day':date.date()==qdate.date(),'answer_session':sid in qa['answer_session_ids']})
        if later:report['same_day_clock_anomalies'].append({'conversation_id':c['conversation_id'],
            'question_date':qa['question_date'],'later_sessions':later})
    all_sessions=[s for c in plan for s in c['session_ids']]
    integrity['duplicate_history_sessions']=len(all_sessions)-len(set(all_sessions))
    for c in plan:
        for original in c['originals']:
            if ''.join(c['source_texts'][mid] for mid in original['chunks'])!=original['text']:
                integrity['lossless_adapter_errors'].append([c['conversation_id'],original['session_id'],original['message_index']])
    diagnostics={};returned=0
    def mids(events):return {s['message_id'] for e in events for s in e['sources']}
    def sessions(events):return {s['session_id'] for e in events for s in e['sources']}
    def recall(g,found):return len(g&found)/len(g) if g else None
    for item in evidence:
        arm,cid=item['arm'],item['conversation_id'];c=convs[cid];qa=c['qa'][0];g=set(qa['evidence']);gs=set(qa['answer_session_ids'])
        response=item['response'];trace=response.get('trace',{});cs=trace.get('candidates',[]);ds=trace.get('decisions',[])
        candidates=[lookup[x['event_id']] for x in cs];pool=candidates[:16]
        accepted=[lookup[d['event_id']] for d in ds if d['verdict'] in ('relevant','partial','contradicting')]
        packet=item['packet'];found={s['message_id'] for p in packet for s in p['sources']};fs={s['session_id'] for p in packet for s in p['sources']}
        promoted=[lookup[x['event_id']] for x in cs[:16] if x.get('bridge_reserved')]
        pre=sorted(enumerate(cs),key=lambda pair:pair[1].get('pre_reservation_rank') or pair[0]+1)
        prepool=[lookup[x['event_id']] for _,x in pre[:16]]
        diagnostic={'arm':arm,'conversation_id':cid,'category':qa['category'],
                    'candidate_recall':recall(g,mids(candidates)),'pool_recall':recall(g,mids(pool)),
                    'accepted_recall':recall(g,mids(accepted)),'final_recall':recall(g,found),
                    'candidate_session_recall':recall(gs,sessions(candidates)), 'pool_session_recall':recall(gs,sessions(pool)),
                    'final_session_recall':recall(gs,fs),'returned':len(packet),'candidate_count':len(cs),
                    'promoted':len(promoted),'promoted_gold':sorted(g&mids(promoted)),
                    'promotion_new_gold':sorted(g&(mids(pool)-mids(prepool))),
                    'promotion_lost_gold':sorted(g&(mids(prepool)-mids(pool))),
                    'degradation':response.get('degradation',[]),'verification_errors':response.get('verification_errors',[]),
                    'search_errors':item['search_errors'],'search_seconds':item['search_seconds'],
                    'search_seconds_observed':item['search_seconds_observed'],
                    'coarse_order':[x['event_id'] for _,x in sorted(enumerate(cs),key=lambda pair:pair[1].get('coarse_rank') or pair[0]+1)],
                    'query_views':trace.get('query_views',[]),'final_gold':sorted(g&found),'gold':sorted(g),
                    'candidate_gold':sorted(g&mids(candidates)),'pool_gold':sorted(g&mids(pool)),
                    'accepted_gold':sorted(g&mids(accepted))}
        diagnostics[(arm,cid)]=diagnostic;report['question_pipeline'].append(diagnostic)
        for hit in response.get('data',[]):
            returned+=1;event=lookup[hit['id']]
            if hit['content']!=event['text'] or hit['sources']!=event['sources'] or not valid_spans([Span(**s) for s in hit['evidence_spans']],event):integrity['invalid_originals'].append([arm,cid,hit['id']])
            if hit.get('source_annotations')!=modules[arm.split('/')[0]].source_annotations(event['sources']):integrity['invalid_annotations'].append([arm,cid,hit['id']])
    integrity['returned_original_items']=returned
    report['dual_exclusive_evidence']=[]
    for version in ('before','after'):
        for c in plan:
            cid=c['conversation_id'];dual=diagnostics.get((version+'/C11',cid))
            single=[diagnostics.get((version+'/'+a,cid)) for a in ('C10','C01')]
            if not dual or not all(single):continue
            exclusive=set(dual['candidate_gold'])-set(single[0]['candidate_gold'])-set(single[1]['candidate_gold'])
            report['dual_exclusive_evidence'].append({'version':version,'conversation_id':cid,
                'candidate_exclusive':sorted(exclusive),
                'exclusive_outside_pool':sorted(exclusive-set(dual['pool_gold'])),
                'exclusive_removed_by_verification':sorted(exclusive&set(dual['pool_gold'])-set(dual['accepted_gold'])),
                'exclusive_final':sorted(exclusive&set(dual['final_gold'])),
                'final_extra_over_single_union':sorted(set(dual['final_gold'])-set(single[0]['final_gold'])-set(single[1]['final_gold']))})
    for (arm,cid),d in diagnostics.items():
        if not arm.startswith('after'):continue
        old=diagnostics.get((arm.replace('after','before'),cid))
        if not old:continue
        if d['query_views']!=old['query_views']:integrity['different_query_views'].append([arm,cid])
        if d['coarse_order']!=old['coarse_order']:integrity['different_coarse_orders'].append([arm,cid])
    for item in results:
        if item['reader']!='v5' or not item.get('reader_details'):continue
        detail=item['reader_details'];e=next(e for e in evidence if e['arm']==item['arm'] and e['conversation_id']==item['conversation_id'])
        original=[s for p in e['packet'] for s in p['sources']];refs={s['ref'] for s in detail['sources']}
        invalid=bool(set(detail['source_refs'])-refs)
        for s in detail['sources']:
            if not any(s['original_text']==x['content'] and s['role']==x['role'] and s['stated_at']==modules['after'].source_time(x)['stated_at'] for x in original):invalid=True
        if invalid:integrity['invalid_reader_sources'].append([item['arm'],item['conversation_id']])
    report['index_costs']={f:call_cost([c for c in calls if c['phase']=='add' and c['arm']==f],prices) for f in ('C0','C1')}
    for arm in sorted({e['arm'] for e in evidence}):
        ds=[d for (a,_),d in diagnostics.items() if a==arm]
        usage=[c for c in calls if c.get('arm')==arm and c['phase']=='search']
        prep=[c for c in calls if c['phase']=='query_prepare' and (arm.endswith('1') or c.get('component')=='original_embed')]
        report['search_arms'][arm]={'questions':len(ds),'pipeline':{m:statistics.mean(d[m] for d in ds if d[m] is not None) for m in
            ('candidate_recall','pool_recall','accepted_recall','final_recall','candidate_session_recall','pool_session_recall','final_session_recall') if any(d[m] is not None for d in ds)},
            'search_p50':percentile([d['search_seconds'] for d in ds],.5),'search_p95':percentile([d['search_seconds'] for d in ds],.95),
            'cost_with_shared_preparation':call_cost(usage+prep,prices),'degraded_questions':sum(bool(d['degradation']) for d in ds),
            'verification_error_questions':sum(bool(d['verification_errors']) for d in ds),'promoted_events':sum(d['promoted'] for d in ds),
            'promoted_gold_questions':sum(bool(d['promoted_gold']) for d in ds),'new_gold_questions':sum(bool(d['promotion_new_gold']) for d in ds),
            'lost_gold_questions':sum(bool(d['promotion_lost_gold']) for d in ds)}
    groups={}
    for (arm,reader,cid),r in by_key.items():groups.setdefault(arm+'/'+reader,{})[cid]=r['score']
    for key,group in groups.items():
        arm,reader=key.rsplit('/',1);rs=[r for r in results if r['arm']==arm and r['reader']==reader]
        report['answers'][key]={'n':len(rs),'correct':sum(group.values()),'accuracy':statistics.mean(group.values()),
            'errors':sum(bool(r['reader_errors']) for r in rs),'answer_p50':percentile([r['answer_seconds'] for r in rs],.5),
            'answer_judge_cost':call_cost([c for c in calls if c.get('arm')==arm and c.get('reader')==reader and c['phase']=='answer_judge'],prices),
            'categories':{cat:{'n':len(xs),'correct':sum(r['score'] for r in xs)} for cat in sorted({r['category'] for r in rs}) if (xs:=[r for r in rs if r['category']==cat])}}
    for arm in ('C00','C10','C01','C11'):
        report['paired_end_to_end'][arm]=paired(groups.get('after/'+arm+'/v5',{}),groups.get('before/'+arm+'/v4',{}))
        for reader in ('v4','v5'):report['retrieval_with_fixed_reader'][arm+'/'+reader]=paired(groups.get('after/'+arm+'/'+reader,{}),groups.get('before/'+arm+'/'+reader,{}))
        for version in ('before','after'):report['reader_with_fixed_retrieval'][version+'/'+arm]=paired(groups.get(version+'/'+arm+'/v5',{}),groups.get(version+'/'+arm+'/v4',{}))
        ids={cid for (a,cid),d in diagnostics.items() if a=='after/'+arm and not d['degradation'] and not diagnostics.get(('before/'+arm,cid),{}).get('degradation')}
        report['nondegraded_comparisons'][arm]=paired({k:v for k,v in groups.get('after/'+arm+'/v5',{}).items() if k in ids},
                                                     {k:v for k,v in groups.get('before/'+arm+'/v4',{}).items() if k in ids})
    for version,reader in (('before','v4'),('after','v5')):
        for arm in ('C00','C10','C01'):report['dual_comparisons'][version+'/C11-'+arm]=paired(groups.get(version+'/C11/'+reader,{}),groups.get(version+'/'+arm+'/'+reader,{}))
    anomaly_ids={c['conversation_id'] for c in report['same_day_clock_anomalies']}
    report['excluding_clock_anomalies']={a:paired(
        {cid:s for cid,s in groups.get('after/'+a+'/v5',{}).items() if cid not in anomaly_ids},
        {cid:s for cid,s in groups.get('before/'+a+'/v4',{}).items() if cid not in anomaly_ids}) for a in ('C00','C10','C01','C11')}
    for arm in ('C00','C10','C01','C11'):
        for c in plan:
            cid=c['conversation_id'];before=by_key.get(('before/'+arm,'v4',cid));after=by_key.get(('after/'+arm,'v5',cid))
            if before and after and before['score']!=after['score']:
                report['case_changes'].append({'arm':arm,'conversation_id':cid,'question':c['qa'][0]['question'],'gold':c['qa'][0]['answer'],
                    'before':before['prediction'],'after':after['prediction'],'scores':[before['score'],after['score']]})
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('run',nargs='?',default=ROOT);args=p.parse_args();root=Path(args.run)
    result=analyze(root);(root/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('question_pipeline','case_changes')},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
