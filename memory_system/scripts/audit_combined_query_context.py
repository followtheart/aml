"""Audit frozen combined-context HTTPS and its fixed-upstream transport effect.

Both arms make independent live Search calls. The reconstructed lexical packet
isolates deterministic transport effects, not causal answer-accuracy effects.
Reference turns are transport proxies, not semantic support. Costs are recorded
usage only; a cancelled/failed provider request may have unreported usage.
"""
import argparse
import ast
from collections import Counter
import copy
import csv
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

from analyze_overflow_https import provider_stage_costs
from audit_https_stages import reference_set_coverage
import audit_https_funnel as audit
from evaluation_model_guard import check_provider_log,check_versions


def records(path):return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main(args):
    root=args.run.resolve();plan=json.loads((root/'plan.json').read_text())
    exit_code=(root/'exit-code').read_text().strip()
    if exit_code not in ('0','1') or (exit_code!='0' and not args.include_reader_errors) or plan['toggle']!='PACKET_QUERY_CONTEXT_COMBINED':
        raise ValueError('Require completed combined-context trial; reader errors must be explicitly included')
    for name,value in plan['source_files'].items():
        if digest(root/'source'/name)!=value:raise ValueError('Frozen source changed')
    sys.path.insert(0,str(root/'source/memory_system'));sys.path.insert(0,str(root/'source/memory_system/scripts'))
    from app import answer_choice,config,evidence_packet
    from verify_combined_query_context import replay
    from probe_combined_query_context import prepare
    csv.field_size_limit(32*1024*1024)
    with args.gold_csv.open() as f:gold=list(csv.DictReader(f))
    reader_lines=(root/'reader-provider.log').read_text().splitlines()
    if any(not any(l.startswith('reader_arm='+a+' ') for a in ('control','treatment')) for l in reader_lines):
        raise ValueError('Unattributed reader metric')
    arms={}
    for arm in ('control','treatment'):
        run=root/arm;rs=records(run/'completed/data/results/personamem-v2-32k.jsonl')
        manifest=json.loads((run/'isolated-manifest.json').read_text())
        failures=[r for r in rs if r.get('error_stage')]
        if (len(rs)!=plan['questions'] or manifest.get('audit_exit_code')!=0
                or manifest.get('client_exit_code')!=int(bool(failures)) or 'server_exit_code' not in manifest
                or (failures and not args.include_reader_errors)
                or any(r['error_stage']!='answer' or r['score']!=0 or not r.get('search_id') for r in failures)):
            raise ValueError('Incomplete cohort or non-reader failure')
        proc=Path('/proc')/str(manifest['server_pid'])/'cmdline'
        if proc.exists() and str(root).encode() in proc.read_bytes():raise ValueError('Server is still running')
        results={r['search_id']:r for r in rs};raw={};rows=[]
        for event in records(run/'server-memory-debug.jsonl'):
            raw.setdefault(event['user_id'],{}).update({audit.identity(s):s for s in event['source_messages']})
        for t in records(run/'server-search-debug.jsonl'):
            check_versions(t['versions'],arm)
            enabled=t['versions']['settings']['PACKET_QUERY_CONTEXT_COMBINED']
            if enabled!=(arm=='treatment'):raise ValueError('Wrong effective toggle')
            r=results[t['search_id']]
            baseline,baseline_hash,baseline_manifest=replay(t,False)
            actual,actual_hash,actual_manifest=replay(t,enabled)
            if actual!=t['returned'] or actual_hash!=t['packet_hash'] or actual_hash!=r['packet_hash']:
                raise ValueError('Production packet did not reproduce')
            for key,value in actual_manifest.items():
                if value!=t['coverage_manifest'][key]:raise ValueError('Production manifest did not reproduce: '+key)
            baseline_trace=copy.deepcopy(t)
            baseline_trace.update(returned=baseline,packet_hash=baseline_hash)
            baseline_trace['cascade']['conversation_context']=baseline_manifest['conversation_context']
            baseline_trace['coverage_manifest'].update(baseline_manifest)
            with patch.multiple(config,**t['versions']['settings']),patch('socket.socket.connect',side_effect=AssertionError('Offline')):
                proposal=prepare(baseline_trace)['packets']['combined' if enabled else 'baseline']
            if actual!=[dict(p,packet_hash=evidence_packet.digest(proposal)) for p in proposal]:
                raise ValueError('Independent proposal differs')
            snippet=gold[int(r['qa_id'])]['related_conversation_snippet']
            try:snippet=json.loads(snippet)
            except ValueError:snippet=ast.literal_eval(snippet)
            raw_cards=[dict(v,id=str(i),text=v['content']) for i,v in enumerate(raw[t['user_id']].values())]
            raw_cov=audit.turn_coverage([audit.norm(s['content']) for s in snippet if s['role']=='user'],raw_cards,raw[t['user_id']])
            coverage={};sets={};sizes={};catalogs={}
            for mode,packet in [('baseline',baseline),('actual',actual)]:
                cards=[dict(s,text=s.get('content',''),id=f'{i}:{j}') for i,p in enumerate(packet) for j,s in enumerate(p['sources'])]
                coverage[mode]={role:[v['full'] for v in audit.turn_coverage(
                    [audit.norm(s['content']) for s in snippet if s['role']==role],cards,raw[t['user_id']],role=role)]
                    for role in ('user','assistant')}
                sets[mode]=reference_set_coverage(raw_cov,[dict(full=v) for v in coverage[mode]['user']])
                catalogs[mode]=answer_choice.build_catalog(packet)
                sizes[mode]=sum(len(c['text'].encode()) for cat in catalogs[mode] for c in cat.values())
            observed_catalog='answer_source_catalog' in r and 'answer_constraint_catalog' in r
            if not observed_catalog and not r.get('error_stage'):
                raise ValueError('Successful answer lacks its catalog')
            if observed_catalog and (list(catalogs['actual'][0].values())!=r['answer_source_catalog'] or list(catalogs['actual'][1].values())!=r['answer_constraint_catalog']):
                raise ValueError('Reader catalog differs')
            rows.append(dict(qa_id=r['qa_id'],score=r['score'],changed=baseline_hash!=actual_hash,
                baseline_hash=baseline_hash,actual_hash=actual_hash,coverage=coverage,complete_sets=sets,catalog_bytes=sizes,
                observed_reader_catalog_verified=observed_catalog))
        if len(rows)!=plan['questions']:raise ValueError('Incomplete trace cohort')
        logs='\n'.join(l for l in reader_lines if l.startswith('reader_arm='+arm+' '))
        reader_stages=provider_stage_costs(logs);dispatches=check_provider_log(logs,arm+' reader')
        search_stages=provider_stage_costs((run/'server.log').read_text())
        for field,logged in [('provider_calls',sum(v['calls'] for v in search_stages.values())),
                             ('provider_tokens',sum(v['tokens'] for v in search_stages.values()))]:
            if sum(r['coverage_manifest'][field] for r in rs)!=logged:raise ValueError('Search ledger mismatch '+field)
        summary=dict(questions=len(rows),changed_questions=sum(r['changed'] for r in rows),
            terminal_reader_errors=[dict(qa_id=r['qa_id'],error_type=r['error_type'],score=r['score']) for r in failures],
            observed_reader_catalogs_verified=sum(r['observed_reader_catalog_verified'] for r in rows),
            reference_questions={m:sum(any(r['coverage'][m]['user']) for r in rows) for m in ('baseline','actual')},
            user_turns={m:sum(sum(r['coverage'][m]['user']) for r in rows) for m in ('baseline','actual')},
            assistant_turns={m:sum(sum(r['coverage'][m]['assistant']) for r in rows) for m in ('baseline','actual')},
            complete_sets={m:sum(r['complete_sets'][m]['available_set_complete'] for r in rows) for m in ('baseline','actual')},
            catalog_bytes={m:sum(r['catalog_bytes'][m] for r in rows) for m in ('baseline','actual')},
            gained=[r['qa_id'] for r in rows if any(r['coverage']['actual']['user']) and not any(r['coverage']['baseline']['user'])],
            lost=[r['qa_id'] for r in rows if any(r['coverage']['baseline']['user']) and not any(r['coverage']['actual']['user'])],
            turn_regressions=[r['qa_id'] for r in rows if any(a and not b for role in ('user','assistant')
                for a,b in zip(r['coverage']['baseline'][role],r['coverage']['actual'][role]))],
            reader_dispatches=dispatches,reader_calls=sum(v['calls'] for v in reader_stages.values()),
            reader_tokens=sum(v['tokens'] for v in reader_stages.values()),reader_provider_stages=reader_stages,
            search_provider_statuses=dict(sum((Counter(v['statuses']) for v in search_stages.values()),Counter())))
        arms[arm]=dict(summary=summary,rows=rows)
    output=dict(arms=arms,source_files_verified=len(plan['source_files']),auditor_sha256=digest(Path(__file__)),
                reader_log_sha256=digest(root/'reader-provider.log'),limitation=__doc__)
    with (root/'combined-query-effect-audit.json').open('x') as f:json.dump(output,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({a:v['summary'] for a,v in arms.items()},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--include-reader-errors',action='store_true')
    p.add_argument('--gold-csv',type=Path,default=Path(__file__).resolve().parents[1]/'data/raw/personamem-v2.csv')
    main(p.parse_args())
