"""Audit shared live HTTPS inputs, isolated readers, provenance and failures."""
import argparse
import asyncio
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys
from unittest.mock import patch

from analyze_overflow_https import provider_stage_costs
from evaluation_model_guard import check_versions,check_provider_log


def records(path):return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def fingerprint(value):return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True).encode()).hexdigest()


async def main(root,include_reader_errors=False):
    root=root.resolve();plan=json.loads((root/'plan.json').read_text());manifest=json.loads((root/'manifest.json').read_text())
    exit_code=(root/'exit-code').read_text().strip()
    if exit_code not in ('0','1') or (exit_code!='0' and not include_reader_errors) or plan['protocol']!='shared-search-reader-https-v1':
        raise ValueError('Require complete successful shared-search trial')
    if any(manifest['audit_exit_codes'].values()):raise ValueError('Stage audit failed')
    for pid in [manifest['server_pid'],*manifest['reader_pids'].values()]:
        p=Path('/proc')/str(pid)/'cmdline'
        if p.exists() and str(root).encode() in p.read_bytes():raise ValueError('Trial process still live')
    for name,key in [('fixture.json','fixture_sha256'),('frozen-memory.db','database_sha256')]:
        if digest(root/name)!=plan[key]:raise ValueError('Frozen input changed')
    for name,value in plan['source_files'].items():
        if digest(root/'source'/name)!=value:raise ValueError('Frozen source changed')
    prepared=Path(manifest['data'])
    if digest(prepared)!=plan['data_sha256']:raise ValueError('Prepared data changed')
    dataset={r['conversation_id']:r for r in records(prepared)}
    fixture=json.loads((root/'fixture.json').read_text());expected={r['qa_id']:r for r in fixture['requests']}
    inputs={r['qa']['id']:r for r in records(root/'http-reader-inputs.jsonl')}
    traces={t['search_id']:t for t in records(root/'search/server-search-debug.jsonl')}
    if len(inputs)!=plan['questions'] or len(traces)!=plan['questions']:raise ValueError('Incomplete shared Search cohort')
    sys.path.insert(0,str(root/'source/memory_system'))
    from app import answer_choice,choice_witness,config,evidence_packet,llm
    for q,job in inputs.items():
        payload=job['payload'];t=traces[payload['search_id']];e=expected[q]
        qa=next(v for v in dataset[job['conversation_id']]['qa'] if v['id']==q)
        if qa!=job['qa'] or job['request']!=e['request'] or t['plan']!=e['plan']:
            raise ValueError('Question, request or frozen plan changed')
        if t['query']!=qa['question'] or t['options']!=qa['options'] or t['user_id']!=job['request']['user_id']:
            raise ValueError('Search identity differs')
        if payload['data']!=t['returned'] or evidence_packet.digest(payload['data'])!=t['packet_hash']:
            raise ValueError('HTTP evidence differs from service trace')
        check_versions(t['versions'],'shared Search')
        if t['versions']['settings'].get(plan['reader_toggle']):raise ValueError('Reader experiment leaked into Search')
    versions={};arms={};rows_by_arm={}
    for arm in ('control','treatment'):
        run=root/arm;versions[arm]=json.loads((run/'reader-versions.json').read_text());check_versions(versions[arm],arm+' reader')
        for name in ('server-search-debug.jsonl','server-memory-debug.jsonl'):
            if digest(run/name)!=digest(root/'search'/name):raise ValueError('Arms did not share Search')
        rows=records(run/'completed/data/results/personamem-v2-32k.jsonl');rs={r['qa_id']:r for r in rows};rows_by_arm[arm]=rs
        if len(rows)!=plan['questions'] or set(rs)!=set(inputs):raise ValueError('Incomplete reader cohort')
        failures=[r for r in rows if r.get('error_stage')]
        if (manifest['reader_exit_codes'][arm]!=int(bool(failures)) or
                failures and (not include_reader_errors or any(r['error_stage']!='answer' or r['score']!=0 for r in failures))):
            raise ValueError('Unpermitted or inconsistent terminal failure')
        verified=0
        for q,r in rs.items():
            job=inputs[q];payload=job['payload'];sources,constraints=answer_choice.build_catalog(payload['data'])
            if (r['reader_input_sha256']!=fingerprint(job) or r['http_payload_sha256']!=fingerprint(payload)
                    or r['packet_hash']!=payload['packet_hash'] or r['search_id']!=payload['search_id']
                    or r['coverage_manifest']!=payload['coverage_manifest']
                    or r.get('reader_toggle_enabled',r['reader_keyed_output'])!=(arm=='treatment')
                    or r.get('reader_toggle',plan['reader_toggle'])!=plan['reader_toggle']):
                raise ValueError('Reader input or arm identity differs')
            if list(sources.values())!=r['answer_source_catalog'] or list(constraints.values())!=r['answer_constraint_catalog']:
                raise ValueError('Reader catalog differs')
            calls=[c for c in r['answer_calls'] if c['stage']=='eval.choice_witness']
            if len(calls)!=1:raise ValueError('Expected exactly one original witness call')
            old=calls[0];captured=[];diagnostics={}
            async def capture(prompt,**kwargs):
                captured.append(dict(prompt=prompt,**kwargs))
                if 'response' not in old:raise llm.LLMError('Recorded failed provider invocation')
                return old['response']
            with patch.multiple(config,**versions[arm]['settings']),patch.object(llm,'complete_json',capture),\
                    patch('socket.socket.connect',side_effect=AssertionError('Offline audit')):
                hints=await choice_witness.select(job['qa']['question'],job['qa']['options'],sources,diagnostics)
                if config.CHOICE_COVERAGE_WITNESSES:
                    hints=choice_witness.supplement_from_coverage(job['qa']['options'],sources,hints,payload['data'])
                if getattr(config,'CHOICE_WITNESS_LEXICAL_FILL',False):
                    hints,fill_trace=choice_witness.fill_unused(job['qa']['options'],sources,hints)
                    if fill_trace!=r.get('answer_witness_lexical_fill'):
                        raise ValueError('Lexical fill did not reproduce')
                if hints!=r['answer_witness_prefill']:
                    raise ValueError('Final witness hints did not reproduce')
            if len(captured)!=1 or hashlib.sha256(captured[0]['prompt'].encode()).hexdigest()!=old['prompt_sha256']:
                raise ValueError('Witness request did not reconstruct')
            if diagnostics['answer_calls'][0]['status']!=old['status']:
                raise ValueError('Witness validation status did not reproduce')
            if diagnostics['answer_witness_retrieval']['status']!=r['answer_witness_retrieval']['status']:
                raise ValueError('Witness fallback status differs')
            verified+=1
        log=(run/'reader.log').read_text();dispatches=check_provider_log(log,arm+' reader')
        errors=[dict(qa_id=r['qa_id'],score=r['score'],**{k:c.get(k) for k in ('stage','error_type','failure_phase','cause_types')})
            for r in rows for c in r['answer_calls'] if c['status']=='error']
        stage=json.loads((run/'funnel/stage-transport.json').read_text())['summary']
        stages=provider_stage_costs(log)
        arms[arm]=dict(questions=len(rows),score=sum(r['score'] for r in rows),inputs_and_catalogs_verified=verified,
            witness_requests_verified=verified,model_dispatches=dispatches,provider_stages=stages,
            provider_calls=sum(v['calls'] for v in stages.values()),provider_tokens=sum(v['tokens'] for v in stages.values()),
            witness_statuses=dict(Counter(r['answer_witness_retrieval']['status'] for r in rows)),
            witness_selected_count=sum(r['answer_witness_retrieval'].get('selected_count',0) for r in rows),
            reader_errors=errors,visible=stage['visible_full_turn_question_counts'],
            terminal_reader_errors=[dict(qa_id=r['qa_id'],score=r['score'],error_type=r.get('error_type')) for r in failures],
            lexical_fill_questions=sum(bool(r.get('answer_witness_lexical_fill',{}).get('added_count')) for r in rows),
            lexical_fill_hints=sum(r.get('answer_witness_lexical_fill',{}).get('added_count',0) for r in rows),
            complete_sets=stage['available_set_complete_question_counts'],turns=stage['delivered_available_turn_counts'])
    differences={k:[versions['control']['settings'].get(k),versions['treatment']['settings'].get(k)]
        for k in versions['control']['settings'].keys()|versions['treatment']['settings'].keys()
        if versions['control']['settings'].get(k)!=versions['treatment']['settings'].get(k)}
    if differences!={plan['reader_toggle']:[False,True]}:raise ValueError('Unexpected reader setting differences')
    if versions['control']['file_hashes']!=versions['treatment']['file_hashes']:raise ValueError('Readers used different code')
    for name,value in versions['control']['file_hashes'].items():
        if digest(root/'source'/name)!=value:raise ValueError('Reader source hash differs')
    pairs=[]
    for q in sorted(inputs,key=int):
        a,b=rows_by_arm['control'][q],rows_by_arm['treatment'][q]
        if a['reader_input_sha256']!=b['reader_input_sha256']:raise ValueError('Paired reader inputs differ')
        pairs.append(dict(qa_id=q,control_score=a['score'],treatment_score=b['score'],
            control_prediction=a['prediction'],treatment_prediction=b['prediction'],
            witness_statuses={arm:rows_by_arm[arm][q]['answer_witness_retrieval']['status'] for arm in arms}))
    excluded={r['qa_id'] for arm in arms.values() for r in arm['reader_errors']}
    llm_errors={r['qa_id'] for arm in arms.values() for r in arm['reader_errors'] if r['error_type']=='LLMError'}
    sensitivity={label:dict(questions=len(inputs.keys()-removed),excluded=sorted(removed,key=int),
        scores={arm:sum(rows_by_arm[arm][q]['score'] for q in inputs.keys()-removed) for arm in arms})
        for label,removed in [('exclude_reader_llm_errors',llm_errors),('exclude_all_reader_call_errors',excluded)]}
    search_log=(root/'search/server.log').read_text();search_stages=provider_stage_costs(search_log)
    calls=sum(j['payload']['coverage_manifest']['provider_calls'] for j in inputs.values())
    tokens=sum(j['payload']['coverage_manifest']['provider_tokens'] for j in inputs.values())
    logged_calls=sum(s['calls'] for s in search_stages.values())
    if tokens!=sum(s['tokens'] for s in search_stages.values()):
        raise ValueError('Shared Search recorded token ledger differs')
    ce_batches=[dict(qa_id=q,search_id=job['payload']['search_id'],**batch)
        for q,job in inputs.items()
        for batch in traces[job['payload']['search_id']]['cascade']['cross_encoder']['batches']]
    ce_timeouts=[b for b in ce_batches if b['status']=='error' and b.get('error')=='TimeoutError']
    ce_logs={k:v for k,v in search_stages.items() if k.startswith('search.cross_encoder')}
    unlogged_timeouts=[]
    unknown_usage_calls=[dict(re.findall(r'(\w+)=([^\s]+)',line)) for line in search_log.splitlines()
                         if 'provider_call ' in line and 'usage_known=false' in line]
    if calls!=logged_calls:
        # The frozen cross_encoder uses wait_for around measured_call. Its
        # cancellation bypasses metrics' except Exception, after before_call
        # increments the request ledger. Require complete CE conservation;
        # never turn the missing event into a successful/zero-cost invocation.
        ce_logged=sum(v['calls'] for v in ce_logs.values())
        if (calls-logged_calls!=len(ce_timeouts) or not ce_timeouts
                or len(ce_batches)!=ce_logged+len(ce_timeouts)
                or sum(b['status']=='ok' for b in ce_batches)!=ce_logged
                or any(set(v['statuses'])!={'ok'} for v in ce_logs.values())):
            raise ValueError('Shared Search call ledger has an unexplained difference')
        unlogged_timeouts=[{k:b[k] for k in ('qa_id','search_id','batch','error','candidate_count')}
                           for b in ce_timeouts]
    result=dict(arms=arms,settings_differences=differences,pairs=pairs,
        original_exit_code=int(exit_code),original_failures_retained=True,
        wins=[r['qa_id'] for r in pairs if r['treatment_score']>r['control_score']],
        losses=[r['qa_id'] for r in pairs if r['treatment_score']<r['control_score']],sensitivity=sensitivity,
        shared_search=dict(questions=len(inputs),calls=calls,tokens=tokens,logged_calls=logged_calls,
            unlogged_ce_timeouts=unlogged_timeouts,
            unknown_usage_calls=unknown_usage_calls,
            accounting_limitation=('Some cancelled calls have unknown provider usage; '
                'recorded tokens exclude any unreported usage.' if unlogged_timeouts or unknown_usage_calls else None),
            model_dispatches=check_provider_log(search_log,'shared Search'),provider_stages=search_stages),
        source_files_verified=len(plan['source_files']),auditor_sha256=digest(Path(__file__)),limitation=__doc__)
    with (root/'shared-reader-audit.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('pairs',)},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--include-reader-errors',action='store_true')
    args=p.parse_args();asyncio.run(main(args.run,args.include_reader_errors))
