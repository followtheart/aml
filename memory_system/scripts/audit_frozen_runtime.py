"""Account for logged Search attempts and reader fallback errors after HTTPS.

Reader call errors can be caught internally and therefore need not set the
client error_stage. Sensitivity subsets are descriptive, not repaired trials.
Reader recorded usage is included when reader-provider.log is available.
Cancelled/failed attempts may have unknown provider usage.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from analyze_overflow_https import provider_stage_costs
from evaluation_model_guard import check_provider_log


def records(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def runtime_failures(value,path=()):
    found=[]
    if isinstance(value,dict):
        kind=value.get('error_type') or value.get('error')
        if isinstance(kind,str) and (kind in ('LLMError','TimeoutError','BudgetExceeded','InternalServerError')
                                    or (path and path[-1]=='question_ce' and value.get('status')=='error')):
            found.append(dict(path=list(path),error_type=kind))
        for key,child in value.items():
            if isinstance(child,(dict,list)):found.extend(runtime_failures(child,(*path,key)))
    elif isinstance(value,list):
        for index,child in enumerate(value):found.extend(runtime_failures(child,(*path,index)))
    return found


def main(root,include_reader_errors=False):
    exit_code=(root/'exit-code').read_text().strip()
    if exit_code not in ('0','1') or (exit_code!='0' and not include_reader_errors):
        raise ValueError('Require a completed trial; reader errors must be explicitly included')
    plan=json.loads((root/'plan.json').read_text())
    arms={};by_question={}
    reader_path=root/'reader-provider.log'
    reader_lines=reader_path.read_text().splitlines() if reader_path.exists() else None
    if reader_lines is not None and any(not any(l.startswith('reader_arm='+a+' ') for a in ('control','treatment')) for l in reader_lines):
        raise ValueError('Unattributed reader provider event')
    for arm in ('control','treatment'):
        run=root/arm
        manifest=json.loads((run/'isolated-manifest.json').read_text())
        if manifest.get('audit_exit_code')!=0 or manifest.get('client_exit_code') not in (0,1):
            raise ValueError('Unexpected client or stage audit failure')
        if 'server_exit_code' not in manifest:
            raise ValueError('Missing terminal server status')
        proc=Path('/proc')/str(manifest['server_pid'])/'cmdline'
        if proc.exists() and str(root.resolve()).encode() in proc.read_bytes():
            raise ValueError('Service still running')
        rows=records(run/'completed/data/results/personamem-v2-32k.jsonl')
        if len(rows)!=plan['questions']:
            raise ValueError('Incomplete answer cohort')
        terminal=[r for r in rows if r.get('error_stage')]
        if (manifest['client_exit_code']!=int(bool(terminal)) or (terminal and not include_reader_errors)
                or any(r['error_stage']!='answer' or r['score']!=0 or not r.get('search_id') for r in terminal)):
            raise ValueError('Only complete attempts with reader errors may be included')
        by_question[arm]={r['qa_id']:r for r in rows}
        if len(by_question[arm])!=len(rows):
            raise ValueError('Duplicate question identity')
        errors=[dict(qa_id=r['qa_id'],stage=c['stage'],status=c['status'],
                     error_type=c.get('error_type'),response_recorded='response' in c)
                for r in rows for c in r.get('answer_calls',[]) if c.get('status')=='error']
        qa_by_search={r['search_id']:r['qa_id'] for r in rows}
        search_errors=[]
        for trace in records(run/'server-search-debug.jsonl'):
            search_errors.extend(dict(qa_id=qa_by_search[trace['search_id']],**e)
                                 for e in runtime_failures(trace['cascade']))
        stages=provider_stage_costs((run/'server.log').read_text())
        budget_calls=sum(r['coverage_manifest']['provider_calls'] for r in rows)
        logged_calls=sum(v['calls'] for v in stages.values())
        budget_tokens=sum(r['coverage_manifest']['provider_tokens'] for r in rows)
        logged_tokens=sum(v['tokens'] for v in stages.values())
        arms[arm]=dict(questions=len(rows),server_exit_code=manifest['server_exit_code'],
            client_errors=[r['qa_id'] for r in rows if r.get('error_stage')],
            search_budget_calls=budget_calls,search_logged_attempts=logged_calls,
            search_budget_tokens=budget_tokens,search_logged_tokens=logged_tokens,
            calls_match=budget_calls==logged_calls,tokens_match=budget_tokens==logged_tokens,
            search_provider_statuses=dict(sum((Counter(s['statuses']) for s in stages.values()),Counter())),
            reader_error_types=dict(Counter(r['error_type'] for r in errors)),reader_errors=errors,
            reader_terminal_errors=[dict(qa_id=r['qa_id'],error_type=r['error_type']) for r in terminal],
            search_runtime_errors=search_errors,
            reader_llm_error_questions=sorted({r['qa_id'] for r in errors+terminal if r['error_type']=='LLMError'},key=int),
            reader_billed_tokens_recorded=False)
        if reader_lines is not None:
            reader_log='\n'.join(l for l in reader_lines if l.startswith('reader_arm='+arm+' '))
            reader_stages=provider_stage_costs(reader_log)
            arms[arm].update(reader_billed_tokens_recorded=True,
                reader_model_dispatches=check_provider_log(reader_log,arm+' reader'),
                reader_recorded_calls=sum(s['calls'] for s in reader_stages.values()),
                reader_recorded_tokens=sum(s['tokens'] for s in reader_stages.values()),
                reader_provider_stages=reader_stages,
                reader_provider_statuses=dict(sum((Counter(s['statuses']) for s in reader_stages.values()),Counter())),
                reader_usage_limitation='Recorded usage only; failed/cancelled calls may have unreported provider usage.')
    if by_question['control'].keys()!=by_question['treatment'].keys():
        raise ValueError('Different question cohorts')
    excluded_llm=set().union(*(set(v['reader_llm_error_questions']) for v in arms.values()))
    excluded_errors={r['qa_id'] for v in arms.values() for r in v['reader_errors']+v['reader_terminal_errors']}
    excluded_search={r['qa_id'] for v in arms.values() for r in v['search_runtime_errors']}
    recovery_path=root/'reader-recovery.json'
    recovered=set()
    if recovery_path.exists():
        recovery=json.loads(recovery_path.read_text())
        recovered={r['qa_id'] for r in recovery['attempts']}
        if not recovery.get('original_unchanged') or not recovery.get('search_traces_unchanged'):
            raise ValueError('Recovery provenance not finalized')
    subsets={}
    for name,excluded in (('original_attempt_pairs',recovered),
                          ('exclude_reader_llm_errors_and_recovery',excluded_llm|recovered),
                          ('exclude_search_runtime_reader_llm_errors_and_recovery',excluded_search|excluded_llm|recovered),
                          ('exclude_all_reader_call_errors_and_recovery',excluded_errors|recovered)):
        keys=by_question['control'].keys()-excluded
        subsets[name]=dict(questions=len(keys),excluded=sorted(excluded,key=int),
            score={a:sum(by_question[a][q]['score'] for q in keys) for a in arms})
    result=dict(arms=arms,sensitivity=subsets,recovered_questions=sorted(recovered,key=int),limitation=__doc__,
        auditor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (root/'runtime-accounting-audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--include-reader-errors',action='store_true')
    args=p.parse_args();main(args.run,args.include_reader_errors)
