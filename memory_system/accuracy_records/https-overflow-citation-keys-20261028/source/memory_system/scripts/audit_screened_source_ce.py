"""Independent exact-request, score, cost and fixed-nomination recall audit."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path

from batched_records import file_digest, json_records
from run_screened_source_ce import read, body_digest
from source_nomination import wire_body


def require(value, message):
    if not value:
        raise ValueError(message)


def audit(root):
    plan = read(root/'plan.json')
    for name,digest in {**plan['source_sha256'],**plan['input_sha256']}.items():
        require(file_digest(Path(name))==digest, 'Frozen source/input changed: '+name)
    # Both complete fixed nominations must precede any reference-bound lookup.
    for cohort in ('regression','validation'):
        manifest=read(root/(cohort+'-selections.manifest.json'))
        require(manifest['status']=='complete_fixed_nominations', 'Both cohorts must complete')
        require(manifest['plan_sha256']==file_digest(root/'plan.json'), 'Frozen paid plan')
        for name,digest in manifest['output_sha256'].items():
            require(file_digest(root/(cohort+'-'+name+'.jsonl'))==digest, 'Frozen paid output')
    reports={}
    for cohort,spec in plan['cohorts'].items():
        selections={r['search_id']:r for r in json_records(root/(cohort+'-selections.jsonl'))}
        require(len(selections)==len(spec['search_ids']), 'Unique complete nominations')
        requests={}
        for event in json_records(root/(cohort+'-requests.jsonl')):
            requests.setdefault((event['search_id'],event['batch']),[]).append(event)
        ledger={}
        for event in json_records(root/(cohort+'-ledger.jsonl')):
            ledger.setdefault((event['search_id'],event['batch']),[]).append(event['fields'])
        registered={(r['search_id'],r['batch']):r for r in spec['requests']}
        require(set(requests)<=set(registered), 'No unregistered source request')
        require(set(ledger)<=set(registered), 'No unregistered provider call')
        fixed=[]
        used_calls=used_tokens=unknown_failures=0
        for job in json_records(Path(spec['inventory'])):
            sid=job['search_id']; selection=selections[sid]
            require(selection['qa_id']==job['qa_id'] and selection['conversation_id']==job['conversation_id'], 'Question identity')
            scores={};calls=tokens=0;failed=False
            if job['scheduling']['eligible']:
                for index,batch in enumerate(job['batches']):
                    key=sid,index;entry=registered[key]
                    body=wire_body(job['ce_model'],job['ce_api_format'],job['ce_query'],[job['views'][mid]['text'] for mid in batch])
                    require(body_digest(body)==entry['body_sha256'], 'Independent exact request construction')
                    events=requests.get(key,[]);records=ledger.get(key,[])
                    if not events:
                        require(failed, 'Unattempted batch without earlier failure')
                        require(not records, 'Unattempted batch has provider usage')
                        continue
                    dispatches=[r for r in events if r['status']=='dispatch']
                    terminal=[r for r in events if r['status'] in ('ok','error')]
                    require(len(dispatches)<=1 and len(terminal)==1, 'Single dispatch and terminal per batch')
                    require(all(r['body_sha256']==entry['body_sha256'] and r['bound']==entry['bound'] for r in events), 'Wire digest/bound')
                    require(len(records)==len(dispatches), 'Every HTTP dispatch has a provider ledger record')
                    end=terminal[0]
                    require(end['calls']==len(records), 'Budget calls agree with provider ledger')
                    for record in records:
                        require(record['kind']=='rerank' and record['stage']=='diagnostic.screened_source_ce'
                            and record['model']=='qwen3-rerank' and record['attempt']=='1' and record['retries']=='0', 'Fixed provider dispatch')
                        unknown_failures += record['status']!='ok'
                    usage=sum(int(r['total_tokens']) for r in records)
                    require(usage==end['tokens'], 'Exact per-batch usage ledger')
                    calls+=len(records);tokens+=usage
                    if end['status']=='ok':
                        require(not failed and len(records)==1 and records[0]['status']=='ok', 'Complete successful source call')
                        require(end['view_ids']==batch and len(end['scores'])==len(batch), 'Complete ordered scores')
                        require(all(type(v) in (int,float) and math.isfinite(v) for v in end['scores']), 'Finite scores')
                        scores.update(zip(batch,end['scores'],strict=True))
                    else:
                        failed=True
            require(selection['provider_calls']==selection['budget_calls']==calls, 'Question calls')
            require(selection['reported_tokens']==tokens, 'Question tokens')
            require(selection['scores']==({} if failed else scores), 'No partial-score nomination')
            expected=None
            if job['scheduling']['eligible'] and not failed:
                require(set(scores)==set(job['views']), 'All planned sources scored')
                ranked=sorted(job['nominees'], key=lambda c:(-max(scores[k] for k in c['source_view_ids']),c['ce_rank']))
                expected=ranked[0]['id']
            require(selection['decision']['addition']==expected, 'Independent source-maximum nomination')
            require(selection['original_main_ids']==job['original_main_ids'], 'Original main')
            require(selection['proposed_main_ids']==job['original_main_ids']+([expected] if expected else []), 'One appended candidate')
            require(selection['original_display_preserved']==(expected is None), 'No-op original display')
            used_calls+=calls;used_tokens+=tokens
            fixed.append(dict(search_id=sid,qa_id=job['qa_id'],conversation_id=job['conversation_id'],
                              addition=expected,status=selection['decision']['status']))
        # Fixed nominations verified above; references may now measure them.
        bounds=read(Path(spec['upper_bound_report']))
        by_question={(r['conversation_id'],str(r['qa_id'])):r for r in bounds['rows']}
        for row in fixed:
            bound=by_question[row['conversation_id'],str(row['qa_id'])]
            gains=next((r['gains'] for r in bound['trials'] if r['id']==row['addition']),[]) if row['addition'] else []
            row.update(baseline=bound['baseline'],gains=gains)
        summary=dict(questions=len(fixed),nominated=sum(bool(r['addition']) for r in fixed),
            fallback=sum(r['status']=='fallback' for r in fixed),calls=used_calls,reported_tokens=used_tokens,
            failed_calls_without_trustworthy_usage=unknown_failures,
            baseline_questions=sum(any(r['baseline']) for r in fixed),
            proposed_questions=sum(any(r['baseline']) or bool(r['gains']) for r in fixed),
            baseline_turns=sum(sum(r['baseline']) for r in fixed),
            proposed_turns=sum(sum(r['baseline'])+len(r['gains']) for r in fixed),
            gains=[dict(qa_id=r['qa_id'],addition=r['addition'],turns=r['gains']) for r in fixed if r['gains']])
        reports[cohort]=dict(summary=summary,rows=fixed)
    complete=all(r['summary']['fallback']==0 and r['summary']['failed_calls_without_trustworthy_usage']==0 for r in reports.values())
    gain=any(r['summary']['proposed_turns']>r['summary']['baseline_turns'] for r in reports.values())
    decision='proceed_to_context_review' if complete and gain else 'reject_errors_or_no_actual_recall_gain'
    result=dict(status='complete_independent_source_ce_audit',decision=decision,cohorts=reports,
        full_context_review_complete=False,runtime_integrated=False,https_answer_evaluated=False,
        limitation='Fixed-main source transport only; no causal answer improvement or live scheduling claim. '
        'Failed calls without trustworthy usage are not zero cost. Same development cohorts as prior experiments.',
        input_sha256={str(p):file_digest(p) for p in [root/'plan.json',*root.glob('*.jsonl'),
            *root.glob('*-selections.manifest.json')]})
    with (root/'audit.json').open('x') as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2);stream.write('\n')
    print(json.dumps(dict(decision=decision,cohorts={k:v['summary'] for k,v in reports.items()})),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',type=Path,required=True)
    audit(parser.parse_args().experiment.resolve())
