"""Re-decode main-layout pairs and measure primary full-user-reference transport.

Report complete-contract pairs separately from complete candidate-decision
pairs. Group errors cannot overwrite the production contract's accepted
candidate records, but can require additional repair and affect downstream
packing. This is not a replacement for recovery or full HTTPS evaluation.
Reference transport and valid relevance labels do not prove semantic usefulness.
"""
import argparse
import ast
from collections import Counter
import csv
import json
from pathlib import Path

from probe_main_source_layout import decode, fingerprint, prepare, records, sha
import audit_https_funnel as funnel
from evaluation_model_guard import check_provider_log


def main(args):
    root = args.run
    if (root/'exit-code').read_text().strip() != '0':
        raise ValueError('Require completed diagnostic')
    plan = json.loads((root/'plan.json').read_text())
    summary = json.loads((root/'summary.json').read_text())
    if not plan['live'] or sha(root/'plan.json') != summary['plan_sha256']:
        raise ValueError('Plan mismatch')
    for name,key in [('requests.jsonl','requests_sha256'),('live.jsonl','responses_sha256')]:
        if sha(root/name) != summary[key]:
            raise ValueError('Recorded artifact changed')
    if sha(Path(__file__).with_name('probe_main_source_layout.py')) != plan['script_sha256']:
        raise ValueError('Request producer changed')
    source = Path(plan['source_run'])
    if sha(source/'server-search-debug.jsonl') != plan['source_trace_sha256']:
        raise ValueError('Historical trace changed')
    jobs = prepare(source)
    if fingerprint(jobs) != plan['jobs_sha256'] or jobs != records(root/'requests.jsonl'):
        raise ValueError('Requests did not reproduce')
    jobs = {j['qa_id']: j for j in jobs}
    responses = records(root/'live.jsonl')
    if (len(responses) != plan['expected_calls'] or
            {(r['qa_id'],r['arm']) for r in responses} != {(q,a) for q in jobs for a in ('control','treatment')}):
        raise ValueError('Incomplete/duplicate response cohort')
    ledger = [dict(p.split('=',1) for p in line.split() if '=' in p)
        for line in (root/'provider.log').read_text().splitlines() if 'provider_call ' in line]
    if check_provider_log((root/'provider.log').read_text(),'main layout') != len(ledger):
        raise ValueError('Unexpected provider kind')
    if len(ledger) != summary['calls'] or sum(int(e['total_tokens']) for e in ledger) != summary['tokens']:
        raise ValueError('Provider accounting differs')
    by_question = {}
    for r in responses:
        j = jobs[r['qa_id']]
        request = dict(j['requests'][r['arm']]); reservation = request.pop('reservation')
        if fingerprint(request) != r['request_sha256'] or reservation != r['reservation']:
            raise ValueError('Request identity differs')
        if 'payload' in r:
            try:
                restored = decode(r['payload'],j)
            except Exception as exc:
                if r.get('error') != type(exc).__name__:
                    raise ValueError('Decoding failure differs') from exc
            else:
                if restored != r.get('decoded'):
                    raise ValueError('Decoded response differs')
        events = [e for e in ledger if e['stage'] == f'diagnostic.main_source_layout.{r["qa_id"]}.{r["arm"]}']
        if len(events) != r['calls'] or sum(int(e['total_tokens']) for e in events) != r['tokens']:
            raise ValueError('Per-response accounting differs')
        by_question.setdefault(r['qa_id'],{})[r['arm']] = r
    # Reference annotations are consulted only after all selections and
    # response/ledger validations above have completed.
    csv.field_size_limit(32*1024*1024)
    with args.gold_csv.open() as stream:
        gold = list(csv.DictReader(stream))
    traces = {t['search_id']:t for t in records(source/'server-search-debug.jsonl')}
    raw = {}
    for event in records(source/'server-memory-debug.jsonl'):
        raw.setdefault(event['user_id'],{}).update({funnel.identity(s):s for s in event['source_messages']})
    rows, additions = [], []
    for q,job in jobs.items():
        t = traces[job['search_id']]
        items = {c['id']:c['_packet_item'] for c in t['ranked'] if '_packet_item' in c}
        snippet = gold[int(q)]['related_conversation_snippet']
        try:
            snippet = json.loads(snippet)
        except ValueError:
            snippet = ast.literal_eval(snippet)
        turns = [funnel.norm(s['content']) for s in snippet if s['role']=='user']
        def coverage(ids):
            cards = [dict(s,id=f'{mid}:{i}',text=s.get('content',''))
                for mid in ids for i,s in enumerate(items[mid].get('sources',[]))]
            covered = funnel.turn_coverage(turns,cards,raw[t['user_id']])
            return dict(full_turns=sum(v['full'] for v in covered), turns=[v['full'] for v in covered])
        historical = coverage([mid for mid,label in job['historical_labels'].items() if label!='irrelevant'])
        outcomes = {}
        for arm,response in by_question[q].items():
            decoded = response.get('decoded',{})
            valid = decoded.get('valid',False) and not response.get('error')
            candidate_valid = (not response.get('error') and decoded.get('selected_ids') is not None
                and 'candidates' not in decoded.get('errors',{}))
            outcomes[arm] = dict(valid=valid,error=response.get('error'),validation_errors=decoded.get('errors'),
                candidate_valid=candidate_valid, selected_ids=decoded.get('selected_ids'),
                coverage=coverage(decoded['selected_ids']) if candidate_valid else None)
        valid_pair = all(v['valid'] for v in outcomes.values())
        candidate_pair = all(v['candidate_valid'] for v in outcomes.values())
        if candidate_pair:
            before,after = (set(outcomes[a]['selected_ids']) for a in ('control','treatment'))
            for mid in sorted(after-before):
                additions.append(dict(qa_id=q,candidate_id=mid,changed_layout=job['changed'],
                    full_contract_valid_pair=valid_pair,
                    query=t['query'],options=t['options'],reference_carrier=coverage([mid])['full_turns']>0,
                    sources=items[mid].get('sources',[])))
        rows.append(dict(qa_id=q,changed=job['changed'],valid_pair=valid_pair,candidate_pair=candidate_pair,
            historical_primary=historical,arms=outcomes))
    def summarize(subset, eligibility='valid_pair'):
        paired = [r for r in subset if r[eligibility]]
        return dict(questions=len(subset),valid_pairs=len(paired),
            eligibility=eligibility, excluded_questions=[r['qa_id'] for r in subset if not r[eligibility]],
            paired_primary_transport={a:dict(reference_questions=sum(r['arms'][a]['coverage']['full_turns']>0 for r in paired),
                full_turns=sum(r['arms'][a]['coverage']['full_turns'] for r in paired)) for a in ('control','treatment')},
            gains=[r['qa_id'] for r in paired if r['arms']['treatment']['coverage']['full_turns']>r['arms']['control']['coverage']['full_turns']],
            losses=[r['qa_id'] for r in paired if any(x and not y for x,y in zip(
                r['arms']['control']['coverage']['turns'],r['arms']['treatment']['coverage']['turns'],strict=True))])
    result = dict(summary=dict(all=summarize(rows),changed=summarize([r for r in rows if r['changed']]),
        unchanged=summarize([r for r in rows if not r['changed']]),
        candidate_stage={name:summarize(subset,'candidate_pair') for name,subset in (
            ('all',rows),('changed',[r for r in rows if r['changed']]),('unchanged',[r for r in rows if not r['changed']]))},
        responses=len(responses),calls=len(ledger),tokens=summary['tokens'],
        response_errors=dict(Counter(r['error'] for r in responses if 'error' in r)),
        provider_statuses=dict(Counter(e['status'] for e in ledger)),retained_additions=len(additions)),
        rows=rows,retained_additions=additions,limitation=__doc__,
        plan_sha256=sha(root/'plan.json'),responses_sha256=sha(root/'live.jsonl'),auditor_sha256=sha(Path(__file__)))
    with (root/'audit-with-candidate-stage.json').open('x') as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2);stream.write('\n')
    print(json.dumps(result['summary'],ensure_ascii=False))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--gold-csv',type=Path,default=Path('data/raw/personamem-v2.csv'))
    main(parser.parse_args())
