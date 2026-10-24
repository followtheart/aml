"""Audit paired review judgments and fixed-cohort selection transport.

No answer accuracy is measured. Main admission, review batches and overflow
candidate membership are frozen; changing selections can alter these in a live
Search. All-source positive counts do not measure semantic precision. Unknown
provider usage remains unknown. Reference IDs enter only this offline audit.
"""
import argparse
import ast
from collections import Counter
import copy
import csv
import hashlib
import json
from pathlib import Path
import re

from probe_candidate_sources import records, load_jobs, digest, spans, summarize
from app import fine_evidence
import audit_https_funnel as transport
from audit_https_stages import reference_set_coverage


def status(j):
    return 'unreviewed' if j is None else 'invalid' if not j['valid'] else 'positive' if j['useful'] else 'negative'


def selected(t, judgments):
    """Apply unchanged deletion and dependency policies to frozen review sets."""
    main = t['cascade']['listwise']
    kept = set(main['selected_ids'])
    for mid, j in judgments.get('main', {}).items():
        if status(j) == 'negative':
            kept.discard(mid)
        else:
            kept.add(mid)
    fine = {r['id']: r for r in t['ranked'] if r['id'] in t['cascade']['fine']['selected_ids']}
    useful = {mid for mid, j in judgments.get('overflow', {}).items() if status(j) == 'positive'}
    accepted = set()
    for mid in useful:
        group, missing = fine_evidence.closure(fine[mid], fine)
        if not missing and group <= kept | useful:
            accepted.add(mid)
    return dict(main=kept, overflow=kept | accepted)


def main(args):
    source = args.diagnostic
    plan = json.loads(source.with_suffix('.plan.json').read_text())
    if not source.with_suffix('.exit-code').exists():
        raise ValueError('Diagnostic is still running')
    live = records(source.with_suffix('.live.jsonl'))
    report = json.loads(source.read_text())
    run = Path(plan['run'])
    jobs = load_jobs(run)
    if [m for m, *_ in jobs] != plan['paired_requests'] or live != report['live']:
        raise ValueError('Source requests or response records changed')
    if len(live) != plan['expected_calls'] or len({(r['job_id'], r['arm']) for r in live}) != len(live):
        raise ValueError('Incomplete or duplicate pair cohort')
    if hashlib.sha256(Path(__file__).with_name('probe_candidate_sources.py').read_bytes()).hexdigest() != plan['script_sha256']:
        raise ValueError('Diagnostic script changed')
    decoded = 0
    decisions = {arm: {} for arm in ('control','treatment')}
    original = {}
    for row in live:
        meta, _, _, built = jobs[row['job_id']]
        if any(row[k] != meta[k] for k in ('qa_id','stage','batch')):
            raise ValueError('Response/request identity differs')
        if 'error' not in row:
            clean = copy.deepcopy(row['payload'])
            ids, judgments = spans.decode_partial(clean, built)
            if sorted(ids) != row['selected_ids'] or judgments != row['judgments']:
                raise ValueError('Response failed independent validation')
            decoded += 1
        by_id = {j['candidate_id']: j for j in row.get('judgments', [])}
        stage = decisions[row['arm']].setdefault(row['qa_id'], {}).setdefault(row['stage'], {})
        for mid in meta['candidate_ids']:
            if mid in stage:
                raise ValueError('Candidate reviewed twice in same stage')
            stage[mid] = by_id.get(mid)
    metrics = []
    for line in source.with_suffix('.provider.log').read_text().splitlines():
        if 'provider_call ' not in line:
            continue
        m = dict(re.findall(r'(\w+)=([^\s]+)', line))
        metrics.append(m)
    for row in live:
        calls = [m for m in metrics if m['stage'] == f'diagnostic.candidate_sources.{row["job_id"]}.{row["arm"]}']
        if len(calls) != row['calls'] or sum(int(m['total_tokens']) for m in calls) != row['tokens']:
            raise ValueError('Recorded provider accounting differs')
    csv.field_size_limit(32*1024*1024)
    with args.gold_csv.open() as f:
        gold = list(csv.DictReader(f))
    raw = {}
    for event in records(run/'server-memory-debug.jsonl'):
        raw.setdefault(event['user_id'], {}).update({transport.identity(s): s for s in event['source_messages']})
    traces = {t['search_id']: t for t in records(run/'server-search-debug.jsonl')}
    stages = {r['qa_id']: r for r in json.loads((run/'funnel/stage-transport.json').read_text())['rows']}
    rows, changes = [], []
    for result in records(run/'completed/data/results/personamem-v2-32k.jsonl'):
        q = result['qa_id']
        t = traces[result['search_id']]
        s = stages[q]
        original[q] = {}
        reviews = [('main', r) for r in t['cascade']['listwise'].get('recovery',{}).get('batches',[])]
        reviews.append(('overflow', t['cascade'].get('overflow',{}).get('review',{})))
        for stage, review in reviews:
            by_id = {j['candidate_id']: j for j in review.get('judgments',[])}
            original[q].setdefault(stage,{}).update({mid: by_id.get(mid) for mid in review.get('candidate_ids',[])})
        historical = selected(t, original[q])
        if (historical['main'] != set(t['cascade']['listwise']['selected_ids'])
                or historical['overflow'] != set(t['cascade']['overflow']['selected_ids'])):
            raise ValueError('Historical selection did not reproduce')
        snippet = gold[int(q)]['related_conversation_snippet']
        try:
            snippet = json.loads(snippet)
        except ValueError:
            snippet = ast.literal_eval(snippet)
        turns = [transport.norm(z['content']) for z in snippet if z['role']=='user']
        packet_items = {r['id']: r.get('_packet_item',{}) for r in t['ranked']}
        def coverage(ids):
            cards = [dict(z, id=f'{mid}:{i}',text=z.get('content','')) for mid in sorted(ids)
                     for i,z in enumerate(packet_items[mid].get('sources',[]))]
            cov = transport.turn_coverage(turns,cards,raw[t['user_id']])
            return dict(full_turns=sum(z['full'] for z in cov),
                        **reference_set_coverage(s['raw_turn_coverage'],cov))
        for name, historical_stage in [('main','listwise_selected'),('overflow','overflow_selected')]:
            expected = s['visible'][historical_stage]
            if any(expected[k] != v for k,v in coverage(historical[name]).items()):
                raise ValueError('Historical source transport did not reproduce')
        arms = {}
        for arm in ('control','treatment'):
            choices = selected(t,decisions[arm].get(q,{}))
            arms[arm] = {stage:dict(coverage(ids), ids=sorted(ids)) for stage,ids in choices.items()}
        rows.append(dict(qa_id=q, arms=arms))
        carriers = set().union(*(set(ids) for ids in s['carrier_ids'].values()))
        for stage in ('main','overflow'):
            a,b = [decisions[arm].get(q,{}).get(stage,{}) for arm in ('control','treatment')]
            if a.keys()!=b.keys():
                raise ValueError('Paired candidate sets differ')
            for mid in a:
                if status(a[mid])!=status(b[mid]):
                    changes.append(dict(qa_id=q,stage=stage,candidate_id=mid,reference_carrier=mid in carriers,
                        control=status(a[mid]),treatment=status(b[mid]),control_judgment=a[mid],treatment_judgment=b[mid]))
    summary = summarize(jobs,live)
    summary['by_layout'] = {
        label: summarize([j for j in jobs if j[0]['layout_applied'] == applied],
            [r for r in live if jobs[r['job_id']][0]['layout_applied'] == applied])
        for label, applied in [('changed', True), ('unchanged_fallback', False)]}
    summary.update(decoded_responses=decoded,historical_selections_verified=len(rows),
        provider_statuses=dict(Counter(m['status'] for m in metrics)),
        fixed_selection_transport={arm:{stage:dict(
            reference_questions=sum(r['arms'][arm][stage]['full_turns']>0 for r in rows),
            full_turns=sum(r['arms'][arm][stage]['full_turns'] for r in rows),
            complete_sets=sum(r['arms'][arm][stage]['available_set_complete'] for r in rows))
            for stage in ('main','overflow')} for arm in ('control','treatment')},
        reference_transitions=[{k:v for k,v in c.items() if not k.endswith('_judgment')} for c in changes if c['reference_carrier']],
        changed_outcomes=dict(Counter(c['control']+' -> '+c['treatment'] for c in changes)))
    output = dict(summary=summary,changes=changes,rows=rows,limitation=__doc__,
                  auditor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  diagnostic_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    with args.output.open('x') as f:
        json.dump(output,f,ensure_ascii=False,indent=2)
        f.write('\n')
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diagnostic',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gold-csv',type=Path,default=Path('data/raw/personamem-v2.csv'))
    main(parser.parse_args())
