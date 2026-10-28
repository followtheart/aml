"""Fixed HTTPS candidates: upper bound before overflow usefulness judgment."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sqlite3
from unittest.mock import patch

from evaluate_recall_context import CSV, packet_coverage, records, reference_turns
from app import budget, config, listwise_overflow, listwise_recovery, schemas


def compact_request(request):
    """Diagnostic only: remove JSON syntax whitespace, never source characters."""
    request=dict(request)
    marker='{"targets":'
    start=request['prompt'].rfind('\n'+marker)
    if start<0:
        raise ValueError('Expected the span-reference JSON payload')
    start+=1
    original=request['prompt'][start:]
    payload=json.loads(original)
    compact=json.dumps(payload,ensure_ascii=False,separators=(',',':'))
    if json.loads(compact)!=payload:
        raise ValueError('Compaction altered the evidence payload')
    request['prompt']=request['prompt'][:start]+compact
    request['reservation']+=len(compact.encode())-len(original.encode())
    return request


def main(args):
    path = args.run / 'server-search-debug.jsonl'
    traces = {t['search_id']: t for t in records(path)}
    conversations = {r['conversation_id']: r for r in records(args.prepared)}
    csv.field_size_limit(32*1024*1024)
    with CSV.open() as source:
        gold = list(csv.DictReader(source))
    raw = {}
    with sqlite3.connect((args.run/'memory.db').resolve().as_uri()+'?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        for row in db.execute('SELECT * FROM source_messages'):
            raw.setdefault(row['user_id'], {})[row['request_id'], row['message_index']] = dict(row)
    rows = []
    reproduced=0
    for result in records(args.run/'completed/data/results/personamem-v2-32k.jsonl'):
        t = traces[result['search_id']]
        c = t['cascade']
        by_id = {v['id']: v for v in t['ranked']}
        fine = [by_id[mid] for mid in c['fine']['selected_ids']]
        submitted = [by_id[mid] for mid in c['listwise']['candidate_ids']]
        selected = [by_id[mid] for mid in c['listwise']['selected_ids']]
        req = schemas.SearchRequest(user_id=t['user_id'], query=t['query'], options=t['options'])
        # Historical budget at admission is an optimistic ceiling: primary
        # listwise and deletion review have not yet consumed their shares.
        headroom = c['admission_budget']
        omissions=list(c['omitted'])
        previous=c.get('overflow')
        if previous:
            # Successful additions remove their original byte omission from
            # the final trace. Reconstruct it from the overflow proposal trace.
            original_ids=set(previous['proposed_ids']) | {d['id'] for d in previous['decisions']}
            omitted_ids={o['id'] for o in omissions if o['stage']=='listwise' and o['reason']=='prompt_budget'}
            assert not original_ids & {v['id'] for v in submitted}
            omissions.extend(dict(id=mid,stage='listwise',reason='prompt_budget') for mid in sorted(original_ids-omitted_ids))
        with patch.multiple(config, **t['versions']['settings']), budget.scope(
                seconds=headroom['deadline_seconds'], calls=headroom['call_headroom'],
                tokens=headroom['token_headroom']), patch('socket.socket.connect', side_effect=AssertionError('Offline')):
            if previous:
                baseline,_,_=listwise_overflow._propose(req,fine,submitted,selected,omissions)
                if [v['id'] for v in baseline]!=previous.get('baseline_ids',previous['proposed_ids']):
                    raise ValueError('Recorded baseline proposal did not reproduce with admission headroom')
                reproduced+=1
            original_request=listwise_recovery.request_for
            make_request=(lambda *a,**k:compact_request(original_request(*a,**k))) if args.compact_review else original_request
            with patch.object(listwise_recovery,'MAX_PROMPT_BYTES',args.review_bytes), patch.object(config,'LISTWISE_OVERFLOW_EXTEND',args.extend), patch.object(listwise_recovery,'request_for',make_request):
                pending, groups, decisions, metadata = listwise_overflow.proposal(req, fine, submitted, selected, omissions)
                if previous and args.extend:
                    assert [v['id'] for v in pending][:len(baseline)] == previous.get('baseline_ids',previous['proposed_ids'])
        conv = conversations[result['conversation_id']]
        q = next(q for q in conv['qa'] if q['id'] == result['qa_id'])
        assert t['query'] == q['question'] and t['options'] == q['options']
        turns = reference_turns(q, conv['source_history'], gold[int(result['qa_id'])])
        def coverage(items):
            return packet_coverage(turns, [v['_packet_item'] for v in items], raw[t['user_id']])
        actual_added=[by_id[mid] for mid in (previous or {}).get('appended_ids',[])]
        original_added=[v for v in actual_added if v['id'] not in (previous or {}).get('extra_ids',[])]
        rows.append(dict(qa_id=result['qa_id'], control=coverage(selected), upper_bound=coverage(selected+pending),
                         baseline_upper_bound=coverage(selected+[by_id[mid] for mid in metadata['baseline_ids']]),
                         realized=coverage(selected+actual_added), without_extra_additions=coverage(selected+original_added),
                         input=coverage(submitted), input_upper_bound=coverage(submitted+pending),
                         proposed_ids=[v['id'] for v in pending], decisions=decisions,
                         admission_budget=headroom, groups=groups, proposal=metadata))
    summary = dict(questions=len(rows),review_bytes=args.review_bytes,extend=args.extend,compact_review=args.compact_review,reproduced_baseline_proposals=reproduced,
                   proposed_candidates=sum(len(r['proposed_ids']) for r in rows),
                   affected_questions=sum(bool(r['proposed_ids']) for r in rows),
                   visible={k:sum(r[k]['visible'] for r in rows) for k in ('input','input_upper_bound','control','baseline_upper_bound','upper_bound','realized','without_extra_additions')},
                   turns={k:sum(r[k]['full_turns'] for r in rows) for k in ('input','input_upper_bound','control','baseline_upper_bound','upper_bound','realized','without_extra_additions')},
                   gained=[r['qa_id'] for r in rows if r['upper_bound']['visible'] and not r['control']['visible']])
    with args.output.open('x') as out:
        json.dump(dict(summary=summary, rows=rows, trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                       limitation='Optimistic admission budget; no model judgment or final packing; not recall improvement.'),out,indent=2)
    print(json.dumps(summary))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--prepared',type=Path,default=Path('memory_system/data/prepared/personamem-v2-32k.jsonl'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--extend',action='store_true')
    p.add_argument('--compact-review',action='store_true',help='Offline JSON syntax-whitespace diagnostic; source text unchanged')
    p.add_argument('--review-bytes',type=int,default=12000,choices=[12000,16000,20000])
    args=p.parse_args()
    if args.extend and args.review_bytes!=12000:
        p.error('--extend preserves the original 12000-byte baseline; do not combine with --review-bytes')
    main(args)
