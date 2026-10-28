"""Hold each Search's upstream decisions fixed and replay packing without later batches.

No model calls or answer scores. Full packet replay must match the live response.
Only singleton later additions are supported: merged dependency groups require
pre-overflow group snapshots and must never be guessed from the final packet.
"""
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
from unittest.mock import patch


def records(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def main(args):
    root = args.run.resolve()
    snapshot = root / 'source'
    sys.path.insert(0, str(snapshot / 'memory_system'))
    from app import config, schemas, search_pipeline
    from evaluate_recall_context import packet_coverage, reference_turns
    from funnel_gold_snippet import CSV
    plan = json.loads((root / 'plan.json').read_text())
    if plan['toggle'] != 'LISTWISE_OVERFLOW_BATCHED':
        raise ValueError('Require the batched overflow experiment')
    for name, sha in plan['source_files'].items():
        if hashlib.sha256((snapshot/name).read_bytes()).hexdigest() != sha:
            raise ValueError('Frozen source changed: ' + name)
    arm = root / 'treatment'
    results = records(arm / 'completed/data/results/personamem-v2-32k.jsonl')
    if not args.progress:
        if len(results) != plan['questions'] or not (root/'exit-code').exists():
            raise ValueError('Require all completed questions')
    traces = {t['search_id']: t for t in records(arm/'server-search-debug.jsonl')}
    repacked = []
    for row in results:
        t = traces[row['search_id']]
        overflow = t['cascade']['overflow']
        later = {mid for b in overflow['batches'][1:] for mid in b['appended_ids']}
        if any(len(g)>1 and set(g)&later for g in overflow['groups']+t['evidence_groups']):
            raise ValueError('Cannot infer pre-overflow atomic groups')
        def pack(first_only):
            ranked = copy.deepcopy(t['ranked'])
            p = copy.deepcopy(t['plan'])
            p.update(_cascade=copy.deepcopy(t['cascade']), _evidence_groups=copy.deepcopy(t['evidence_groups']),
                     _coverage_requirements=t['coverage_manifest']['coverage']['requirements'],
                     _routes=t['routes'], _query_specs=t['query_specs'])
            if first_only:
                for c in ranked:
                    if c['id'] in later:
                        c['_cascade_selected'] = False
            req = schemas.SearchRequest(user_id=t['user_id'], query=t['query'], options=t['options'],
                top_k=t['top_k'], evidence_token_budget=t['evidence_token_budget'],
                reference_time=t.get('reference_time'), include_sensitive=t['requested_include_sensitive'])
            with patch.multiple(config, **t['versions']['settings']), \
                    patch('socket.socket.connect', side_effect=AssertionError('Offline audit')):
                selected = search_pipeline._select_evidence(req, p, ranked, defer_limits=True)
                packet, digest, _ = search_pipeline._pack_evidence(None, req, p, selected, t['anchor_time'], ranked)
                return [schemas.SearchItem(**v).model_dump() for v in packet], digest
        actual, digest = pack(False)
        if actual != t['returned'] or digest != t['packet_hash']:
            raise ValueError('Live packet did not reproduce for Q'+str(row['qa_id']))
        first, _ = pack(True)
        repacked.append((row, t, actual, first, sorted(later)))
    # Gold is loaded only after all deterministic packing comparisons.
    prepared = Path(json.loads((arm/'isolated-manifest.json').read_text())['data'])
    if hashlib.sha256(prepared.read_bytes()).hexdigest() != plan['data_sha256']:
        raise ValueError('Prepared data changed')
    convs = {c['conversation_id']:c for c in records(prepared)}
    csv.field_size_limit(32 * 1024 * 1024)
    with CSV.open() as stream:
        gold = list(csv.DictReader(stream))
    with sqlite3.connect(f'file:{arm/"memory.db"}?mode=ro', uri=True) as db:
        db.row_factory=sqlite3.Row
        raw={}
        for r in db.execute('SELECT * FROM source_messages'):
            raw.setdefault(r['user_id'],{})[r['request_id'],r['message_index']]=dict(r)
    rows=[]
    for row, t, actual, first, later in repacked:
        conv=convs[row['conversation_id']]
        q=next(q for q in conv['qa'] if q['id']==row['qa_id'])
        turns=reference_turns(q,conv['source_history'],gold[int(row['qa_id'])])
        a=packet_coverage(turns,actual,raw[t['user_id']])
        b=packet_coverage(turns,first,raw[t['user_id']])
        rows.append(dict(qa_id=row['qa_id'],conversation_id=row['conversation_id'],
            later_additions=later,first_only=b,actual=a,
            first_bytes=sum(len(x['content'].encode()) for x in first),
            actual_bytes=sum(len(x['content'].encode()) for x in actual),
            first_ids=[x['id'] for x in first],actual_ids=[x['id'] for x in actual]))
    summary=dict(questions=len(rows),expected=plan['questions'],progress_snapshot=args.progress,
        exact_packet_replays=len(rows),additional_candidates=sum(len(r['later_additions']) for r in rows),
        affected_questions=sum(bool(r['later_additions']) for r in rows),
        first_only_reference_questions=sum(r['first_only']['visible'] for r in rows),
        actual_reference_questions=sum(r['actual']['visible'] for r in rows),
        first_only_reference_turns=sum(r['first_only']['full_turns'] for r in rows),
        actual_reference_turns=sum(r['actual']['full_turns'] for r in rows),
        gained=[r['qa_id'] for r in rows if r['actual']['visible'] and not r['first_only']['visible']],
        lost=[r['qa_id'] for r in rows if r['first_only']['visible'] and not r['actual']['visible']],
        fewer_reference_turns=[r['qa_id'] for r in rows if r['actual']['full_turns']<r['first_only']['full_turns']],
        first_bytes=sum(r['first_bytes'] for r in rows),actual_bytes=sum(r['actual_bytes'] for r in rows),
        limitation=__doc__)
    target=root/('progress-overflow-attribution.json' if args.progress else 'overflow-attribution.json')
    target.write_text(json.dumps(dict(summary=summary,rows=rows,
        auditor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--progress',action='store_true')
    main(p.parse_args())
