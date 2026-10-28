"""Offline bound for one additional batch of never-reviewed fine candidates.

Preserve the completed main and first overflow decisions. Reuse the original
12 KB / four-candidate policy for remaining prompt-byte omissions, with the
recorded end-of-Search call/token remainder as a conservative budget proxy.
No provider calls, relevance judgments, final packing or answer gains are measured.
"""
import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import budget, config, listwise_overflow, listwise_recovery, schemas
from evaluate_recall_context import CSV, packet_coverage, reference_turns
from evaluation_model_guard import check_versions


def read_snapshot(path, partial=False):
    data = path.read_bytes()
    if partial and data and not data.endswith(b'\n'):
        data = data[:data.rfind(b'\n') + 1]
    return ([json.loads(line) for line in data.splitlines() if line.strip()],
            dict(path=str(path.resolve()), prefix_bytes=len(data),
                 prefix_sha256=hashlib.sha256(data).hexdigest()))


def propose(trace, result):
    check_versions(trace['versions'], 'source Search')
    cascade = trace['cascade']
    previous = cascade['overflow']
    by_id = {c['id']: c for c in trace['ranked']}
    fine = [by_id[mid] for mid in cascade['fine']['selected_ids']]
    submitted = [by_id[mid] for mid in cascade['listwise']['candidate_ids']]
    selected = [by_id[mid] for mid in cascade['listwise']['selected_ids']]
    req = schemas.SearchRequest(user_id=trace['user_id'], query=trace['query'], options=trace['options'])
    omissions = list(cascade['omitted'])
    original_ids = set(previous['proposed_ids']) | {d['id'] for d in previous['decisions']}
    omitted_ids = {o['id'] for o in omissions if o['stage'] == 'listwise' and o['reason'] == 'prompt_budget'}
    omissions.extend(dict(id=mid, stage='listwise', reason='prompt_budget')
                     for mid in sorted(original_ids - omitted_ids))
    headroom = cascade['admission_budget']
    settings = trace['versions']['settings']
    with patch.multiple(config, **settings), patch('socket.socket.connect', side_effect=AssertionError('Offline')):
        if config.LISTWISE_OVERFLOW_EXTEND or not config.LISTWISE_SPAN_REFS:
            raise ValueError('Require original span-reference overflow policy')
        with budget.scope(seconds=max(.01, headroom['deadline_seconds']),
                          calls=headroom['call_headroom'], tokens=headroom['token_headroom']):
            baseline, _, _ = listwise_overflow._propose(req, fine, submitted, selected, omissions)
        if [c['id'] for c in baseline] != previous['proposed_ids']:
            raise ValueError('Original proposal did not reconstruct')
        kept = selected + [by_id[mid] for mid in previous['appended_ids']]
        if [c['id'] for c in kept] != previous['selected_ids']:
            raise ValueError('Original overflow selection did not reconstruct')
        elapsed = (datetime.fromisoformat(trace['finished_at']) -
                   datetime.fromisoformat(trace['started_at'])).total_seconds()
        if elapsed < 0:
            raise ValueError('Search timestamps are reversed')
        # Total Search elapsed overestimates time before the second review.
        # This wall-clock proxy is diagnostic only; production needs monotonic checks.
        seconds = max(0, min(config.RERANK_DEADLINE_SECONDS - elapsed,
                             config.SEARCH_DEADLINE_SECONDS - elapsed - 2))
        remainder = dict(calls=config.SEARCH_MAX_CALLS - result['coverage_manifest']['provider_calls'],
                         tokens=config.SEARCH_MAX_TOKENS - result['coverage_manifest']['provider_tokens'],
                         conservative_seconds_proxy=seconds)
        pending, groups, decisions = [], [], []
        if remainder['calls'] > 0 and remainder['tokens'] > 0 and seconds > 0:
            with budget.scope(seconds=seconds, calls=remainder['calls'], tokens=remainder['tokens']):
                pending, groups, decisions = listwise_overflow._propose(
                    req, fine, submitted + baseline, kept, omissions)
        if set(c['id'] for c in pending) & ({c['id'] for c in submitted} | set(previous['proposed_ids'])):
            raise ValueError('Second proposal repeats a previously submitted candidate')
        request = None
        if pending:
            cards = [dict(candidate_id=c['id'], sources=listwise_recovery.sources_for(c)) for c in pending]
            request = listwise_recovery.request_for(req, cards)
            if (len(request['prompt'].encode()) > 12000 or len(pending) > 4
                    or request['reservation'] > remainder['tokens']):
                raise ValueError('Second proposal exceeds its recorded budget proxy')
    return dict(qa_id=result['qa_id'], conversation_id=result['conversation_id'],
                search_id=trace['search_id'], baseline_ids=previous['proposed_ids'],
                original_selected_ids=previous['selected_ids'], proposed_ids=[c['id'] for c in pending],
                groups=groups, decisions=decisions, remainder=remainder,
                input_bytes=len(request['prompt'].encode()) if request else 0,
                request_reservation=request['reservation'] if request else 0,
                request_sha256=hashlib.sha256(request['prompt'].encode()).hexdigest() if request else None)


def main(args):
    if not args.allow_partial and (args.run / 'exit-code').read_text().strip() != '0':
        raise ValueError('Require successful completion or explicit partial diagnostic')
    results, result_hash = read_snapshot(args.run / 'completed/data/results/personamem-v2-32k.jsonl', args.allow_partial)
    searches, search_hash = read_snapshot(args.run / 'server-search-debug.jsonl', args.allow_partial)
    if not results or len({(r['conversation_id'], r['qa_id']) for r in results}) != len(results):
        raise ValueError('Empty or duplicate result set')
    if len({r['run_id'] for r in results}) != 1:
        raise ValueError('Mixed run identities')
    traces = {t['search_id']: t for t in searches}
    if len(traces) != len(searches):
        raise ValueError('Duplicate Search identity')
    manifest = json.loads((args.run / 'isolated-manifest.json').read_text())
    data_sha256 = hashlib.sha256(args.prepared.read_bytes()).hexdigest()
    if manifest['data_sha256'] != data_sha256:
        raise ValueError('Prepared data differs from source run')
    prepared = read_snapshot(args.prepared)[0]
    conversations = {c['conversation_id']: c for c in prepared}
    expected = manifest.get('questions')
    if expected is None:
        evaluation = manifest['evaluation']
        if evaluation['AML_EVAL_LIMIT'] != 0:
            raise ValueError('Fresh-ingestion diagnostic requires complete histories')
        expected = sum(len(c['qa']) for c in prepared[:evaluation['AML_EVAL_CONVS'] or None])
    if len(results) > expected or not args.allow_partial and len(results) != expected:
        raise ValueError('Result count differs from expected cohort')
    proposals = []
    for result in results:
        if result.get('error_stage') or result.get('fake') is not False or result['transport'] != 'https':
            raise ValueError('Invalid source result; do not silently exclude failures')
        trace = traces[result['search_id']]
        if trace['user_id'] != result['user_id'] or trace['packet_hash'] != result['packet_hash']:
            raise ValueError('Search/result identity mismatch')
        conv = conversations[result['conversation_id']]
        qa = next(q for q in conv['qa'] if q['id'] == result['qa_id'])
        if trace['query'] != qa['question'] or trace['options'] != qa['options']:
            raise ValueError('Question/options mismatch')
        proposals.append(propose(trace, result))
    # Labels are loaded only after every proposed request has been constructed.
    csv.field_size_limit(32 * 1024 * 1024)
    with CSV.open() as stream:
        gold = list(csv.DictReader(stream))
    raw = {}
    with sqlite3.connect(f'file:{(args.run / "memory.db").resolve()}?mode=ro', uri=True) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute('BEGIN')
        for row in conn.execute('SELECT * FROM source_messages'):
            raw.setdefault(row['user_id'], {})[row['request_id'], row['message_index']] = dict(row)
    for proposal, result in zip(proposals, results):
        trace = traces[result['search_id']]
        by_id = {c['id']: c for c in trace['ranked']}
        conv = conversations[result['conversation_id']]
        qa = next(q for q in conv['qa'] if q['id'] == result['qa_id'])
        turns = reference_turns(qa, conv['source_history'], gold[int(result['qa_id'])])
        actual = trace['returned']
        extra = [by_id[mid]['_packet_item'] for mid in proposal['proposed_ids']]
        proposal['actual_final'] = packet_coverage(turns, actual, raw[trace['user_id']])
        proposal['unpacked_upper_bound'] = packet_coverage(turns, actual + extra, raw[trace['user_id']])
    summary = dict(questions=len(proposals), expected_questions=expected, progress_snapshot=args.allow_partial,
                   reproduced_baseline_proposals=len(proposals),
                   additional_batches=sum(bool(p['proposed_ids']) for p in proposals),
                   additional_candidates=sum(len(p['proposed_ids']) for p in proposals),
                   total_prompt_bytes=sum(p['input_bytes'] for p in proposals),
                   reservation_sum=sum(p['request_reservation'] for p in proposals),
                   actual_reference_questions=sum(p['actual_final']['visible'] for p in proposals),
                   upper_bound_reference_questions=sum(p['unpacked_upper_bound']['visible'] for p in proposals),
                   actual_reference_turns=sum(p['actual_final']['full_turns'] for p in proposals),
                   upper_bound_reference_turns=sum(p['unpacked_upper_bound']['full_turns'] for p in proposals),
                   gained_questions=[p['qa_id'] for p in proposals if p['unpacked_upper_bound']['visible'] and not p['actual_final']['visible']])
    report = dict(summary=summary, rows=proposals, inputs=[result_hash, search_hash],
                  prepared_sha256=data_sha256,
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), limitation=__doc__,
                  budget_limitation='End-of-Search remainder and total elapsed are conservative diagnostic proxies, not an exact replay of live reservations or monotonic timing. Upper bounds append all proposed units without judgment or final packet limits.')
    with args.output.open('x') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--prepared', type=Path, default=Path('data/prepared/personamem-v2-32k.jsonl'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-partial', action='store_true')
    main(parser.parse_args())
