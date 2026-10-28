"""Replay two existing CE boundary seats on current frozen fine/main inputs.

Reuse semantic_seats rather than inventing a new ranking rule. Keep the core,
whole sources, dependencies and total capacity. Only offline reference transport
is measured; live main decisions, final packets and answers are not replayed.
"""
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import config, fine_evidence
from evaluate_recall_context import CSV, packet_coverage, reference_turns
from evaluation_model_guard import check_versions
from probe_coarse_fine_core import admit
from probe_fine_topic_context import semantic_seats
from probe_overflow_second_batch import read_snapshot


def prepare(trace):
    cascade = trace['cascade']
    if cascade['cross_encoder']['status'] != 'ok':
        raise ValueError('Need complete CE scores; do not drop incomplete questions')
    by_id = {c['id']: c for c in trace['ranked']}
    coarse = [by_id[mid] for mid in cascade['coarse']['selected_ids']]
    frozen = copy.deepcopy(coarse)
    with patch.multiple(config, **trace['versions']['settings']), patch('socket.socket.connect', side_effect=AssertionError('Offline')):
        if not config.FINE_EVIDENCE_ADAPTIVE or config.FINE_TOPIC_CONTEXT or config.FINE_SEMANTIC_RESERVE:
            raise ValueError('Unsupported historical fine variant')
        baseline_ids = set(cascade['coarse']['baseline_ids'])
        pool = [c for c in coarse if c['id'] in baseline_ids]
        extra = sorted([c for c in coarse if c['id'] not in baseline_ids], key=fine_evidence.key, reverse=True)
        allowed = baseline_ids | {c['id'] for c in extra[:1]}
        fused = {c['id'] for c in sorted(pool, key=lambda c: -(c.get('_fused') or 0))[:config.CASCADE_FUSED_RESERVE]}
        if not config.COARSE_EVIDENCE_EXTEND or not config.FINE_COARSE_CORE_QUOTA:
            raise ValueError('Requires historical extended coarse/core quota')
        before, selection = fine_evidence.propose(coarse, min(config.CASCADE_FINE_LIMIT, len(coarse)),
            trace['coverage_manifest']['coverage']['requirements'], fused_ids=fused,
            extra_limit=config.FINE_EXTRA_LIMIT, core_pool_ids=allowed,
            soft_promotion_limit=1 if config.FINE_WEAK_PROMOTION_ONE else None)
        main, size = admit(trace, before, selection)
        if ([c['id'] for c in before] != cascade['fine']['selected_ids'] or
                [c['id'] for c in main] != cascade['listwise']['candidate_ids'] or size != cascade['listwise']['input_bytes']):
            raise ValueError('Historical fine/main admission mismatch')
        after = semantic_seats(coarse, before, selection, 2)
        revised = copy.deepcopy(selection)
        chosen_ids = {c['id'] for c in after}
        core = set(selection['adaptive']['core_ids'])
        groups = []
        additions = []
        for candidate in after:
            group, missing = fine_evidence.closure(candidate, by_id)
            if missing or not group <= chosen_ids:
                raise ValueError('Proposed selection has incomplete dependencies')
            if len(group) > 1:
                groups.append(sorted(group))
            if candidate['id'] not in core:
                additions.append(dict(candidate_id=candidate['id'], ids=[c['id'] for c in after if c['id'] in group - core],
                                      group_ids=sorted(group), reason='boundary_seat_diagnostic', requirement_ids=[]))
        revised['adaptive'].update(additions=additions, path_groups=groups,
                                   proposed_ids=[c['id'] for c in after])
        new_main, new_size = admit(trace, after, revised)
        if not core <= chosen_ids or len(after) > selection['adaptive']['capacity'] or new_size > config.RERANK_MAX_PROMPT_BYTES:
            raise ValueError('Core or capacity changed')
        if coarse != frozen:
            raise ValueError('Prepared source content changed')
    return dict(control=dict(fine=before, main=main, bytes=size),
                treatment=dict(fine=after, main=new_main, bytes=new_size))


def main(args):
    if not args.allow_partial and (args.run / 'exit-code').read_text().strip() != '0':
        raise ValueError('Require terminal success or explicit progress mode')
    results, result_hash = read_snapshot(args.run / 'completed/data/results/personamem-v2-32k.jsonl', args.allow_partial)
    searches, search_hash = read_snapshot(args.run / 'server-search-debug.jsonl', args.allow_partial)
    manifest = json.loads((args.run / 'isolated-manifest.json').read_text())
    if hashlib.sha256(args.prepared.read_bytes()).hexdigest() != manifest['data_sha256']:
        raise ValueError('Prepared data identity mismatch')
    traces = {s['search_id']: s for s in searches}
    if len(traces) != len(searches) or len({r['qa_id'] for r in results}) != len(results) or not results:
        raise ValueError('Invalid source inventory')
    if not args.allow_partial and len(results) != manifest['questions']:
        raise ValueError('Incomplete frozen cohort')
    conversations = {c['conversation_id']: c for c in read_snapshot(args.prepared)[0]}
    jobs = []
    for result in results:
        if result.get('error_stage') or result.get('fake') is not False or result['transport'] != 'https':
            raise ValueError('Invalid result')
        trace = traces[result['search_id']]
        check_versions(trace['versions'], 'source Search')
        conv = conversations[result['conversation_id']]
        qa = next(q for q in conv['qa'] if q['id'] == result['qa_id'])
        if (trace['user_id'] != result['user_id'] or trace['packet_hash'] != result['packet_hash']
                or trace['query'] != qa['question'] or trace['options'] != qa['options']):
            raise ValueError('Question/packet identity mismatch')
        jobs.append((result, trace, qa, conv, prepare(trace)))
    # Reference scoring follows all selection, and never enters prepare().
    csv.field_size_limit(32 * 1024 * 1024)
    with CSV.open() as stream:
        gold = list(csv.DictReader(stream))
    raw = {}
    for event in read_snapshot(args.run / 'server-memory-debug.jsonl', args.allow_partial)[0]:
        raw.setdefault(event['user_id'], {}).update({(s['request_id'], s['message_index']): s for s in event['source_messages']})
    rows = []
    for result, trace, qa, conv, variants in jobs:
        turns = reference_turns(qa, conv['source_history'], gold[int(result['qa_id'])])
        row = dict(qa_id=result['qa_id'], variants={})
        for arm, variant in variants.items():
            row['variants'][arm] = dict(main_bytes=variant['bytes'])
            for stage in ('fine', 'main'):
                row['variants'][arm][stage] = dict(ids=[c['id'] for c in variant[stage]],
                    coverage=packet_coverage(turns, [c['_packet_item'] for c in variant[stage]], raw[trace['user_id']]))
        rows.append(row)
    summary = dict(questions=len(rows), progress_snapshot=args.allow_partial, reproduced_fine_and_main=len(rows), stages={})
    for stage in ('fine', 'main'):
        summary['stages'][stage] = dict(
            reference_questions={arm:sum(r['variants'][arm][stage]['coverage']['visible'] for r in rows) for arm in ('control','treatment')},
            reference_turns={arm:sum(r['variants'][arm][stage]['coverage']['full_turns'] for r in rows) for arm in ('control','treatment')},
            gained=[r['qa_id'] for r in rows if r['variants']['treatment'][stage]['coverage']['visible'] and not r['variants']['control'][stage]['coverage']['visible']],
            lost=[r['qa_id'] for r in rows if r['variants']['control'][stage]['coverage']['visible'] and not r['variants']['treatment'][stage]['coverage']['visible']],
            turn_losses=[r['qa_id'] for r in rows if r['variants']['treatment'][stage]['coverage']['full_turns'] < r['variants']['control'][stage]['coverage']['full_turns']])
    with args.output.open('x') as stream:
        json.dump(dict(summary=summary, rows=rows, inputs=[result_hash,search_hash], limitation=__doc__,
                       script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()), stream, indent=2)
        stream.write('\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--prepared', type=Path, default=Path('data/prepared/personamem-v2-32k.jsonl'))
    parser.add_argument('--allow-partial', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
