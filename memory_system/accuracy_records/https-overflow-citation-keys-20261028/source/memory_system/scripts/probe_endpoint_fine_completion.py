"""Offline endpoint-preserving whole-fine completion under original capacity.

Replay the exact original endpoint, then append complete prepared fine units
with novel visible user text. Existing evidence/constraints stay unchanged.
All reference annotations are joined after packets are fixed. No providers.
"""
import argparse
import ast
import copy
import csv
import json
import math
from pathlib import Path
import sqlite3
import sys
from unittest.mock import patch

from audit_retained_porter_https import require, same
from inventory_support_carriers import sha, batches
from inventory_visible_source_redundancy import records
import audit_https_funnel as transport
import probe_fine_direct_packet as original_probe
import endpoint_fine_completion as completion


def csv_batches(path):
    def physical_records():
        pending = b''
        with path.open('rb') as stream:
            while block := stream.read(4 * 1024 * 1024):
                complete = (pending + block).split(b'\n')
                pending = complete.pop()
                yield from ((record + b'\n').decode() for record in complete)
            if pending:
                yield pending.decode()
    yield from csv.DictReader(physical_records())


def pack(trace, modules, direct):
    original, digest, manifest, catalog, _ = original_probe.pack(trace, modules, False)
    if not direct:
        return original, digest, manifest, catalog, {}
    search, cascade, fine, overflow, packet, schemas, answer = modules
    pool = {row['id']: row for row in trace['ranked']}
    rows = [pool[mid] for mid in trace['cascade']['fine']['selected_ids']]
    fine_pool = {row['id']: row for row in rows}
    groups = list(trace['cascade']['fine']['selection'].get('adaptive', {}).get('path_groups', []))
    for row in rows:
        group, missing = fine.closure(row, fine_pool)
        require(not missing and group <= fine_pool.keys(), 'Complete fine dependency closure')
        if len(group) > 1:
            groups.append(list(group))
    groups = overflow.merge_groups(groups, rows)
    plan = dict(trace['plan'], _anchor=trace['anchor_time'])
    eligible = {row['id'] for row in search._foresight_filter(rows, plan)
                if row.get('view_status') != 'stale'}
    broken = {mid for group in groups if not set(group) <= eligible for mid in group}
    rows = [row for row in rows if row['id'] in eligible - broken]
    groups = [group for group in groups if not set(group) & broken]
    result, digest, details = completion.complete(original, rows, groups, packet, schemas, answer,
        top_k=trace['top_k'], byte_budget=manifest['token_budget'])
    new_manifest = dict(manifest, included_ids=[item['id'] for item in result],
                        token_upper_bound=details['total_bytes'], endpoint_fine_completion=details)
    new_catalog = completion.visible_cards(result, answer)
    same(packet.digest(result), digest, 'Candidate packet digest')
    require(details['total_bytes'] <= manifest['token_budget'] and len(result) <= trace['top_k'],
            'Original byte and item capacity')
    details.update(original_fine_order=trace['cascade']['fine']['selected_ids'],
                   eligible_fine_ids=[row['id'] for row in rows], original_packet_hash=trace['packet_hash'])
    return result, digest, new_manifest, new_catalog, details


def main(args):
    run = args.run.resolve()
    root = run.parent
    plan = json.loads((root / 'plan.json').read_bytes())
    audit = json.loads((root / 'retained-selection-audit.json').read_bytes())
    same(audit['status'], 'passed', 'Original complete selection audit')
    same(sha(root / 'plan.json'), audit['plan_sha256'], 'Frozen plan')
    for name, digest in plan['source_files'].items():
        same(sha(root / 'source' / name), digest, 'Frozen source')
    paths = [run / name for name in ('server-search-debug.jsonl', 'funnel/stage-transport.json',
        'isolated-manifest.json', 'completed/data/results/personamem-v2-32k.jsonl')]
    for path in paths:
        same(sha(path), audit['input_sha256'][str(path)], 'Audited input')
    manifest = json.loads(paths[2].read_bytes())
    require(manifest['client_exit_code'] == manifest['audit_exit_code'] == 0
            and manifest['server_exit_code'] in (0, -15), 'Terminal source run')
    prepared = Path(manifest['data'])
    same(sha(prepared), plan['data_sha256'], 'Prepared data')
    cohort = 'regression' if root.name.endswith('-regression') else 'validation'
    earlier = Path(__file__).resolve().parents[1] / f'runs/ce-inferred-primary-20261026/{cohort}/plan.json'
    same(sha(run / 'memory.db'), json.loads(earlier.read_bytes())['database_sha256'], 'Audited database')
    sys.path.insert(0, str(root / 'source/memory_system'))
    from app import search_pipeline, cascade_rerank, fine_evidence, listwise_overflow, evidence_packet, schemas, answer_choice, config
    modules = search_pipeline, cascade_rerank, fine_evidence, listwise_overflow, evidence_packet, schemas, answer_choice
    results = {r['search_id']: {k: r[k] for k in ('qa_id', 'conversation_id', 'user_id', 'packet_hash')}
               for r in records(paths[3])}
    same(len(results), plan['questions'], 'Complete cohort')
    fixed, seen = [], set()
    with patch('socket.socket.connect', side_effect=AssertionError('Offline direct packet diagnostic')):
        for trace in records(paths[0]):
            sid = trace['search_id']
            require(sid in results and sid not in seen, 'Unique trace')
            seen.add(sid)
            result = results[sid]
            same((trace['packet_hash'], trace['user_id']), (result['packet_hash'], result['user_id']), 'Trace identity')
            settings = trace['versions']['settings']
            require(not settings['PACKET_QUESTION_CONTEXT_CE'], 'No unreplayed post-packet provider stage')
            with patch.multiple(config, **settings):
                original, digest, original_manifest, old_catalog, _ = pack(trace, modules, False)
                same(original, trace['returned'], 'Exact original endpoint packet')
                same(digest, trace['packet_hash'], 'Original packet hash')
                for key, value in original_manifest.items():
                    same(value, trace['coverage_manifest'][key], 'Original manifest ' + key)
                candidate, new_digest, new_manifest, new_catalog, details = pack(trace, modules, True)
            fixed.append(dict(**result, query=trace['query'], options=trace['options'],
                original_packet=original, candidate_packet=candidate,
                original_catalog=old_catalog, candidate_catalog=new_catalog,
                candidate_packet_hash=new_digest, candidate_manifest=new_manifest, proposal=details))
            print(json.dumps(dict(qa_id=result['qa_id'], original_exact=True,
                candidate_items=len(candidate), packet_changed=digest != new_digest)), flush=True)
    same(seen, set(results), 'All original packets replayed')
    # References cannot influence the already-fixed packets above.
    conversations = {r['conversation_id']: r for r in records(prepared)}
    wanted = {int(r['qa_id']) for r in fixed}
    csv.field_size_limit(32 * 1024 * 1024)
    gold = {}
    for index, row in enumerate(csv_batches(args.gold_csv)):
        if index in wanted:
            gold[str(index)] = row
        if index >= max(wanted):
            break
    raw = {}
    with sqlite3.connect((run / 'memory.db').as_uri() + '?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        for source in batches(db.execute('SELECT * FROM source_messages')):
            source = dict(source)
            raw.setdefault(source['user_id'], {})[transport.identity(source)] = dict(source)
    stages = {(r['conversation_id'], r['qa_id']): r for r in json.loads(paths[1].read_bytes())['rows']}
    for row in fixed:
        conv = conversations[row['conversation_id']]
        qa = next(q for q in conv['qa'] if str(q['id']) == row['qa_id'])
        metadata = gold[row['qa_id']]
        same((qa['question'], qa['options']), (row['query'], row['options']), 'Prepared question')
        same(transport.norm(ast.literal_eval(metadata['user_query'])['content']), transport.norm(qa['question']), 'Gold query')
        same(metadata['correct_answer'], qa['answer'], 'Gold answer identity')
        require(conv['source_history'] in (metadata['chat_history_32k_link'], metadata['chat_history_128k_link']), 'Gold history')
        try:
            snippet = json.loads(metadata['related_conversation_snippet'])
        except ValueError:
            snippet = ast.literal_eval(metadata['related_conversation_snippet'])
        turns = [transport.norm(s['content']) for s in snippet if s['role'] == 'user']
        row['coverage'] = {}
        row['verified_source_instances'] = {}
        for arm in ('original', 'candidate'):
            cards = [dict(s, id=f'{i}:{j}', text=s.get('content', ''))
                     for i, item in enumerate(row[arm + '_packet']) for j, s in enumerate(item.get('sources', []))]
            for source in cards:
                original = raw[row['user_id']][transport.identity(source)]
                same(source.get('role'), original['role'], 'Source role')
                same(source.get('timestamp'), original['timestamp'], 'Source timestamp')
                require(source.get('content', '') in original['content'], 'Verbatim source containment')
            row['verified_source_instances'][arm] = len(cards)
            for view, visible in (('packet', cards), ('catalog', row[arm + '_catalog'])):
                row['coverage'][arm + '_' + view] = [c['full'] for c in transport.turn_coverage(turns, visible, raw[row['user_id']])]
            row[arm + '_catalog_bytes'] = sum(len(c['text'].encode()) for c in row[arm + '_catalog'])
        stage = stages[row['conversation_id'], row['qa_id']]
        for view in ('packet', 'catalog'):
            same(row['coverage']['original_' + view], [c['full'] for c in stage['visible'][view]['turn_coverage']],
                 'Original visible ' + view)
    summary = dict(questions=len(fixed), original_packets_and_manifests_exact=len(fixed), variants={})
    for arm in ('original', 'candidate'):
        summary['variants'][arm] = dict(catalog_bytes=sum(r[arm + '_catalog_bytes'] for r in fixed),
            stages={view: dict(reference_questions=sum(any(r['coverage'][arm + '_' + view]) for r in fixed),
                reference_turns=sum(sum(r['coverage'][arm + '_' + view]) for r in fixed)) for view in ('packet', 'catalog')})
    for view in ('packet', 'catalog'):
        summary[view + '_gains'], summary[view + '_losses'] = [], []
        for row in fixed:
            for index, (a, b) in enumerate(zip(row['coverage']['original_' + view], row['coverage']['candidate_' + view], strict=True)):
                if b and not a:
                    summary[view + '_gains'].append(dict(qa_id=row['qa_id'], turn=index))
                if a and not b:
                    summary[view + '_losses'].append(dict(qa_id=row['qa_id'], turn=index))
    inputs = [Path(__file__), Path(original_probe.__file__), Path(completion.__file__), args.gold_csv, prepared, run / 'memory.db', earlier,
              root / 'plan.json', root / 'retained-selection-audit.json', *paths]
    with args.output.open('x') as stream:
        json.dump(dict(status='complete_offline_packet_ablation', summary=summary, rows=fixed,
            provider_calls=0, production_changed=False, limitation=__doc__,
            input_sha256={str(p.resolve()): sha(p) for p in inputs}), stream, indent=2)
        stream.write('\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gold-csv', type=Path, default=Path(__file__).resolve().parents[1] / 'data/raw/personamem-v2.csv')
    main(parser.parse_args())
