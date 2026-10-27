"""Reconstruct real endpoint completion and attribute its source transport.

Replay the baseline packet from each request's actual upstream decisions, then
the final supplement. Do not infer answer effects from this counterfactual.
"""
import argparse
import ast
import copy
import csv
import json
from pathlib import Path
import sqlite3
import sys
from unittest.mock import patch

from batched_records import csv_records, file_digest, json_records
from audit_retained_porter_https import require, same
from inventory_support_carriers import batches
from probe_fine_direct_packet import pack
import audit_https_funnel as transport


def audit(run):
    run = run.resolve()
    root = run.parent
    plan = json.loads((root / 'plan.json').read_bytes())
    manifest = json.loads((run / 'isolated-manifest.json').read_bytes())
    same(plan['toggle'], 'PACKET_ENDPOINT_FINE_COMPLETION', 'Expected experiment')
    require(manifest.get('server_exit_code') in (0, -15) and manifest.get('audit_exit_code') == 0,
            'Terminal server and stage audit')
    for name, digest in plan['source_files'].items():
        same(file_digest(root / 'source' / name), digest, 'Frozen code')
    sys.path.insert(0, str(root / 'source/memory_system'))
    from app import answer_choice, cascade_rerank, config, endpoint_fine_completion, evidence_packet, fine_evidence, listwise_overflow, schemas, search_pipeline
    modules = search_pipeline, cascade_rerank, fine_evidence, listwise_overflow, evidence_packet, schemas, answer_choice
    results = {r['search_id']: r for r in json_records(run / 'completed/data/results/personamem-v2-32k.jsonl')}
    same(len(results), plan['questions'], 'Full question count')
    raw = {}
    with sqlite3.connect((run / 'memory.db').as_uri() + '?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        for value in batches(db.execute('SELECT * FROM source_messages')):
            value = dict(value)
            raw.setdefault(value['user_id'], {})[transport.identity(value)] = value
    fixed, seen = [], set()
    with patch('socket.socket.connect', side_effect=AssertionError('Offline HTTPS attribution')):
        for trace in json_records(run / 'server-search-debug.jsonl'):
            sid = trace['search_id']
            require(sid in results and sid not in seen, 'Unique request trace')
            seen.add(sid)
            result = results[sid]
            same(result['packet_hash'], trace['packet_hash'], 'Result packet identity')
            with patch.multiple(config, **trace['versions']['settings']):
                require(not config.PACKET_QUESTION_CONTEXT_CE, 'No unreplayed post-packet CE')
                original, digest, old_manifest, old_catalog, _ = pack(trace, modules, False)
                req = schemas.SearchRequest(**{key: trace[key] for key in (
                    'user_id', 'query', 'options', 'top_k', 'reference_time', 'evidence_token_budget')})
                local_plan = dict(copy.deepcopy(trace['plan']), _cascade=copy.deepcopy(trace['cascade']),
                                  _anchor=trace['anchor_time'])
                recorded = trace['coverage_manifest'].get('endpoint_fine_completion', {})
                if recorded.get('status') == 'budget_skipped':
                    require(config.PACKET_ENDPOINT_FINE_COMPLETION, 'Skip belongs to enabled arm')
                    same(recorded['base_packet_hash'], digest, 'Skipped baseline hash')
                    same(recorded['appended_ids'], [], 'Skipped additions')
                    actual, new_digest, updated = original, digest, old_manifest
                else:
                    actual, new_digest, updated = search_pipeline._append_fine_completion(
                        req, local_plan, trace['ranked'], original, digest, old_manifest)
                    same(updated.get('endpoint_fine_completion', {}), recorded, 'Exact completion decisions')
                actual = [schemas.SearchItem.model_validate(item).model_dump() for item in actual]
                same(actual, trace['returned'], 'Exact actual HTTPS packet')
                same(new_digest, trace['packet_hash'], 'Exact actual HTTPS digest')
                same([endpoint_fine_completion.without_digest(item) for item in actual[:len(original)]],
                     [endpoint_fine_completion.without_digest(item) for item in original], 'Original prefix retained')
                sources, constraints = answer_choice.build_catalog(actual)
                catalog = list(sources.values()) + list(constraints.values())
                answer_choice.add_current_question(sources, constraints, trace['query'], {})
                if 'answer_source_catalog' in result:
                    same(list(sources.values()), result['answer_source_catalog'], 'Actual reader sources')
                    same(list(constraints.values()), result['answer_constraint_catalog'], 'Actual reader constraints')
                elif not result.get('error_stage'):
                    raise AssertionError('Successful answer lacks reader catalogs')
            source_instances = 0
            for item in actual:
                for source in item.get('sources', []):
                    if not source.get('content'):
                        continue
                    origin = raw[trace['user_id']][transport.identity(source)]
                    same(source.get('role'), origin['role'], 'Raw source role')
                    same(source.get('timestamp'), origin['timestamp'], 'Raw source timestamp')
                    require(source['content'] in origin['content'], 'Verbatim raw source')
                    source_instances += 1
            fixed.append(dict(qa_id=str(result['qa_id']), user_id=trace['user_id'],
                original_catalog=old_catalog, actual_catalog=catalog, source_instances=source_instances,
                appended_ids=recorded.get('appended_ids', []), status=recorded.get('status', 'disabled'),
                original_bytes=sum(len(card['text'].encode()) for card in old_catalog),
                actual_bytes=sum(len(card['text'].encode()) for card in catalog)))
    same(seen, set(results), 'All actual requests reproduced')
    # Only after every baseline/candidate packet is fixed, load reference probes.
    gold_path = Path(__file__).resolve().parents[1] / 'data/raw/personamem-v2.csv'
    wanted = {int(row['qa_id']) for row in fixed}
    csv.field_size_limit(32 * 1024 * 1024)
    gold = {}
    for index, row in enumerate(csv_records(gold_path)):
        if index in wanted:
            gold[str(index)] = row
        if index >= max(wanted):
            break
    details = []
    for row in fixed:
        try:
            snippets = json.loads(gold[row['qa_id']]['related_conversation_snippet'])
        except ValueError:
            snippets = ast.literal_eval(gold[row['qa_id']]['related_conversation_snippet'])
        turns = [transport.norm(value['content']) for value in snippets if value['role'] == 'user']
        before = [value['full'] for value in transport.turn_coverage(turns, row.pop('original_catalog'), raw[row['user_id']])]
        after = [value['full'] for value in transport.turn_coverage(turns, row.pop('actual_catalog'), raw[row['user_id']])]
        details.append(dict(row, original_coverage=before, actual_coverage=after,
            gained_turns=[i for i,(a,b) in enumerate(zip(before,after,strict=True)) if b and not a],
            lost_turns=[i for i,(a,b) in enumerate(zip(before,after,strict=True)) if a and not b]))
    summary = dict(questions=len(details), actual_packets_reproduced=len(details),
        original_prefixes_preserved=len(details), appended_units=sum(len(row['appended_ids']) for row in details),
        source_instances=sum(row['source_instances'] for row in details),
        gained_turns=sum(len(row['gained_turns']) for row in details),
        lost_turns=sum(len(row['lost_turns']) for row in details),
        original_catalog_bytes=sum(row['original_bytes'] for row in details),
        actual_catalog_bytes=sum(row['actual_bytes'] for row in details))
    paths = [Path(__file__), root/'plan.json', root/'fixture.json', gold_path,
             run/'server-search-debug.jsonl', run/'completed/data/results/personamem-v2-32k.jsonl']
    report = dict(status='passed', summary=summary, rows=details, limitation=__doc__,
                  input_sha256={str(path): file_digest(path) for path in paths})
    with (run/'endpoint-completion-audit.json').open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    audit(parser.parse_args().run)
