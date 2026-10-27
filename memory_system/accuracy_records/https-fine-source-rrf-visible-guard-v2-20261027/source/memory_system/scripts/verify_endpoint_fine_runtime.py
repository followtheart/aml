"""Replay current production packet builder and final completion on frozen cases."""
import argparse
import copy
import json
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import answer_choice, cascade_rerank, config, evidence_packet, fine_evidence, listwise_overflow, schemas, search_pipeline
from audit_fine_direct_lifecycle import read_json
from inventory_visible_source_redundancy import records
from inventory_support_carriers import sha
from audit_retained_porter_https import same, require
from probe_fine_direct_packet import pack


def run(args):
    modules = search_pipeline, cascade_rerank, fine_evidence, listwise_overflow, evidence_packet, schemas, answer_choice
    expected = {row['qa_id']: row for row in read_json(args.expected)['rows']}
    identities = {r['search_id']: str(r['qa_id']) for r in records(
        args.source / 'completed/data/results/personamem-v2-32k.jsonl')}
    checked, summaries = set(), []
    with patch('socket.socket.connect', side_effect=AssertionError('Offline runtime parity')):
        for trace in records(args.source / 'server-search-debug.jsonl'):
            qid = identities[trace['search_id']]
            require(qid not in checked, 'Unique question')
            row = expected[qid]
            settings = dict(trace['versions']['settings'], PACKET_ENDPOINT_FINE_COMPLETION=False)
            with patch.multiple(config, **settings):
                original, digest, manifest, _, _ = pack(trace, modules, False)
                same(original, row['original_packet'], 'Current default-off original packet')
                same(digest, trace['packet_hash'], 'Current default-off original hash')
                for key, value in manifest.items():
                    same(value, trace['coverage_manifest'][key], 'Current original manifest ' + key)
                req = schemas.SearchRequest(**{key: trace[key] for key in (
                    'user_id', 'query', 'options', 'top_k', 'reference_time', 'evidence_token_budget')})
                plan = dict(copy.deepcopy(trace['plan']), _cascade=copy.deepcopy(trace['cascade']),
                            _anchor=trace['anchor_time'])
                original_copy = copy.deepcopy(original)
                disabled = search_pipeline._append_fine_completion(req, plan, trace['ranked'], original, digest, manifest)
                same(disabled, (original, digest, manifest), 'Default-off no-op')
                with patch.object(config, 'PACKET_ENDPOINT_FINE_COMPLETION', True):
                    actual, new_digest, updated = search_pipeline._append_fine_completion(
                        req, plan, trace['ranked'], original, digest, manifest)
                same(original, original_copy, 'Immutable input')
                actual = [schemas.SearchItem.model_validate(item).model_dump() for item in actual]
                same(actual, row['candidate_packet'], 'Exact current-runtime candidate packet')
                same(new_digest, row['candidate_packet_hash'], 'Exact current-runtime candidate hash')
                same(updated['token_upper_bound'], row['candidate_manifest']['token_upper_bound'], 'Exact bytes')
                same(updated['evidence_count'] + updated['constraint_count'], len(actual), 'Accurate counts')
                same(updated['included_ids'], [item['id'] for item in actual], 'Actual included IDs')
                same(updated['endpoint_fine_completion']['appended_ids'], row['proposal']['appended_ids'], 'Addition parity')
            checked.add(qid)
            summaries.append(dict(qa_id=qid, original_exact=True, candidate_exact=True,
                                  additions=len(row['proposal']['appended_ids'])))
    same(checked, set(expected), 'Complete cohort')
    inputs = [Path(__file__), Path(search_pipeline.__file__), args.expected,
              Path(__file__).resolve().parents[1] / 'app/endpoint_fine_completion.py',
              args.source / 'server-search-debug.jsonl']
    report = dict(status='passed', questions=len(checked), rows=summaries, provider_calls=0,
                  input_sha256={str(path.resolve()): sha(path) for path in inputs})
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(status='passed', questions=len(checked))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--expected', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
