"""Paired CE diagnostic changing only verification wording to context retrieval.

All original question/premise/subquestion bytes and candidate documents stay
fixed. This is a sequential hypothesis on previously examined cohorts, not
independent validation, HTTPS behavior, or answer accuracy. Read fixed-size
batches and discard unused debug fields to fit the 2-core/4-GB host.
"""
import argparse
import asyncio
import copy
from contextlib import ExitStack, contextmanager
import hashlib
import json
import logging
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import config, rerank_query
import probe_full_option_ce as probe


OLD_SENTENCE = 'Verify these alternative premises against the evidence; they are not established facts. '
NEW_SENTENCE = 'Find context relevant to these alternative premises; they are not established facts. '
QUERY_POLICY = 'retrieval_guidance_only_preserve_original_clauses'
READ_BATCH_BYTES = 4 * 1024 * 1024


def replace_guidance(question, original):
    if original == question:
        return original
    header = '\n\n' + rerank_query._GUIDANCE
    if not original.startswith(question + header + '\n'):
        raise ValueError('Unexpected original auxiliary header')
    if not rerank_query._GUIDANCE.startswith(OLD_SENTENCE):
        raise ValueError('Verification guidance changed')
    # Replace only at the known auxiliary boundary; never rebuild the suffix,
    # which could admit extra clauses under the now slightly shorter header.
    tail = original[len(question + '\n\n' + OLD_SENTENCE):]
    candidate = question + '\n\n' + NEW_SENTENCE + tail
    if len(candidate.encode()) > len(original.encode()):
        raise ValueError('Expanded query budget')
    return candidate


def queries_for(t):
    req = SimpleNamespace(query=t['query'], options=t['options'])
    plan = copy.deepcopy(t['plan'])
    plan['_query_specs'] = copy.deepcopy(t['query_specs'])
    with patch.object(config, 'CE_FULL_OPTION_TEXT', False):
        original = rerank_query.build(req, plan)
    if original != t['cascade']['cross_encoder']['query']:
        raise ValueError('Historical CE query mismatch')
    candidate = replace_guidance(req.query, original)
    if len(candidate[len(req.query):].encode()) > rerank_query.MAX_AUXILIARY_BYTES:
        raise ValueError('Auxiliary byte budget exceeded')
    return dict(control=original, candidate=candidate)


def compact_trace(t):
    """Keep whole original coarse candidates and all selection/query inputs."""
    c = t['cascade']
    ids = set(c['coarse']['selected_ids'])
    value = {k: t[k] for k in ('search_id', 'user_id', 'packet_hash', 'query',
        'options', 'plan', 'query_specs', 'versions', 'fusion_edges', 'fusion_triples')}
    value['ranked'] = [item for item in t['ranked'] if item['id'] in ids]
    value['coverage_manifest'] = {'coverage': {
        'requirements': t['coverage_manifest']['coverage']['requirements']}}
    value['cascade'] = {
        'coarse': {k: c['coarse'][k] for k in ('selected_ids', 'baseline_ids') if k in c['coarse']},
        'cross_encoder': {k: c['cross_encoder'][k] for k in ('status', 'model', 'query')},
        'fine': {'selected_ids': c['fine']['selected_ids']},
        'listwise': {k: c['listwise'][k] for k in ('candidate_ids', 'input_bytes')},
        'admission_budget': c['admission_budget'],
    }
    return value


def json_batches(path, batch_bytes=READ_BATCH_BYTES):
    """Read binary blocks; carry an unfinished record to the next batch.

    Decode UTF-8 only after a complete JSON record is available. A single
    oversized record is kept whole, so memory is bounded by batch size plus
    the largest record, not batch size alone. No readline/file iteration.
    """
    if type(batch_bytes) is not int or batch_bytes <= 0:
        raise ValueError('Batch size must be a positive integer')
    pending = b''
    with path.open('rb') as stream:
        while block := stream.read(batch_bytes):
            complete = (pending + block).split(b'\n')
            pending = complete.pop()
            batch = [json.loads(record) for record in complete if record.strip()]
            if batch:
                yield batch
        if pending.strip():
            yield [json.loads(pending)]


def trace_records(path):
    for batch in json_batches(path):
        for trace in batch:
            yield compact_trace(trace)


def records(path):
    if path.name == 'server-search-debug.jsonl':
        return trace_records(path)
    return [record for batch in json_batches(path) for record in batch]


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


@contextmanager
def bounded_readers():
    import audit_full_option_ce as auditor
    with ExitStack() as stack:
        stack.enter_context(patch.object(probe.base, 'records', records))
        stack.enter_context(patch.object(probe, 'sha', sha))
        stack.enter_context(patch.object(auditor, 'sha', sha))
        yield


def main(args):
    with bounded_readers():
        if args.audit:
            from audit_full_option_ce import audit
            plan = json.loads((args.output / 'plan.json').read_text())
            if plan['query_policy'] != QUERY_POLICY:
                raise ValueError('Wrong experiment policy')
            with patch('socket.socket.connect', side_effect=AssertionError('Offline audit')):
                result = audit(args.output, args.gold_csv, query_builder=queries_for)
            result['limitation'] = __doc__
            result['inputs'][str(Path(__file__).resolve())] = sha(Path(__file__))
            with (args.output / 'audit.json').open('x') as stream:
                json.dump(result, stream, ensure_ascii=False, indent=2)
                stream.write('\n')
            print(json.dumps({k: v for k, v in result.items() if k != 'inputs'}))
        else:
            if not args.run:
                raise ValueError('--run required')
            args.query_builder = queries_for
            args.additional_scripts = [Path(__file__), Path(__file__).with_name('audit_full_option_ce.py')]
            args.query_policy = QUERY_POLICY
            args.limitation = __doc__
            asyncio.run(probe.main(args))


if __name__ == '__main__':
    logger = logging.getLogger('aml.metrics')
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())
    logger.propagate = False
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--audit', action='store_true')
    parser.add_argument('--gold-csv', type=Path,
        default=Path(__file__).resolve().parents[1] / 'data/raw/personamem-v2.csv')
    main(parser.parse_args())
