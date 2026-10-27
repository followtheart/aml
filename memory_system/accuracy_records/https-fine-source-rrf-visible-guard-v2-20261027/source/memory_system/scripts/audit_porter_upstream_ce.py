"""Audit complete Porter CE diagnostics against upstream texts and provider ledgers."""
import argparse
import copy
import json
from pathlib import Path
import re
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import config, cross_encoder
from evaluate_porter_upstream_ce import identity, score_outputs, select, selection_code, sha


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def differences(left, right, path=''):
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            yield dict(path=path, reason='keys', left=sorted(left), right=sorted(right))
        for key in sorted(left.keys() & right.keys()):
            yield from differences(left[key], right[key], path + '/' + key)
    elif isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            yield dict(path=path, reason='length', left=len(left), right=len(right))
        elif left != right and all(isinstance(v, str) for v in left + right):
            yield dict(path=path, reason='string_list', left=left, right=right)
        else:
            for i, (a, b) in enumerate(zip(left, right)):
                yield from differences(a, b, path + '/' + str(i))
    elif left != right:
        yield dict(path=path, reason='value', left=left, right=right)


def comparable(selection):
    """The reservation trace lists a membership map, not ranked candidates.

    evidence_selection.select builds its map by iterating reserve/fused sets;
    selection and main admission consume the membership, not map insertion order.
    Preserve every row/value, including duplicates, and all other list ordering.
    """
    value = copy.deepcopy(selection)
    value['fine_selection']['reservations'].sort(key=lambda r: json.dumps(r, sort_keys=True))
    return value


def main(args):
    for directory in (args.upstream, args.output):
        if directory.with_suffix('.exit-code').read_text().strip() != '0':
            raise ValueError('Requires a successful terminal evaluation')
    plan = json.loads((args.output / 'plan.json').read_text())
    result = json.loads((args.output / 'result.json').read_text())
    upstream_result = json.loads((args.upstream / 'result.json').read_text())
    if (sha(args.output / 'plan.json') != result['plan_sha256'] or
            sha(args.output / 'selections.jsonl') != result['selections_sha256'] or
            sha(args.upstream / 'selections.jsonl') != plan['upstream_sha256'] or
            plan['upstream_sha256'] != upstream_result['selection_sha256'] or
            sha(args.run / 'server-search-debug.jsonl') != plan['trace_sha256'] or
            sha(args.run / 'memory.db') != plan['database_sha256']):
        raise ValueError('Artifact hashes differ')
    for hashes in (plan['source_hashes'], upstream_result['source_hashes']):
        if any(sha(Path(p)) != v for p, v in hashes.items()):
            raise ValueError('Implementation hash differs')
    traces = {t['search_id']: t for t in records(args.run / 'server-search-debug.jsonl')}
    jobs = {j['qa_id']: j for j in plan['jobs']}
    outputs = {j['qa_id']: j for j in records(args.output / 'selections.jsonl')}
    events = records(args.output / 'live.jsonl')
    if len(jobs) != args.expected_count or set(jobs) != set(outputs):
        raise ValueError('Incomplete or inconsistent cohort')
    by_qa = {}
    for e in events:
        if 'error' in e or e['calls'] != 1:
            raise ValueError('Unexpected failed or retried CE batch')
        by_qa.setdefault(e['qa_id'], []).append(e)
    code, fragment_sha = selection_code()
    if fragment_sha != plan['selection_ast_sha256']:
        raise ValueError('Production selection block differs')
    reconstructed = 0
    reservation_order_differences = []
    assignments = dict(historical_successful=0, historical_missing=0, newly_measured=0)
    with patch('socket.socket.connect', side_effect=AssertionError('Offline CE audit')):
        for line in (args.upstream / 'selections.jsonl').open():
            j = json.loads(line)
            qa = j['qa_id']
            p = jobs[qa]
            t = traces[j['search_id']]
            if p['query'] != t['cascade']['cross_encoder']['query']:
                raise ValueError('Query differs')
            old_ids = set(t['cascade']['coarse']['selected_ids'])
            old = {identity(c): c.get('_ce_score') for c in t['ranked'] if c['id'] in old_ids}
            novel = {}
            for v in j['variants'].values():
                for c in v['coarse']:
                    key = identity(c)
                    if key not in old:
                        novel.setdefault(key, dict(key=key, candidate_id=c['id'], text=c['_rank_text']))
            if list(novel.values()) != p['novel']:
                raise ValueError('CE requests are not the actual new upstream text pairs')
            scores = dict(old)
            batch_events = by_qa.get(qa, [])
            if len(batch_events) != len(p['batches']):
                raise ValueError('Batch count differs')
            measured = set()
            with patch.multiple(config, **t['versions']['settings']):
                for i, (keys, event) in enumerate(zip(p['batches'], batch_events, strict=True)):
                    if event['batch'] != i or event['keys'] != keys or measured & set(keys):
                        raise ValueError('CE batch identities differ or were repeated')
                    if cross_encoder.request_size(p['query'], [novel[key]['text'] for key in keys]) != event['request_bound']:
                        raise ValueError('CE request budget differs')
                    measured.update(keys)
                    scores.update(zip(keys, event['scores'], strict=True))
                if measured != set(novel):
                    raise ValueError('Missing novel pair scores')
                for mode, variant in j['variants'].items():
                    rebuilt = select(t, variant, scores, code)
                    saved = outputs[qa]['variants'][mode]
                    if rebuilt != saved and comparable(rebuilt) == comparable(saved):
                        reservation_order_differences.append(dict(qa_id=qa, mode=mode))
                    if comparable(rebuilt) != comparable(saved):
                        mismatch = dict(qa_id=qa, mode=mode, differences=list(differences(rebuilt, outputs[qa]['variants'][mode])))
                        with (args.output / 'audit-mismatch.json').open('x') as stream:
                            json.dump(mismatch, stream, indent=2)
                        raise ValueError('Saved fine/main output is not the production selection: ' + qa)
                    for c in variant['coarse']:
                        key = identity(c)
                        if key not in old:
                            assignments['newly_measured'] += 1
                        elif old[key] is None:
                            assignments['historical_missing'] += 1
                        else:
                            assignments['historical_successful'] += 1
            reconstructed += 1
        rescored = score_outputs(args)
    if rescored['rows'] != result['rows'] or rescored['summary'] != result['summary']:
        raise ValueError('Reference coverage does not reproduce')
    ledger = []
    for line in (args.output / 'provider.log').read_text().splitlines():
        if 'provider_call ' not in line:
            continue
        fields = dict(re.findall(r'(\w+)=([^\s]+)', line))
        if fields['kind'] != 'rerank' or fields['model'] != 'qwen3-rerank' or fields['status'] != 'ok':
            raise ValueError('Unexpected provider record')
        ledger.append(fields)
    calls = sum(e['calls'] for e in events)
    tokens = sum(e['tokens'] for e in events)
    if calls != result['calls'] or calls != len(ledger) or calls != plan['expected_calls']:
        raise ValueError('Call ledger differs')
    if tokens != result['tokens'] or tokens != sum(int(e['total_tokens']) for e in ledger):
        raise ValueError('Token ledger differs')
    if reconstructed != args.expected_count:
        raise ValueError('Incomplete output reconstruction')
    final = dict(status='passed', questions=reconstructed, reconstructed_selections=2 * reconstructed,
        score_assignments=assignments, novel_text_pairs=plan['new_pairs'], calls=calls, tokens=tokens,
        historical_partial=plan['historical_partial'], result_sha256=sha(args.output / 'result.json'),
        reservation_order_differences=reservation_order_differences,
        selections_sha256=sha(args.output / 'selections.jsonl'), provider_log_sha256=sha(args.output / 'provider.log'),
        script_sha256=sha(Path(__file__)), provider_calls_during_audit=0,
        limitation='Checks saved inputs/outputs and recorded provider usage, not external billing or live HTTPS scheduling.')
    with (args.output / 'audit.json').open('x') as stream:
        json.dump(final, stream, indent=2)
        stream.write('\n')
    print(json.dumps(final), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--upstream', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--expected-count', type=int, required=True)
    p.add_argument('--gold-csv', type=Path, default=Path('data/raw/personamem-v2.csv'))
    main(p.parse_args())
