"""Offline audit of full-option CE scores, admission, coverage and accounting.

Replays recorded scores only; network access is blocked. Reference coverage is
recomputed from the source database, independently of the probe's debug-log
source lookup. No HTTPS or answer-quality claim is made.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import config
import audit_https_funnel as funnel
from evaluate_recall_context import reference_turns
from probe_full_option_ce import prepare, batches_for, sha, base


def audit(directory, gold_csv, *, query_builder=None):
    plan = json.loads((directory / 'plan.json').read_text())
    report = json.loads((directory / 'result.json').read_text())
    live = base.records(directory / 'responses.jsonl')
    assert (directory / 'exit-code').read_text().strip() == '0'
    run = Path(plan['run'])
    assert sha(run / 'server-search-debug.jsonl') == plan['trace_sha256']
    assert sha(run / 'completed/data/results/personamem-v2-32k.jsonl') == plan['result_sha256']
    assert sha(Path(plan['prepared'])) == plan['data_sha256']
    for path, expected in plan['script_hashes'].items():
        assert sha(Path(path)) == expected, path
    root = Path(__file__).resolve().parents[2]
    for path, expected in plan['versions']['file_hashes'].items():
        assert sha(root / path) == expected, path
    jobs, manifest, conversations = prepare(run, query_builder=query_builder)
    assert len(jobs) == len(plan['jobs']) == len(report['rows'])
    expected_live = [(j['conversation_id'], j['qa_id'], arm)
                     for j in jobs if j['changed'] for arm in ('control', 'candidate')]
    assert [(v['conversation_id'], v['qa_id'], v['arm']) for v in live] == expected_live
    by_key = {(v['qa_id'], v['arm']): v for v in live}
    raw = {}
    database = run / 'memory.db'
    with sqlite3.connect(database.resolve().as_uri() + '?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        for source in db.execute('SELECT * FROM source_messages'):
            source = dict(source)
            raw.setdefault(source['user_id'], {})[funnel.identity(source)] = source
    csv.field_size_limit(32 * 1024 * 1024)
    with gold_csv.open() as stream:
        gold = list(csv.DictReader(stream))
    convs = {c['conversation_id']: c for c in conversations}
    historical_matches = dict(fine=0, main=0)
    token_totals = dict(control=0, candidate=0)
    checked_sources = set()
    recomputed = []
    expected_calls = 0
    for job, recorded_plan, row in zip(jobs, plan['jobs'], report['rows']):
        assert job['qa_id'] == recorded_plan['qa_id'] == row['qa_id']
        assert job['conversation_id'] == recorded_plan['conversation_id'] == row['conversation_id']
        assert job['changed'] == recorded_plan['changed'] == row['changed']
        assert job['t']['search_id'] == recorded_plan['search_id']
        assert job['queries'] == recorded_plan['queries']
        assert [c['id'] for c in job['coarse']] == recorded_plan['candidate_ids']
        assert [hashlib.sha256(d.encode()).hexdigest() for d in job['docs']] == recorded_plan['document_hashes']
        for candidate in job['coarse']:
            for source in candidate['_packet_item'].get('sources', []):
                key = funnel.identity(source)
                original = raw[job['t']['user_id']][key]
                assert source['role'] == original['role']
                assert funnel.norm(source.get('content', '')) in funnel.norm(original['content'])
                checked_sources.add((job['t']['user_id'], *key))
        scores = dict(archived=[c['_ce_score'] for c in job['coarse']])
        costs = {}
        for arm in ('control', 'candidate'):
            if not job['changed']:
                scores[arm] = scores['archived']
                costs[arm] = 0
                continue
            event = by_key[job['qa_id'], arm]
            assert event['query'] == job['queries'][arm]
            with patch.multiple(config, **job['t']['versions']['settings']):
                count = batches_for(event['query'], job['docs'])
            assert event['calls'] == event['batches'] == count > 0
            assert type(event['tokens']) is int and event['tokens'] > 0
            expected_calls += count
            token_totals[arm] += event['tokens']
            costs[arm] = event['tokens']
            scores[arm] = event['scores']
        delta = max(0, costs['candidate'] - costs['control'])
        assert row['additional_token_cost'] == delta
        selections = {name: base.replay(job, values) for name, values in scores.items()}
        adjusted = copy.deepcopy(job)
        headroom = adjusted['t']['cascade']['admission_budget']['token_headroom']
        if headroom is not None:
            adjusted['t']['cascade']['admission_budget']['token_headroom'] = max(0, headroom - delta)
        selections['candidate_cost_adjusted'] = base.replay(adjusted, scores['candidate'])
        conv = convs[job['conversation_id']]
        question = next(q for q in conv['qa'] if q['id'] == job['qa_id'])
        turns = reference_turns(question, conv['source_history'], gold[int(job['qa_id'])])
        variants = {}
        for name, selection in selections.items():
            result = dict(main_bytes=selection['main_bytes'])
            for stage in ('fine', 'main'):
                cards = [dict(s, text=s.get('content', ''), id=f'{i}:{k}')
                         for i, c in enumerate(selection[stage])
                         for k, s in enumerate(c['_packet_item'].get('sources', []))]
                values = funnel.turn_coverage(turns, cards, raw[job['t']['user_id']])
                result[stage] = dict(ids=[c['id'] for c in selection[stage]],
                                     user=[v['full'] for v in values])
            assert result == row['variants'][name], (job['qa_id'], name)
            variants[name] = result
        for stage in historical_matches:
            historical_matches[stage] += variants['archived'][stage]['ids'] == variants['control'][stage]['ids']
        recomputed.append(dict(qa_id=job['qa_id'], variants=variants))
    summary = report['summary']
    for name in ('archived', 'control', 'candidate', 'candidate_cost_adjusted'):
        for stage in ('fine', 'main'):
            expected = dict(
                reference_questions=sum(any(r['variants'][name][stage]['user']) for r in recomputed),
                user_turns=sum(sum(r['variants'][name][stage]['user']) for r in recomputed),
                gained=[r['qa_id'] for r in recomputed if any(r['variants'][name][stage]['user']) and not any(r['variants']['control'][stage]['user'])],
                lost=[r['qa_id'] for r in recomputed if any(r['variants']['control'][stage]['user']) and not any(r['variants'][name][stage]['user'])],
                fewer_turns=[r['qa_id'] for r in recomputed if sum(r['variants'][name][stage]['user']) < sum(r['variants']['control'][stage]['user'])])
            assert summary['variants'][name][stage] == expected
    preflight = dict(questions=len(jobs), changed=sum(j['changed'] for j in jobs),
        unchanged_reuse=sum(not j['changed'] for j in jobs), expected_calls=expected_calls,
        reconstructed_fine_and_main=len(jobs), max_auxiliary_bytes=max(
            len(j['queries']['candidate'][len(j['t']['query']):].encode()) for j in jobs))
    assert plan['preflight'] == preflight
    assert all(summary[key] == value for key, value in preflight.items())
    assert summary['calls'] == expected_calls == sum(v['calls'] for v in live)
    assert summary['tokens'] == sum(token_totals.values()) == sum(v['tokens'] for v in live)
    return dict(status='passed', questions=len(jobs), paired_questions=preflight['changed'],
        calls=expected_calls, tokens=token_totals, historical_control_matches=historical_matches,
        verified_source_messages=len(checked_sources), variants=summary['variants'],
        inputs={str(p): sha(p) for p in (directory / 'plan.json', directory / 'result.json',
            directory / 'responses.jsonl', gold_csv, database, Path(__file__))}, limitation=__doc__)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gold-csv', type=Path, default=Path(__file__).resolve().parents[1] / 'data/raw/personamem-v2.csv')
    args = parser.parse_args()
    with patch('socket.socket.connect', side_effect=AssertionError('Offline audit')):
        result = audit(args.probe, args.gold_csv)
    with args.output.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(result, ensure_ascii=False))
