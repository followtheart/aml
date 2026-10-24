"""Verify current topic CE requests and replay recorded scores without providers."""
import argparse
import hashlib
import json
from pathlib import Path
from probe_current_ce_topics import prepare, records, replay, sha


def audit(output):
    plan = json.loads(output.with_suffix('.plan.json').read_text())
    live = records(output.with_suffix('.live.jsonl'))
    report = json.loads(output.read_text())
    run = Path(plan['run'])
    assert sha(run / 'server-search-debug.jsonl') == plan['trace_sha256']
    assert sha(Path(__file__).with_name('ce_option_topics.py')) == plan['rule_sha256']
    assert sha(Path(__file__).with_name('probe_current_ce_topics.py')) == plan['script_sha256']
    jobs = prepare(run)
    assert len(jobs) == len(plan['jobs']) == len(report['rows'])
    expected_live = [(j['qa_id'], arm) for j in jobs if j['changed'] for arm in ('control', 'candidate')]
    assert [(v['qa_id'], v['arm']) for v in live] == expected_live
    by_key = {(v['qa_id'], v['arm']): v for v in live}
    matches = {stage: 0 for stage in ('fine', 'main')}
    for job, p, row in zip(jobs, plan['jobs'], report['rows']):
        assert job['qa_id'] == p['qa_id'] == row['qa_id']
        assert job['changed'] == p['changed'] == row['changed']
        assert p['search_id'] == job['t']['search_id']
        assert p['queries'] == job['queries']
        assert p['candidate_ids'] == [c['id'] for c in job['coarse']]
        assert p['document_hashes'] == [hashlib.sha256(d.encode()).hexdigest() for d in job['docs']]
        for name in ('archived', 'control', 'candidate'):
            scores = [c['_ce_score'] for c in job['coarse']]
            if job['changed'] and name != 'archived':
                recorded = by_key[(job['qa_id'], name)]
                assert recorded['query'] == p['queries'][name]
                assert recorded['calls'] == recorded['batches'] > 0 and recorded['tokens'] > 0
                scores = recorded['scores']
            selection = replay(job, scores)
            assert row['variants'][name]['main_bytes'] == selection['main_bytes']
            for stage in ('fine', 'main'):
                assert row['variants'][name][stage]['ids'] == [c['id'] for c in selection[stage]]
        for stage in matches:
            matches[stage] += row['variants']['archived'][stage]['ids'] == row['variants']['control'][stage]['ids']
    assert report['summary']['calls'] == sum(v['calls'] for v in live)
    assert report['summary']['tokens'] == sum(v['tokens'] for v in live)
    return dict(status='passed', questions=len(jobs), paired_questions=sum(j['changed'] for j in jobs),
                historical_control_matches=matches, calls=report['summary']['calls'],
                tokens=report['summary']['tokens'],
                limitation='Validates query/document identity, score completeness, exact fine/main admission and recorded accounting. Coverage and answers are not independently validated here; untriggered questions use archived scores.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--probe', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = audit(a.probe)
    with a.output.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps(result))
