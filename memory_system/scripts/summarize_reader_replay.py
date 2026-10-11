"""Validate paired reader-replay identities and summarize only complete pairs."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path


def read_complete(path, partial=False):
    lines = path.read_text().splitlines(keepends=True)
    if lines and not lines[-1].endswith('\n'):
        if not partial:
            raise ValueError('Incomplete final output line')
        lines.pop()
    return [json.loads(line) for line in lines if line.strip()]


def main(args):
    baseline = read_complete(args.baseline)
    expected = {(r['conversation_id'], r['qa_id']): r['packet_hash'] for r in baseline}
    if len(expected) != len(baseline):
        raise ValueError('Duplicate baseline question')
    rows = read_complete(args.replay, args.allow_partial)
    pairs = defaultdict(dict)
    statuses, errors, scope_calls, calls = Counter(), [], Counter(), Counter()
    for r in rows:
        question = (r['conversation_id'], r['qa_id'])
        rep, enabled = r['repetition'], r['enabled']
        if type(enabled) is not bool or not 1 <= rep <= args.repeat:
            raise ValueError('Unexpected arm or repetition')
        if question not in expected or r['packet_hash'] != expected[question]:
            raise ValueError('Replay changed original question/packet identity')
        if r['versions']['settings'][args.setting] is not enabled:
            raise ValueError('Recorded setting disagrees with arm')
        key = (*question, rep)
        if enabled in pairs[key]:
            raise ValueError('Duplicate arm')
        pairs[key][enabled] = r
        arm = 'treatment' if enabled else 'control'
        d = r['diagnostics']
        statuses[(arm, (d.get('answer_premise_scope') or {}).get('status', 'not_requested'))] += 1
        calls[arm] += len(r['live_calls'])
        scope_calls[arm] += sum(c['stage'].startswith('eval.choice_premise_scope') for c in r['live_calls'])
        if d.get('answer_status') == 'error':
            errors.append(dict(qa_id=r['qa_id'], arm=arm, error=d.get('error_type')))
    count = len(expected) * args.repeat * 2
    partial = len(rows) != count
    if partial and not args.allow_partial:
        raise ValueError(f'Expected {count} rows, found {len(rows)}')
    results = []
    for (conversation, qid, rep), arms in pairs.items():
        if set(arms) != {False, True}:
            continue
        a, b = arms[False], arms[True]
        results.append(dict(conversation_id=conversation, qa_id=qid, repetition=rep,
                            control_score=a['score'], treatment_score=b['score'],
                            control_prediction=a['prediction'], treatment_prediction=b['prediction']))
    report = dict(partial=partial, rows=len(rows), expected_rows=count, complete_pairs=len(results),
                  control_score=sum(r['control_score'] for r in results),
                  treatment_score=sum(r['treatment_score'] for r in results),
                  wins=[r['qa_id'] for r in results if r['treatment_score'] > r['control_score']],
                  losses=[r['qa_id'] for r in results if r['treatment_score'] < r['control_score']],
                  errors=errors, live_calls=dict(calls), scope_calls=dict(scope_calls),
                  scope_statuses=[dict(arm=a, status=s, count=n) for (a, s), n in statuses.items()],
                  pairs=results, caveat='Scores include complete pairs only. Frozen packet/upstream '
                  'diagnostic; upstream token/time use omitted, not fresh HTTPS accuracy.')
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('partial', 'rows', 'complete_pairs', 'control_score',
                                           'treatment_score', 'wins', 'losses', 'errors')}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline', 'replay', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--setting', required=True)
    p.add_argument('--repeat', type=int, choices=(1, 2, 3), default=1)
    p.add_argument('--allow-partial', action='store_true')
    main(p.parse_args())
