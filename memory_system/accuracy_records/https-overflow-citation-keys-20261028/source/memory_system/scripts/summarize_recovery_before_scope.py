"""Audit complete paired scope-order diagnostics against their frozen snapshot."""
import argparse
import hashlib
import json
from pathlib import Path

from summarize_reader_replay import read_complete


def summarize(directory, allow_partial=False):
    plan = json.loads((directory / 'plan.json').read_text())
    snapshot = directory / 'reader-snapshot.jsonl'
    if hashlib.sha256(snapshot.read_bytes()).hexdigest() != plan['snapshot_sha256']:
        raise ValueError('Archived snapshot changed')
    originals = {r['qa_id']: r for r in read_complete(snapshot) if r['arm'] == 'window'}
    questions = plan['questions']
    if len(questions) != len(set(questions)) or not set(questions) <= set(originals):
        raise ValueError('Invalid planned cohort')
    pairs, missing, errors, unverified_orders = [], [], [], []
    for qid in questions:
        original = originals[qid]
        calls = original['diagnostics']['answer_calls']
        boundary = next(i for i, c in enumerate(calls) if c['stage'] == 'eval.choice_premise_scope')
        path = directory / (qid + '.jsonl')
        rows = read_complete(path, allow_partial) if path.exists() else []
        arms = {}
        for row in rows:
            rep, enabled = row['repetition'], row['enabled']
            if (type(enabled) is not bool or row['qa_id'] != qid
                    or not 1 <= rep <= plan['repetitions']
                    or row['packet_hash'] != original['packet_hash']):
                raise ValueError('Question, arm, repetition or packet mismatch')
            key = (rep, enabled)
            if key in arms:
                raise ValueError('Duplicate arm')
            diagnostics = row['diagnostics']
            prefix = diagnostics['answer_calls'][:boundary]
            fields = ('stage', 'prompt_sha256', 'response', 'status')
            signature = lambda cs: [{k: c.get(k) for k in fields} for c in cs]
            if signature(prefix) != signature(calls[:boundary]):
                raise ValueError('Frozen prefix drift')
            failed = diagnostics.get('answer_status') == 'error' or diagnostics.get('error_stage')
            if enabled != (diagnostics.get('answer_recovery_order') == 'before_scope'):
                # The marker is written after recovery returns. A failed early
                # verification can exit before that point; retain its zero score
                # and report missing confirmation, never silently omit the row.
                if not (enabled and failed and 'answer_recovery_order' not in diagnostics):
                    raise ValueError('Runtime recovery order differs from assigned arm')
                unverified_orders.append(dict(qa_id=qid, repetition=rep, enabled=enabled))
            if failed:
                errors.append(dict(qa_id=qid, repetition=rep, enabled=enabled))
            arms[key] = row
        for rep in range(1, plan['repetitions'] + 1):
            if any((rep, enabled) not in arms for enabled in (False, True)):
                missing.append(dict(qa_id=qid, repetition=rep))
                continue
            control, treatment = arms[rep, False], arms[rep, True]
            pairs.append(dict(qa_id=qid, repetition=rep,
                control=control['score'], treatment=treatment['score'],
                control_prediction=control['prediction'], treatment_prediction=treatment['prediction'],
                control_calls=len(control['live_calls']), treatment_calls=len(treatment['live_calls']),
                control_eligible=control['diagnostics'].get('answer_eligible_options', []),
                treatment_eligible=treatment['diagnostics'].get('answer_eligible_options', [])))
    if missing and not allow_partial:
        raise ValueError('Incomplete planned comparison')
    return dict(partial=bool(missing), expected_pairs=len(questions)*plan['repetitions'],
        complete_pairs=len(pairs), missing=missing, errors=errors,
        unverified_orders=unverified_orders,
        control=sum(r['control'] for r in pairs), treatment=sum(r['treatment'] for r in pairs),
        wins=[r['qa_id'] for r in pairs if r['treatment'] > r['control']],
        losses=[r['qa_id'] for r in pairs if r['treatment'] < r['control']],
        live_calls={k: sum(r[k+'_calls'] for r in pairs) for k in ('control', 'treatment')},
        pairs=pairs, excluded=plan['excluded'],
        caveat='Fixed source-window packets and frozen upstream responses; live scope/recovery and downstream. '
               'Frozen prefix counts calls but does not reproduce provider token usage or elapsed time. '
               'These are diagnostic scores, not full HTTPS or resource-budget equivalence.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args()
    report = summarize(args.directory, args.allow_partial)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('pairs', 'missing', 'caveat')}))
