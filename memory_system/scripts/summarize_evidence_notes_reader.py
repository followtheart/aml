"""Keep answer correctness and evidence-note transport validity as separate axes."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from summarize_reader_replay import read_complete


def summarize(path, allow_partial=False):
    plan = json.loads(path.with_suffix('.plan.json').read_text())
    prepared = Path(plan['prepared'])
    if hashlib.sha256(prepared.read_bytes()).hexdigest() != plan['prepared_sha256']:
        raise ValueError('Prepared input changed')
    questions = {q['id']: q for c in read_complete(prepared) for q in c['qa']}
    expected = plan['questions']
    if len(expected) != len(set(expected)) or set(expected) != set(questions):
        raise ValueError('Planned cohort differs from input')
    grouped = {}
    for row in read_complete(path, allow_partial):
        qid, arm = row['qa_id'], row['arm']
        if qid not in questions or arm not in ('direct', 'notes'):
            raise ValueError('Unknown question or arm')
        arms = grouped.setdefault(qid, {})
        if arm in arms:
            raise ValueError('Duplicate arm')
        arms[arm] = row
    missing = [qid for qid in expected if set(grouped.get(qid, {})) != {'direct', 'notes'}]
    if missing and not allow_partial:
        raise ValueError('Incomplete planned comparison')
    pairs = []
    errors = Counter()
    for qid in expected:
        if qid in missing:
            continue
        arms = grouped[qid]
        if any(arms['direct'][key] != arms['notes'][key]
               for key in ('conversation_id', 'source_sha256', 'packet_hash')):
            raise ValueError('Arms did not see identical source context')
        pair = dict(qa_id=qid)
        for arm, row in arms.items():
            answer = row.get('response', {}).get('answer')
            gold = questions[qid]['gold_labels']
            if questions[qid]['qa_type'] != 'single_choice' or len(gold) != 1:
                raise ValueError('Single-choice cohort required')
            answer_score = float(isinstance(answer, str) and answer == gold[0])
            if row['status'] == 'ok' and answer_score != row['score']:
                raise ValueError('Stored score disagrees with response')
            if row['status'] != 'ok':
                errors[arm + ':' + row.get('error', row.get('error_type', 'unknown'))] += 1
            pair[arm] = dict(validated_score=row['score'], answer_only_score=answer_score,
                             answer=answer, status=row['status'])
        pairs.append(pair)
    report = dict(partial=bool(missing), complete_pairs=len(pairs), expected_pairs=len(expected),
                  missing=missing, errors=dict(errors), pairs=pairs)
    for metric in ('validated_score', 'answer_only_score'):
        report[metric] = dict(
            direct=sum(r['direct'][metric] for r in pairs), notes=sum(r['notes'][metric] for r in pairs),
            wins=[r['qa_id'] for r in pairs if r['notes'][metric] > r['direct'][metric]],
            losses=[r['qa_id'] for r in pairs if r['notes'][metric] < r['direct'][metric]])
    report['caveat'] = ('validated_score retains zero for malformed notes or invalid quotes. '
        'answer_only_score is an additional diagnostic metric counting returned letters even when notes fail; '
        'it does not certify grounding or repair failed rows. Both arms use the same frozen catalog. '
        'No end-to-end HTTPS or semantic-validity claim.')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args()
    report = summarize(args.replay, args.allow_partial)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('pairs','missing','caveat')}))
