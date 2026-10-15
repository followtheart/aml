"""Trace audited CE carrier gains through listwise admission and reader catalogs."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

from audit_https_funnel import norm
from inventory_snippet_provenance import parse_snippet


def analyze(replay, upstream, audit):
    plan = json.loads(replay.with_suffix('.plan.json').read_text())
    checked = json.loads(audit.read_text())
    fine = json.loads(upstream.read_text())
    assert checked['status'] == fine['status'] == 'passed'
    records = [json.loads(line) for line in replay.read_text().splitlines() if line.strip()]
    ids = plan['qa_ids']
    assert checked['questions'] == fine['questions'] == len(ids)
    expected = [(q, arm) for i, q in enumerate(ids) for arm in
                (['control', 'candidate'] if i % 2 == 0 else ['candidate', 'control'])]
    assert [(r['qa_id'], r['arm']) for r in records] == expected
    assert hashlib.sha256(Path(plan['probe']).read_bytes()).hexdigest() == plan['probe_sha256']
    carriers = {r['qa_id']: r for r in fine['rows']}
    assert set(carriers) == set(ids)
    with open('data/raw/personamem-v2.csv') as stream:
        metadata = list(csv.DictReader(stream))
    prepared = [json.loads(line) for line in Path(plan['prepared']).read_text().splitlines() if line.strip()]
    originals = {q['id']: [norm(m['content']) for session in c['sessions'] for m in session
                          if m['role'] == 'user'] for c in prepared for q in c['qa']}
    questions = {q['id']: q for c in prepared for q in c['qa']}
    rows = []
    for qid in ids:
        turns = [norm(m['content']) for m in parse_snippet(metadata[int(qid)]['related_conversation_snippet'])
                 if m['role'] == 'user']
        valid = [t for t in turns if t and any(t in original for original in originals[qid])]
        carrier_ids = set(carriers[qid]['carriers'])
        arms = {}
        for row in (r for r in records if r['qa_id'] == qid):
            rank = row['ranking']['cascade']
            upstream_arm = {'control': 'fresh_control', 'candidate': 'isolated'}[row['arm']]
            assert rank['fine']['selected_ids'] == carriers[qid]['arms'][upstream_arm]['ids']
            diag = row['diagnostics']
            sources = diag.get('answer_source_catalog', [])
            constraints = diag.get('answer_constraint_catalog', [])
            arms[row['arm']] = dict(
                fine_carriers=sorted(carrier_ids.intersection(rank['fine']['selected_ids'])),
                admitted_carriers=sorted(carrier_ids.intersection(rank['listwise']['candidate_ids'])),
                selected_carriers=sorted(carrier_ids.intersection(rank['listwise']['selected_ids'])),
                catalog_coverage=any(t in norm(s['text']) for t in valid for s in sources
                                     if (s.get('declared') or s['role']) in ['user', 'persona']),
                constraint_coverage=any(t in norm(s['text']) for t in valid for s in constraints),
                eligible=diag.get('answer_eligible_options'), prediction=row['prediction'], score=row['score'])
        rows.append(dict(qa_id=qid, gold=questions[qid]['answer'], valid_user_turns=len(valid), arms=arms))
    summary = {arm: {key: sum(bool(r['arms'][arm][key]) for r in rows)
                     for key in ['fine_carriers', 'admitted_carriers', 'selected_carriers', 'catalog_coverage', 'constraint_coverage']}
               for arm in ['control', 'candidate']}
    return dict(status='complete', questions=len(ids), scores=checked['scores'],
                wins=checked['wins'], losses=checked['losses'], coverage=summary, rows=rows,
                input_hashes={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [replay, upstream, audit]},
                caveat='Original full user-turn carrier coverage only, not entailment. Catalog coverage is independently checked against valid original turns. Arm outcomes include fresh downstream model variability; coverage changes alone do not establish a causal answer gain. Upstream CE cost excluded from reader budgets.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['replay', 'upstream', 'audit', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.replay, args.upstream, args.audit)
    with args.output.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ['rows', 'caveat', 'input_hashes']}))
