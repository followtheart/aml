"""Separate route coverage, downstream admission, costs and paired answer changes."""
import argparse
import ast
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

from audit_https_funnel import norm
from inventory_snippet_provenance import parse_snippet
from inventory_source_admission import classify


def analyze(replay, audit):
    plan = json.loads(replay.with_suffix('.plan.json').read_text())
    checked = json.loads(audit.read_text())
    assert checked['status'] == 'passed'
    rows = [json.loads(line) for line in replay.read_text().splitlines() if line.strip()]
    expected = [(qid, arm) for i, qid in enumerate(plan['qa_ids']) for arm in
                (['control', 'candidate'] if i % 2 == 0 else ['candidate', 'control'])]
    assert [(r['qa_id'], r['arm']) for r in rows] == expected
    assert checked['questions'] == len(plan['qa_ids'])
    prepared = Path(plan['prepared'])
    assert hashlib.sha256(prepared.read_bytes()).hexdigest() == plan['prepared_sha256']
    conversations = [json.loads(line) for line in prepared.read_text().splitlines() if line.strip()]
    questions = {q['id']: (q, c) for c in conversations for q in c['qa']}
    run = Path(plan['run'])
    archived = {r['qa_id']: r for r in map(json.loads, (run / 'completed/data/results/personamem-v2-32k.jsonl').open())}
    with open('data/raw/personamem-v2.csv') as stream:
        metadata = list(csv.DictReader(stream))
    details = []
    for row in rows:
        qid, arm = row['qa_id'], row['arm']
        q, conv = questions[qid]
        meta = metadata[int(qid)]
        assert q['answer'] == meta['correct_answer']
        assert norm(q['question']) == norm(ast.literal_eval(meta['user_query'])['content'])
        turns = [norm(m['content']) for m in parse_snippet(meta['related_conversation_snippet']) if m['role'] == 'user']
        originals = [m for session in conv['sessions'] for m in session]
        diagnostics, search = row['diagnostics'], row['search']
        sources = {s['id']: s for s in diagnostics['answer_source_catalog']}
        admission = classify(turns, originals, search, sources, diagnostics['answer_constraint_catalog'])
        capacity = plan['capacities'][0 if arm == 'control' else 1]
        assert search['versions']['settings']['RECALL_VECTOR_LIMIT'] == capacity
        if arm == 'control':
            assert row['original_vector_route_exact']
        calls = row['response']['coverage_manifest']['provider_calls']
        assert calls <= search['versions']['settings']['SEARCH_MAX_CALLS']
        details.append(dict(qa_id=qid, arm=arm, gold=q['gold_labels'][0], prediction=row['prediction'], score=row['score'],
                            archived_prediction=archived[qid]['prediction'], archived_score=archived[qid]['score'],
                            admission=admission, search_calls=calls,
                            search_tokens=row['response']['coverage_manifest']['provider_tokens'],
                            provider_events=dict(Counter(e['kind'] for e in row['provider_events'])),
                            reader_events=sum(e['kind'] == 'llm' and str(e.get('stage', '')).startswith('eval.')
                                              for e in row['provider_events']),
                            search_degraded=search['coverage_manifest']['search_degraded'],
                            packet_hash=search['packet_hash'], catalog_size=len(sources)))
    controls = {r['qa_id']: r for r in details if r['arm'] == 'control'}
    candidates = {r['qa_id']: r for r in details if r['arm'] == 'candidate'}
    scores = {arm: sum(r['score'] for r in details if r['arm'] == arm) for arm in ['control', 'candidate']}
    assert scores == checked['scores']
    delivered = lambda r: r['admission']['stage'] in ['delivered_full_user_turn', 'delivered_constraint']
    changes = [dict(qa_id=qid, control=controls[qid], candidate=candidates[qid]) for qid in plan['qa_ids']
               if controls[qid]['prediction'] != candidates[qid]['prediction'] or delivered(controls[qid]) != delivered(candidates[qid])]
    return dict(status='complete', questions=len(controls), scores=scores, wins=checked['wins'], losses=checked['losses'],
                coverage={arm: dict(Counter(r['admission']['stage'] for r in details if r['arm'] == arm)) for arm in ['control', 'candidate']},
                catalog_gains=[q for q in controls if delivered(candidates[q]) and not delivered(controls[q])],
                catalog_losses=[q for q in controls if delivered(controls[q]) and not delivered(candidates[q])],
                fresh_control_vs_archived_changes=[q for q in controls if controls[q]['prediction'] != controls[q]['archived_prediction']],
                fresh_control_vs_archived_score=sum(r['archived_score'] for r in controls.values()),
                costs={arm: {key: sum(r[key] for r in details if r['arm'] == arm) for key in
                             ['search_calls', 'search_tokens', 'reader_events']} for arm in ['control', 'candidate']},
                provider_errors=checked['provider_errors'], changes=changes, rows=details,
                replay_sha256=hashlib.sha256(replay.read_bytes()).hexdigest(), audit_sha256=hashlib.sha256(audit.read_bytes()).hexdigest(),
                caveat='Full paired fresh downstream study with fixed original database and query plan. Literal source coverage is not semantic support. Fresh control differs from archived answers due to fresh calls and possible snapshot effects; single-run score gains require attribution and replication. Search cost includes one charged frozen planning call but omits its latency/tokens; reader_events counts wrapper invocations, not retries or model tokens.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['replay', 'audit', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.replay, args.audit)
    with args.output.open('x') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ['rows', 'changes', 'caveat']}))
