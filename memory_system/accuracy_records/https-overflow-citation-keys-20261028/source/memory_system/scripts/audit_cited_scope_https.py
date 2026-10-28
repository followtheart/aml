"""Reproduce all reader calls and inspect cited-scope effects after HTTPS.

Shared Search isolates retrieval inputs, but live pre-scope responses can vary.
Removing a cited claim is a mechanism observation, not a causal score claim.
"""
import argparse
import asyncio
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


async def main(root):
    root = root.resolve()
    plan = json.loads((root / 'plan.json').read_text())
    if (root / 'exit-code').read_text().strip() != '0' or plan['reader_toggle'] != 'CHOICE_CITED_SCOPE':
        raise ValueError('Require successful completed cited-scope HTTPS trial')
    project = root / 'source/memory_system'
    sys.path.insert(0, str(project / 'scripts'))
    sys.path.insert(0, str(project))
    from probe_scope_boundary_current import execute
    from audit_https_funnel import reader_gate, reader_gate_detail, gold_forget_matches
    from app import config
    jobs = {j['qa']['id']: j for j in records(root / 'http-reader-inputs.jsonl')}
    traces = {t['search_id']: t for t in records(root / 'search/server-search-debug.jsonl')}
    arms, outcomes = {}, {}
    for arm in ('control', 'treatment'):
        enabled = arm == 'treatment'
        versions = json.loads((root / arm / 'reader-versions.json').read_text())
        stage = json.loads((root / arm / 'funnel/stage-transport.json').read_text())
        stages = {r['qa_id']: r for r in stage['rows']}
        results = records(root / arm / 'completed/data/results/personamem-v2-32k.jsonl')
        if len(results) != plan['questions']:
            raise ValueError('Incomplete cohort')
        summaries = []
        for row in results:
            qa = jobs[row['qa_id']]['qa']
            trace = copy.deepcopy(traces[row['search_id']])
            trace['versions']['settings'] = versions['settings']
            with patch('socket.socket.connect', side_effect=AssertionError('Offline audit')):
                replay = await execute((row, trace, qa), enabled, offline=True)
            scopes = row.get('answer_premise_scope', {})
            targets = {}
            for entry in replay['scope_inputs']:
                for i, claim in enumerate(entry['claims']):
                    targets[f"{entry['letter']}:{i}"] = dict(letter=entry['letter'], claim=claim['text'],
                        option=entry['option'], citations=claim.get('citations', []),
                        original_status=claim['status'], validation_errors=claim.get('validation_errors', []))
            removed_cited = [dict(claim_id=cid, **targets[cid]) for cid in scopes.get('removed', [])
                             if targets[cid]['citations']]
            boundary = next((i for i, c in enumerate(row['answer_calls']) if c['stage'] == 'eval.choice_premise_scope'), None)
            prefix = row['answer_calls'][:boundary] if boundary is not None else None
            gold = qa['gold_labels'][0]
            alignment = next(e for e in row['choice_alignment'] if e['letter'] == gold)
            observed = dict(qa_id=row['qa_id'], gold=gold, prediction=row['prediction'], score=row['score'],
                scope_status=scopes.get('status', 'no_targets'),
                scope_requested=len(scopes.get('requested', [])), removed_cited=removed_cited,
                replayed_calls=len(replay['fixed_calls']),
                prefix_sha256=hashlib.sha256(json.dumps(prefix, sort_keys=True).encode()).hexdigest() if prefix else None,
                reference_turns=stages[row['qa_id']]['visible']['catalog']['full_turns'],
                gold_alignment=alignment, eligible=row.get('answer_eligible_options', []),
                gold_valid_citations=sum(c.get('valid', False) for claim in alignment.get('claims', []) for c in claim.get('citations', [])),
                unresolved_support_options=row.get('answer_support_validation', {}).get('unresolved_options', []),
                gold_forget_constraint_blocked=gold in row.get('answer_blocked_options', []) and bool(gold_forget_matches(row, gold)),
                error_stage=row.get('error_stage'), answer_reader_mode=row.get('answer_reader_mode'))
            observed.update(reader_gate=reader_gate(observed), reader_gate_detail=reader_gate_detail(observed))
            summaries.append(observed)
        arms[arm] = summaries
        wrong = [r for r in summaries if r['score'] < 1 and r['reference_turns']]
        outcomes[arm] = dict(questions=len(summaries), score=sum(r['score'] for r in summaries),
            all_calls_replayed=sum(r['replayed_calls'] for r in summaries),
            scope_statuses=dict(Counter(r['scope_status'] for r in summaries)),
            removed_cited_questions=[r['qa_id'] for r in summaries if r['removed_cited']],
            removed_cited_claims=sum(len(r['removed_cited']) for r in summaries),
            wrong_with_reference=len(wrong), gates=dict(Counter(r['reader_gate'] for r in wrong)),
            visible=stage['summary']['visible_full_turn_question_counts'],
            turns=stage['summary']['delivered_available_turn_counts'],
            complete_sets=stage['summary']['available_set_complete_question_counts'])
    control = {r['qa_id']: r for r in arms['control']}
    changed = []
    for treatment in arms['treatment']:
        old = control[treatment['qa_id']]
        if old['score'] == treatment['score']:
            continue
        changed.append(dict(qa_id=treatment['qa_id'], control=old['score'], treatment=treatment['score'],
            same_prefix=old['prefix_sha256'] is not None and old['prefix_sha256'] == treatment['prefix_sha256'],
            removed_cited=treatment['removed_cited']))
    result = dict(summary=outcomes, changed=changed, arms=arms, provider_calls=0,
        auditor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), limitation=__doc__)
    with (root / 'scope-outcome-audit.json').open('x') as out:
        json.dump(result, out, ensure_ascii=False, indent=2)
        out.write('\n')
    print(json.dumps(outcomes, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    asyncio.run(main(parser.parse_args().run))
