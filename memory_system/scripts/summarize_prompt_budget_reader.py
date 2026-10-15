"""Audit complete-cohort paired prompt-budget ranking and reader replays."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import answer_choice, eval_scoring, evidence_packet
from summarize_reader_replay import read_complete


def summarize(run, prepared, replay, treatment_bytes, allow_partial=False):
    summary = json.loads((run / 'funnel/https-summary.json').read_text())
    if summary['partial'] or summary['exit_code'] != 0:
        raise ValueError('Require complete successful HTTPS baseline')
    prepared_rows = read_complete(prepared)
    prepared_sha = hashlib.sha256(prepared.read_bytes()).hexdigest()
    if prepared_sha != summary['prepared_sha256']:
        # Older archives ran one conversation from the complete dataset. Verify
        # the exact extracted conversation against that original hashed input.
        manifest = json.loads((run / 'isolated-manifest.json').read_text())
        original_data = Path(manifest['data'])
        if hashlib.sha256(original_data.read_bytes()).hexdigest() != summary['prepared_sha256']:
            raise ValueError('Archived full input changed')
        originals = {c['conversation_id']: c for c in read_complete(original_data)}
        if any(c != originals.get(c['conversation_id']) for c in prepared_rows):
            raise ValueError('Prepared subset differs from archived conversation')
    questions = {q['id']: q for c in prepared_rows for q in c['qa']}
    archived = read_complete(run / 'completed/data/results/personamem-v2-32k.jsonl')
    baseline = {r['qa_id']: r for r in archived}
    if len(baseline) != len(archived) or set(baseline) != set(questions):
        raise ValueError('Incomplete or duplicate source cohort')
    searches = {r['search_id']: r for r in read_complete(run / 'server-search-debug.jsonl')}
    grouped = {}; errors = []; order = []
    for row in read_complete(replay, allow_partial):
        qid = row['qa_id']
        if qid not in baseline:
            raise ValueError('Unexpected question')
        original = baseline[qid]; search = searches[original['search_id']]
        control_bytes = search['versions']['settings']['RERANK_MAX_PROMPT_BYTES']
        if control_bytes == treatment_bytes or row['prompt_bytes'] not in [control_bytes, treatment_bytes]:
            raise ValueError('Unexpected budget arm')
        arm = 'control' if row['prompt_bytes'] == control_bytes else 'treatment'
        if row['capacity_limit'] is not None or row['shared_sources'] or row['repetition'] != 1:
            raise ValueError('Intervention differs from prompt-budget-only plan')
        if (not row['frozen_packet_verified'] or row['baseline_packet_hash'] != original['packet_hash']
                or evidence_packet.digest(row['packet']) != row['packet_hash']):
            raise ValueError('Packet identity mismatch')
        sources, constraints = answer_choice.build_catalog(row['packet'])
        diagnostics = row['diagnostics']
        if (list(sources.values()) != diagnostics['answer_source_catalog']
                or list(constraints.values()) != diagnostics['answer_constraint_catalog']):
            raise ValueError('Reader catalog differs from packet')
        qa = questions[qid]
        if row['score'] != eval_scoring.choice_score(row['prediction'], qa['gold_labels'], qa['qa_type']):
            raise ValueError('Score mismatch')
        if diagnostics.get('answer_status') == 'error':
            if row['score'] != 0:
                raise ValueError('Failed answer must score zero')
            errors.append(dict(qa_id=qid, arm=arm))
        for call in row['calls']:
            if hashlib.sha256(call['prompt'].encode()).hexdigest() != call['prompt_sha256']:
                raise ValueError('Ranking prompt hash mismatch')
        arms = grouped.setdefault(qid, {})
        if arm in arms:
            raise ValueError('Duplicate arm')
        arms[arm] = row; order.append((qid, arm))
    expected_order = [(r['qa_id'], a) for i, r in enumerate(archived)
                      for a in (['control', 'treatment'] if i % 2 == 0 else ['treatment', 'control'])]
    if order != expected_order[:len(order)]:
        raise ValueError('Unexpected arm order or skipped question')
    missing = [qid for qid in questions if set(grouped.get(qid, {})) != {'control', 'treatment'}]
    if missing and not allow_partial:
        raise ValueError('Incomplete paired experiment')
    def first_prompt(row):
        return next((c['prompt_sha256'] for c in row['calls'] if c['stage'] == 'search.listwise'), None)

    pairs = [dict(qa_id=qid, packet_changed=arms['control']['packet_hash'] != arms['treatment']['packet_hash'],
        same_initial_ranking_prompt=(first_prompt(arms['control']) is not None
            and first_prompt(arms['control']) == first_prompt(arms['treatment'])),
        admitted_added=sorted(set(arms['treatment']['cascade']['listwise']['candidate_ids'])
            - set(arms['control']['cascade']['listwise']['candidate_ids'])),
        admitted_removed=sorted(set(arms['control']['cascade']['listwise']['candidate_ids'])
            - set(arms['treatment']['cascade']['listwise']['candidate_ids'])),
        **{arm: dict(score=arms[arm]['score'], prediction=arms[arm]['prediction'],
                    admitted=len(arms[arm]['cascade']['listwise']['candidate_ids']),
                    source_count=len(arms[arm]['diagnostics']['answer_source_catalog']))
           for arm in ['control', 'treatment']}) for qid, arms in grouped.items() if qid not in missing]
    return dict(prepared_sha256=prepared_sha, archived_input_sha256=summary['prepared_sha256'],
        expected_pairs=len(questions), complete_pairs=len(pairs), partial=bool(missing), missing=missing,
        control=sum(p['control']['score'] for p in pairs), treatment=sum(p['treatment']['score'] for p in pairs),
        wins=[p['qa_id'] for p in pairs if p['treatment']['score'] > p['control']['score']],
        losses=[p['qa_id'] for p in pairs if p['treatment']['score'] < p['control']['score']],
        errors=errors, pairs=pairs, caveat='Full-cohort identity, alternating order, packet/catalog identity, '
        'ranking prompt hashes and scores checked. Frozen retrieval/CE; diagnostic reader budget. '
        'Does not prove live HTTPS improvement, semantic entailment or attribution of every score change.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['run', 'prepared', 'replay', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--treatment-bytes', type=int, required=True)
    parser.add_argument('--allow-partial', action='store_true')
    a = parser.parse_args(); report = summarize(a.run, a.prepared, a.replay, a.treatment_bytes, a.allow_partial)
    a.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ['pairs', 'missing', 'caveat']}))
