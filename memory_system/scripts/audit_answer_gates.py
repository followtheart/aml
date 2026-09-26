"""Separate missing citations, semantic rejection and final selection failures.

Offline diagnostic of observed gates; not proof the gold is supported by history.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def support_failure_detail(result, gold):
    """Keep the terminal per-option failure, including dropped citation causes."""
    raw = result.get('answer_support_validation', {}).get('errors', {}).get(gold)
    if raw is None:
        return None, []
    try:
        detail = json.loads(raw) if isinstance(raw, str) else raw
    except (ValueError, TypeError):
        return raw, ['unstructured_failure']
    if not isinstance(detail, dict):
        return detail, ['unstructured_failure']
    codes = set(detail.get('option_errors', []))
    for claim in detail.get('claims', []):
        codes.update(claim.get('errors', []))
        for citation in claim.get('dropped_citations', []):
            codes.update(citation.get('errors', []))
    return detail, sorted(codes)


def main(args):
    result_path = args.run / 'completed/data/results/personamem-v2-32k.jsonl'
    raw = result_path.read_bytes()
    results = [json.loads(x) for x in raw.splitlines() if x.strip()]
    prepared_path = Path(__file__).resolve().parents[1] / 'data/prepared/personamem-v2-32k.jsonl'
    questions = {(d['conversation_id'], q['id']): q
        for d in map(json.loads, prepared_path.read_text().splitlines()) for q in d['qa']}
    funnel = json.loads((args.run / 'funnel/audit.json').read_text())
    probes = {r['qa_id']: r['probes'] for r in funnel['rows']}
    rows = []
    for r in results:
        q = questions[r['conversation_id'], r['qa_id']]
        assert len(q['gold_labels']) == 1
        gold = q['gold_labels'][0]
        assert r['score'] == float(r['prediction'] == gold)
        e = next((e for e in r.get('choice_alignment', []) if e['letter'] == gold), None)
        if e is None:
            detail, codes = support_failure_detail(r, gold)
            rows.append(dict(qa_id=r['qa_id'], gold=gold, prediction=r['prediction'],
                gate='answer_error', related_probe_count=len(probes[r['qa_id']]),
                visible_related_probes=sum(bool(x['visible_source_ids']) for x in probes[r['qa_id']]),
                source_visibility='related_missing' if probes[r['qa_id']] and not any(
                    x['visible_source_ids'] for x in probes[r['qa_id']]) else
                    'related_visible' if probes[r['qa_id']] else 'not_probed',
                primary_text=None, primary_reason=None, primary_type=None,
                primary_proposed_status=None, primary_status=None, primary_verified=None,
                whole_option_verified=None, primary_source_ids=[], support_failure_codes=codes,
                support_failure_detail=detail, gold_status='unresolved', gold_validation='answer_error',
                error_stage=r.get('error_stage'), error_type=r.get('error_type'),
                error_detail=r.get('error_detail')))
            continue
        index = e.get('primary_claim')
        primary = e['claims'][index] if index is not None else None
        refs = [c for c in (primary or {}).get('citations', []) if c.get('valid')]
        if r['score']:
            gate = 'correct'
        elif gold in r['answer_eligible_options']:
            gate = 'final_selection'
        elif e.get('validation_status') == 'unresolved':
            gate = 'support_structure'
        elif gold in r['answer_blocked_options'] and e['status'] in ('generic', 'supported', 'partial'):
            gate = 'forget_constraint'
        elif e['kind'] == 'generic':
            gate = 'generic_option_rejected'
        elif primary and not refs:
            gate = 'primary_no_valid_citation'
        elif primary and primary.get('entailment_verified') is False:
            gate = 'primary_entailment_rejected'
        elif (primary and primary.get('entailment_verified') is True and
              primary['status'] == 'unsupported' and e.get('entailment_verified') is False):
            gate = 'recovery_blocked_by_whole_option'
        else:
            gate = 'other_or_multiple_gates'
        ps = probes[r['qa_id']]
        failure_detail, failure_codes = support_failure_detail(r, gold)
        visible = sum(bool(p['visible_source_ids']) for p in ps)
        visibility = 'related_visible' if visible else 'related_missing' if ps else 'not_probed'
        rows.append(dict(qa_id=r['qa_id'], gold=gold, prediction=r['prediction'], gate=gate,
            related_probe_count=len(ps), visible_related_probes=visible, source_visibility=visibility,
            primary_text=(primary or {}).get('text'), primary_reason=(primary or {}).get('reason'),
            primary_type=(primary or {}).get('premise_type'),
            primary_proposed_status=(primary or {}).get('proposed_status'),
            primary_status=(primary or {}).get('status'), primary_verified=(primary or {}).get('entailment_verified'),
            whole_option_verified=e.get('entailment_verified'),
            primary_source_ids=sorted({c['source_id'] for c in refs}),
            support_failure_codes=failure_codes, support_failure_detail=failure_detail,
            gold_status=e['status'], gold_validation=e.get('validation_status')))
    assert len({r['qa_id'] for r in rows}) == len(rows) == len(probes)
    artifact = dict(result_sha256=hashlib.sha256(raw).hexdigest(),
        caveat='Observed exclusion/selection gate, not causal proof. Related passages need not entail gold.',
        counts=dict(Counter(r['gate'] for r in rows)),
        support_failure_code_counts=dict(Counter(code for r in rows if r['gate'] == 'support_structure'
                                                 for code in r['support_failure_codes'])),
        gate_by_visibility={gate: dict(Counter(r['source_visibility'] for r in rows if r['gate'] == gate))
                            for gate in sorted({r['gate'] for r in rows})}, rows=rows)
    (args.run / 'funnel/answer-gates.json').write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(artifact['counts'], ensure_ascii=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    main(p.parse_args())
