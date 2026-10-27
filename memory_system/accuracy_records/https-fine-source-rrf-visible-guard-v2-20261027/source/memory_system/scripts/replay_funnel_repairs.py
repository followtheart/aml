"""Offline counterfactual gates on frozen logs; never a new benchmark score.

Reuses original support proposals and CE scores. No model call, database write,
or gold-conditioned selection. Newly admitted evidence needs fresh model verdicts.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ['AML_FAKE'] = '1'
read = Path.read_text
with patch.object(Path, 'read_text', lambda p, *a, **kw: '' if p.name == '.env' else read(p, *a, **kw)):
    from app import answer_choice, config, evidence_selection, persona_source


def load(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def previous_citation_status(entry):
    errors = [e for e in entry['validation_errors'] if e != 'not_entailed']
    if errors:
        return 'unsupported'
    if entry['kind'] == 'generic':
        return 'generic'
    supported = [c['proposed_status'] == 'supported' and not any(
        error != 'not_entailed' for error in c['validation_errors']) for c in entry['claims']]
    if supported and all(supported):
        return 'supported'
    primary = answer_choice._primary_index(entry['claims'], entry['option'])
    return 'partial' if primary is not None and supported[primary] else 'unsupported'


def replay(base):
    paths = {name: base / f'{name}.snapshot.jsonl' for name in ('memory', 'search', 'results')}
    events, searches, results = (load(paths[n]) for n in ('memory', 'search', 'results'))
    search_by_id = {s['search_id']: s for s in searches}
    originals = {(s['request_id'], s['message_index']): s for e in events for s in e['source_messages']}
    rows = []
    for result in results:
        search = search_by_id[result['search_id']]
        assert result['packet_hash'] == search['packet_hash']
        sources = {s['id']: copy.deepcopy(s) for s in result.get('answer_source_catalog', [])}
        restored = []
        for source in sources.values():
            original = originals.get((source.get('request_id'), source.get('message_index')), {})
            if original.get('role') == 'user' and persona_source.is_persona_message(original.get('content', '')):
                # Emulates evidence_units stamping the full message before excerpting.
                source['declared'] = 'persona'
                restored.append(source['id'])
        payload = dict(options=[dict(letter=e['letter'], kind=e['kind'], claims=[dict(
            text=c['text'], status=c['proposed_status'], citations=[{k: ref[k] for k in
                ('source_id', 'quote', 'basis', 'subject')} for ref in c['citations']]) for c in e['claims']])
            for e in result.get('choice_alignment', [])])
        entries = answer_choice.validate_assessments(payload, search['options'], sources) if payload['options'] else []
        before = {e['letter']: previous_citation_status(e) for e in result.get('choice_alignment', [])}
        changes = [dict(letter=e['letter'], before=before[e['letter']], after=e['status'],
            warnings=e['warnings'], removed_claims=e['removed_claims']) for e in entries
            if e['status'] != before[e['letter']] or e['warnings'] or e['removed_claims']]
        checks = answer_choice.entailment_checks(entries, sources, search['options']) if entries else []
        coarse_ids = set(search['cascade']['coarse']['selected_ids'])
        coarse = [c for c in search['ranked'] if c['id'] in coarse_ids]
        requirements = search['coverage_manifest']['coverage']['requirements']
        trace = {}
        fine_trace = search['cascade']['fine']
        fine = evidence_selection.select(coarse, config.CASCADE_FINE_LIMIT, requirements,
            lambda c: (c['_ce_score'] if c.get('_ce_score') is not None else -1e30, c.get('_fused', 0)),
            reserve_ids=set(fine_trace.get('reserved_ids', [])) - set(fine_trace.get('fused_reserved_ids', [])),
            fused_ids=fine_trace.get('fused_reserved_ids', []), trace=trace)
        carried = [r['candidate_id'] for r in trace['reservations'] if r['selected']]
        soft = {r['candidate_id'] for r in trace['reservations'] if r['reasons'] == ['fused_head']}
        positions = {c['id']: i for i, c in enumerate(fine)}
        admitted = evidence_selection.select(fine, config.CASCADE_LLM_LIMIT, [],
            lambda c: (c['id'] in carried and c['id'] not in soft, -positions[c['id']]), carry_ids=carried)
        old_ids = set(fine_trace['selected_ids'])
        new_ids = {c['id'] for c in fine}
        rows.append(dict(qa_id=result['qa_id'], original_prediction=result['prediction'],
            citation_changes=changes, citation_status={e['letter']: e['status'] for e in entries},
            persona_cards_restored=restored, checks_with_neighbors=[c['claim_id'] for c in checks if c['context_neighbors']],
            no_recorded_support_proposal=not bool(entries), fine_added=sorted(new_ids-old_ids),
            fine_removed=sorted(old_ids-new_ids), fine_ids=sorted(new_ids),
            listwise_count_admitted_ids=[c['id'] for c in admitted], option_reservations=[r for r in trace['reservations']
                if 'option_topic_coverage' in r['reasons']], selection_trace=trace))
    # Gold is annotations only, loaded AFTER all selection and validation work.
    gold_path = base / 'gold.snapshot.json'
    if gold_path.exists():
        gold = {q['id']: q for q in json.loads(gold_path.read_text(encoding='utf-8'))}
        for row in rows:
            row['gold_label_annotation'] = gold[row['qa_id']]['gold_labels'][0]
    return dict(mode='offline_counterfactual; not benchmark score',
        limitations=['Frozen support proposals and original coarse pool/CE scores.',
                     'No new entailment, contextual review, listwise or final-choice calls.',
                     'Listwise admission here checks count only, not provider/token budgets.',
                     'Format-failure QA has no recoverable support proposal; covered by synthetic regressions.'],
        input_hashes={n: hashlib.sha256(p.read_bytes()).hexdigest() for n, p in paths.items()},
        policy=answer_choice.VERSION, inferred_enabled=config.CHOICE_ALLOW_INFERRED,
        code_hashes={n: hashlib.sha256((ROOT / 'app' / n).read_bytes()).hexdigest()
                     for n in ('answer_choice.py', 'choice_premises.py', 'evidence_selection.py', 'evidence_units.py')},
        questions=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with patch('socket.socket.connect', side_effect=AssertionError('Network forbidden in offline replay')):
        report = replay(args.snapshot_dir)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    for row in report['questions']:
        label = row.get('gold_label_annotation')
        changed = next((c for c in row['citation_changes'] if c['letter'] == label), None)
        print(json.dumps(dict(qa=row['qa_id'], gold=label, gold_citation_change=changed,
            fine_added=len(row['fine_added']), persona_cards=len(row['persona_cards_restored']),
            format_unreplayable=row['no_recorded_support_proposal']), ensure_ascii=False))
    print(f"Replayed {len(report['questions'])} questions without model calls; no new accuracy measured.")


if __name__ == '__main__':
    main()
