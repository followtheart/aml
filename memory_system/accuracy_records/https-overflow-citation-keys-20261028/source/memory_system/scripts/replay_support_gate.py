"""Offline replay of the deterministic support gate on frozen round-2 logs.

Rebuilds the model's support proposal from `choice_alignment` and re-runs
`validate_assessments` plus the witness pre-fill with the current code. No
model or network calls; gold is used only to label rows.
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('AML_FAKE', '1')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app import answer_choice, choice_witness  # noqa: E402

RUN = ROOT / 'data/analysis/personamem-v2-32k-20260921-131757-funnel'
results = [json.loads(l) for l in (RUN / 'results.snapshot.jsonl').open(encoding='utf-8')]
per_question = json.load((RUN / 'per_question.json').open(encoding='utf-8'))


def proposal(entries):
    options = []
    for e in entries:
        claims = []
        for c in e['claims']:
            cites = [dict(source_id=r['source_id'], quote=r['quote'], basis=r['basis'], subject=r['subject'])
                     for r in c.get('citations', []) + c.get('dropped_citations', [])]
            seen = {(r['source_id'], r['quote']): r for r in cites}
            claims.append(dict(text=c['text'], status=c['proposed_status'], premise_type=c.get('premise_type', 'unknown'),
                               citations=list(seen.values())[:3]))
        for r in e.get('removed_claims', []):
            claims.append(dict(text=r['text'], status='unsupported', citations=[]))
        kind = 'personal' if claims else e['kind']
        options.append(dict(letter=e['letter'], kind=kind, claims=claims))
    return dict(options=options)


targets = [int(a) for a in sys.argv[1:]] or list(range(26))
import importlib.util  # noqa: E402
spec = importlib.util.spec_from_file_location('app.answer_choice_baseline', RUN / 'answer_choice.run.py')
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)
promoted = dict(gold=[], distractor=[])
for index in targets:
    row, pq = results[index], per_question[index]
    sources = {c['id']: c for c in row['answer_source_catalog']}
    options = pq['options']
    gold = pq['gold']
    payload = proposal(row['choice_alignment'])
    entries = answer_choice.validate_assessments(payload, options, sources)
    before = {e['letter']: e['status'] for e in baseline.validate_assessments(payload, options, sources)}
    after = {e['letter']: e['status'] for e in entries}
    witnesses = choice_witness.build(options, sources)
    changed = {k: (before[k], after[k]) for k in after if before.get(k) != after[k]}
    for letter, (old, new) in changed.items():
        if old == 'unsupported' and new != 'unsupported':
            promoted['gold' if letter == gold else 'distractor'].append(f'{index}/{letter}:{new}')
    print(f"QA {index:2d} gold={gold} pred={row['predicted_answer']} gate_before={before[gold]} "
          f"gate_after={after[gold]} changed={changed}")
    g = next(e for e in entries if e['letter'] == gold)
    for c in g['claims']:
        print(f"    claim[{c['premise_type']}] {c['text']!r} -> {c['status']} errors={c['validation_errors']} "
              f"gaps={[r.get('strength_gap') for r in c['citations']]} dropped={[r['validation_errors'] for r in c['dropped_citations']]}")
    for r in g.get('removed_claims', []):
        print(f"    removed {r['text']!r}")
    for w in witnesses.get(gold, []):
        print(f"    witness [{w['source_id']} {w['role']}] {w['quote'][:110]!r} matched={w['matched']}")
print('promotions (unsupported -> non-unsupported, current code vs frozen baseline code):')
print(f"  gold={promoted['gold']}")
print(f"  distractor={promoted['distractor']}")
