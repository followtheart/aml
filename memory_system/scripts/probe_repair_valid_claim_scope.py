"""Probe semantic scope after retaining structurally valid pre-repair claims.

Diagnostic only: does not change production repair or grant evidence support.
"""
import argparse
import asyncio
import copy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import answer_choice as ac, budget, config


async def main(args):
    if config.FAKE:
        raise ValueError('Real provider required')
    rows = [json.loads(line) for line in args.results.read_bytes().splitlines(keepends=True)
            if line.endswith(b'\n')]
    row = next(r for r in rows if r['qa_id'] == args.question)
    raw = next(c['response'] for c in row['answer_calls'] if c['stage'] == 'eval.choice_support')
    assessments = ac.assessment_rows(raw)
    # This probe deliberately handles only uncited extraction trajectories.
    if any(c.get('citations') for a in assessments for c in a['claims']):
        raise ValueError('Requires uncited initial claims')
    options = [e['option'] for e in row['choice_alignment']]
    original = ac.validate_assessments({'options': assessments}, options, {})
    retained = []
    for entry in original:
        candidate = copy.deepcopy(entry)
        candidate['claims'] = [c for c in candidate['claims']
                               if not c['validation_errors'] and c['option_span']]
        if args.sort_spans:
            candidate['claims'].sort(key=lambda c: c['option_span']['start'])
        candidate['validation_errors'] = []
        candidate['validation_status'] = 'valid'
        candidate['kind'] = 'personal' if candidate['claims'] else 'generic'
        candidate['primary_claim'] = ac._primary_index(candidate['claims'], candidate['option'])
        candidate['status'] = ac._option_status(candidate['kind'], candidate['claims'], [], candidate['option'])
        retained.append(candidate)
    with args.output.open('x') as output:
        for repeat in range(args.repeat):
            diagnostics = {}
            with budget.scope(seconds=180, calls=20, tokens=120000):
                scoped = await ac.reclassify_uncited(copy.deepcopy(retained), diagnostics)
            result = dict(qa_id=args.question, repetition=repeat + 1,
                          sort_spans=args.sort_spans,
                          original=original, retained=retained, scoped=scoped,
                          diagnostics=diagnostics,
                          caveat='Selected scope diagnostic only; no downstream entailment, selection or accuracy claim.')
            output.write(json.dumps(result, ensure_ascii=False) + '\n')
            output.flush()
            print(json.dumps(dict(repetition=repeat + 1,
                scope_status=diagnostics.get('answer_premise_scope', {}).get('status'),
                remaining_counts={e['letter']: len(e['claims']) for e in scoped})), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results', type=Path, required=True)
    p.add_argument('--question', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--repeat', type=int, default=2)
    p.add_argument('--sort-spans', action='store_true')
    asyncio.run(main(p.parse_args()))
