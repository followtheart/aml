"""Full population: archived preflight, then paired fresh selector/downstream arms."""
import argparse
import asyncio
import copy
import hashlib
import json
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import answer_choice as ac, budget, config, eval_scoring, llm, metrics
from probe_question_source_filter import load_cohort, sha
from replay_query_context import recorded_response
from pairwise_generic_selection import pairs, winner


async def main(a):
    cohort = load_cohort(a.run, a.prepared)
    provider, original_judge = llm.complete_json, ac._judge
    rows = []

    async def run(data, arm):
        r, q, packet, _, _ = data
        pending = copy.deepcopy(r['answer_calls'])
        live, mismatches, comparisons = [], [], []
        started = False
        comparison_active = False

        async def frozen(prompt, *pos, **kw):
            nonlocal started
            old = pending.pop(0) if pending else None
            if comparison_active and kw['stage'].startswith(('eval.choice_select', 'eval.choice_pairwise')):
                started = True
            if not started:
                if not old or old['stage'] != kw['stage'] or old['prompt_sha256'] != sha(prompt):
                    mismatches.append(kw['stage'])
                    raise ValueError('Frozen pre-selection trajectory diverged')
                budget.current.get().before_call()
                return recorded_response(old)
            event = dict(stage=kw['stage'], prompt=prompt, prompt_sha256=sha(prompt),
                         schema=kw.get('schema'), system=kw.get('system'))
            live.append(event)
            try:
                response = await provider(prompt, *pos, **kw)
                event.update(status='ok', response=response)
                return response
            except Exception as exc:
                cause, malformed = exc, isinstance(exc, ValueError)
                while cause is not None:
                    malformed = malformed or isinstance(cause, metrics.ResponseParseError)
                    cause = cause.__cause__
                event.update(status='error', error_type=type(exc).__name__,
                             error_detail=str(exc)[:2000], provider_parse=malformed)
                raise

        async def judge(prompt, schema, validator, stage, diagnostics, **kw):
            nonlocal comparison_active
            eligible = schema.get('properties', {}).get('answer', {}).get('enum', [])
            planned = pairs(prompt, eligible) if stage == 'eval.choice_select' else []
            if planned and arm != 'preflight':
                comparison_active = True
            if arm != 'candidate':
                planned = []
            if not planned:
                return await original_judge(prompt, schema, validator, stage, diagnostics, **kw)
            answers = []
            for order, pair_prompt in planned:
                pair_schema = copy.deepcopy(schema)
                pair_schema['properties']['answer']['enum'] = order

                def validate(value):
                    if not isinstance(value, dict) or set(value) != {'answer'} or value['answer'] not in order:
                        raise ValueError('Pairwise answer outside verified pair')
                    return value['answer']

                answer = await original_judge(pair_prompt, pair_schema, validate,
                                             'eval.choice_pairwise', diagnostics, **kw)
                answers.append((order, answer))
            selected, scores = winner(eligible, answers)
            comparisons.append(dict(results=answers, scores=scores, winner=selected,
                                    fallback=selected is None))
            if selected is None:
                return await original_judge(prompt, schema, validator, stage, diagnostics, **kw)
            return validator({'answer': selected})

        with patch.multiple(config, **r['versions']['settings']), patch.object(ac, '_judge', judge), \
                patch.object(llm, 'complete_json', frozen), budget.scope(seconds=300, calls=40, tokens=200000):
            prediction, score, diag = await eval_scoring.evaluate(q, copy.deepcopy(packet))
        if mismatches:
            raise ValueError(str(mismatches))
        if arm == 'preflight' and (pending or live or (prediction, score) != (r['prediction'], r['score'])):
            raise ValueError('Archived control not reproduced')
        for field in ['choice_alignment', 'answer_eligible_options', 'answer_blocked_options',
                      'answer_source_catalog', 'answer_constraint_catalog']:
            if diag.get(field) != r.get(field):
                raise ValueError('Pre-selection state changed: ' + field)
        return dict(qa_id=r['qa_id'], arm=arm, prediction=prediction, score=score,
                    archived_prediction=r['prediction'], archived_score=r['score'],
                    comparisons=comparisons, live_calls=live,
                    eligible=diag.get('answer_eligible_options'), diagnostics=diag)

    for data in cohort:
        await run(data, 'preflight')
    for index, data in enumerate(cohort):
        for arm in (['control', 'candidate'] if index % 2 == 0 else ['candidate', 'control']):
            row = await run(data, arm)
            rows.append(row)
            print(json.dumps({k: row[k] for k in ['qa_id', 'arm', 'prediction', 'score', 'comparisons']}), flush=True)
    files = [Path(__file__), Path(__file__).with_name('pairwise_generic_selection.py')]
    report = dict(run=str(a.run), prepared=str(a.prepared),
                  prepared_sha256=hashlib.sha256(a.prepared.read_bytes()).hexdigest(),
                  scripts={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                  rows=rows, scores={arm: sum(r['score'] for r in rows if r['arm'] == arm)
                                     for arm in ['control', 'candidate']},
                  live_calls={arm: sum(len(r['live_calls']) for r in rows if r['arm'] == arm)
                              for arm in ['control', 'candidate']},
                  caveat='Archived full-cohort preflight. Original evidence, claim validation and eligibility frozen exactly; '
                  'both arms use fresh selection and all later calls on the same generic 3-4 option trigger; '
                  'non-triggered questions replay unchanged. Only triggered candidate ties use all-pairs selection; '
                  'tied tournament falls back to original selector, existing review retained. Frozen earlier calls charge '
                  'call count but omit original latency/tokens. Diagnostic, not fresh HTTPS accuracy.')
    with a.output.open('x') as f:
        f.write(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(scores=report['scores'], live_calls=report['live_calls'])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['run', 'prepared', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    asyncio.run(main(parser.parse_args()))
