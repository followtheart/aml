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
from option_source_links import plan as link_plan, validate as validate_links, suffix as link_suffix


async def main(a):
    cohort = load_cohort(a.run, a.prepared)
    provider, original_judge = llm.complete_json, ac._judge
    rows = []

    async def run(data, arm):
        r, q, packet, _, _ = data
        pending = copy.deepcopy(r['answer_calls'])
        live, mismatches, associations = [], [], []
        started = False
        comparison_active = False

        async def frozen(prompt, *pos, **kw):
            nonlocal started
            old = pending.pop(0) if pending else None
            if comparison_active and kw['stage'].startswith(('eval.choice_select', 'eval.choice_option_sources')):
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

        linked = None
        async def judge(prompt, schema, validator, stage, diagnostics, **kw):
            nonlocal comparison_active, linked
            eligible = schema.get('properties', {}).get('answer', {}).get('enum', [])
            spec = link_plan(prompt, eligible, diagnostics) if stage == 'eval.choice_select' else None
            if spec is not None and arm != 'preflight':
                comparison_active = True
            if arm == 'candidate' and spec is not None:
                linked = await original_judge(spec['prompt'], spec['schema'],
                    lambda value: validate_links(value, spec), 'eval.choice_option_sources', diagnostics, **kw)
                associations.append(dict(links=linked, source_ids=spec['source_ids']))
            if arm == 'candidate' and linked is not None and stage in ['eval.choice_select', 'eval.choice_select_review']:
                prompt += link_suffix(linked)
            return await original_judge(prompt, schema, validator, stage, diagnostics, **kw)

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
                    associations=associations, live_calls=live,
                    eligible=diag.get('answer_eligible_options'), diagnostics=diag)

    for data in cohort:
        await run(data, 'preflight')
    for index, data in enumerate(cohort):
        for arm in (['control', 'candidate'] if index % 2 == 0 else ['candidate', 'control']):
            row = await run(data, arm)
            rows.append(row)
            print(json.dumps({k: row[k] for k in ['qa_id', 'arm', 'prediction', 'score', 'associations']}), flush=True)
    files = [Path(__file__), Path(__file__).with_name('option_source_links.py')]
    report = dict(run=str(a.run), prepared=str(a.prepared),
                  prepared_sha256=hashlib.sha256(a.prepared.read_bytes()).hexdigest(),
                  scripts={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                  rows=rows, scores={arm: sum(r['score'] for r in rows if r['arm'] == arm)
                                     for arm in ['control', 'candidate']},
                  live_calls={arm: sum(len(r['live_calls']) for r in rows if r['arm'] == arm)
                              for arm in ['control', 'candidate']},
                  caveat='Archived full-cohort preflight. Original evidence, claim validation and eligibility frozen exactly; '
                  'both arms use fresh selection and all later calls on the same generic 2-4 option trigger; '
                  'non-triggered questions replay unchanged. Only triggered candidate ties first associate original source IDs with each option; '
                  'one separate association call supplies ID hints to original selector and existing review; no source rewriting or eligibility changes. Frozen earlier calls charge '
                  'call count but omit original latency/tokens. Diagnostic, not fresh HTTPS accuracy.')
    with a.output.open('x') as f:
        f.write(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(scores=report['scores'], live_calls=report['live_calls'])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['run', 'prepared', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    asyncio.run(main(parser.parse_args()))
