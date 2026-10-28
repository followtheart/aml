"""Fixed-prefix identity-grounding diagnostic on every augmented scope-on case.

All upstream calls, evidence and checks are archived and verified. Only the
entailment instruction changes; entailment and later selection run live. Cases
come from structural packet changes in the parent factorial, not their scores.
This is a selected-case diagnostic, not a whole-cohort or HTTPS accuracy result.
"""
import argparse
import asyncio
from collections import Counter
import copy
import hashlib
import json
import logging
from pathlib import Path
from unittest.mock import patch

from audit_core_gap_scope_factorial import replay as verify_parent, response
from probe_core_gap_scope_factorial import records, normalized, sha
from app import answer_choice, budget, config, eval_scoring, llm
from evaluation_model_guard import check_runtime


NOTE = ('\nIdentity evidence: A claim about the user\'s nationality, ancestry, ethnic origin, '
        'or cultural heritage requires source evidence establishing that identity or family origin. '
        'Language use, a name, residence, cuisine, or cultural familiarity alone does not establish it. '
        'A plausible explanation is not evidence of personal identity. Check who the source describes; '
        'do not assign a fictional or quoted narrator\'s identity to the user without evidence linking them. '
        'This rule does not add identity requirements to generic advice, hypothetical suggestions, '
        'or literal evidence of topical interest. Apply the ordinary evidence rules to other claims.')


async def execute(row, case, settings, enabled, saved=None):
    calls = row['diagnostics']['answer_calls']
    boundary = next(i for i, c in enumerate(calls) if c['stage'] == 'eval.choice_entailment')
    pending = copy.deepcopy(calls[:boundary])
    remaining = copy.deepcopy(saved['diagnostics']['answer_calls'][boundary:]) if saved else None
    requests, replayed, violations, usages = [], [], [], []
    provider, judge = llm.complete_json, answer_choice._judge

    async def grounded(prompt, schema, validator, stage, diagnostics, **kwargs):
        if enabled and stage == 'eval.choice_entailment':
            prompt += NOTE
        return await judge(prompt, schema, validator, stage, diagnostics, **kwargs)

    async def capture(prompt, *a, **kw):
        identity = dict(stage=kw['stage'], prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest())
        usages.append(budget.current.get())
        if pending:
            old = pending.pop(0)
            if any(old[k] != v for k, v in identity.items()):
                violations.append(identity)
                raise ValueError('Prefix request differs')
            budget.current.get().before_call()
            replayed.append(identity)
            return response(old)
        if violations:
            raise ValueError('Refusing continuation after prefix mismatch')
        requests.append(dict(**identity, prompt=prompt,
            **{k: kw[k] for k in ('schema', 'system', 'max_tokens', 'attempts', 'timeout') if k in kw}))
        if remaining is not None:
            if not remaining:
                violations.append(identity)
                raise ValueError('Unexpected audit tail call')
            old = remaining.pop(0)
            if any(old[k] != v for k, v in identity.items()):
                violations.append(identity)
                raise ValueError('Tail audit request differs')
            budget.current.get().before_call()
            return response(old)
        return await provider(prompt, *a, **kw)

    with patch.multiple(config, **dict(settings, CHOICE_CITED_SCOPE=True)), \
            patch.object(answer_choice, '_judge', grounded), patch.object(llm, 'complete_json', capture):
        prediction, score, diagnostics = await eval_scoring.evaluate(case['qa'], copy.deepcopy(case['packets']['augmented']))
    diagnostics = normalized(diagnostics)
    if pending or violations or remaining or not requests:
        raise ValueError('Incomplete prefix/tail verification')
    if requests[0]['stage'] != 'eval.choice_entailment':
        raise ValueError('Tail starts at wrong boundary')
    expected_prompt = row['requests'][boundary]['prompt'] + (NOTE if enabled else '')
    if requests[0]['prompt'] != expected_prompt:
        raise ValueError('Uncontrolled entailment input change')
    for key in ('answer_source_catalog', 'answer_constraint_catalog', 'answer_premise_scope'):
        if diagnostics.get(key) != row['diagnostics'].get(key):
            raise ValueError('Fixed upstream diagnostic changed: ' + key)
    if diagnostics['answer_calls'][:boundary] != calls[:boundary]:
        raise ValueError('Prefix diagnostics changed')
    if saved:
        if (prediction, score) != (saved['prediction'], saved['score']):
            raise ValueError('Tail answer replay differs')
        for key in ('answer_calls', 'answer_source_catalog', 'answer_constraint_catalog',
                    'choice_alignment', 'answer_eligible_options', 'answer_premise_scope', 'answer_validation'):
            if diagnostics.get(key) != saved['diagnostics'].get(key):
                raise ValueError('Tail diagnostic replay differs: ' + key)
        if requests != saved['requests']:
            raise ValueError('Tail request replay differs')
    usage = usages[-1]
    return dict(prediction=prediction, score=score, diagnostics=diagnostics, requests=requests,
                fixed_calls=replayed, total_budget_calls=usage.calls,
                live_tokens=usage.tokens, live_calls=usage.calls-len(replayed))


async def main(args):
    check_runtime()
    if config.FAKE or config.CE_MODEL != 'qwen3-rerank':
        raise ValueError('Real unchanged models required')
    parent = json.loads((args.run/'plan.json').read_text())
    audit = json.loads((args.run/'audit.json').read_text())
    if sha(args.run/'plan.json') != audit['plan_sha256'] or sha(args.run/'results.jsonl') != audit['results_sha256']:
        raise ValueError('Parent audit is stale')
    cases = {c['qa']['id']: c for c in parent['cases']}
    rows = [r for r in records(args.run/'results.jsonl') if r['packet_mode']=='augmented' and r['cited_scope']]
    if len(rows) != len(cases)*parent['repeats']:
        raise ValueError('Incomplete parent strata')
    verified = 0
    for row in rows:
        verified += await verify_parent(row, cases[row['qa_id']], parent['settings'])
    args.output.mkdir(parents=True, exist_ok=False)
    plan = dict(parent=str(args.run.resolve()), parent_plan_sha256=sha(args.run/'plan.json'),
                parent_results_sha256=sha(args.run/'results.jsonl'), script_sha256=sha(Path(__file__)),
                note=NOTE, model=config.LLM_MODEL, settings=parent['settings'],
                strata=[dict(qa_id=r['qa_id'], archive_repetition=r['repetition']) for r in rows],
                repeats=args.repeat, expected_tails=len(rows)*2*args.repeat,
                preflight_calls=verified, protocol=__doc__,
                limitation='Replayed calls count against the call budget. Archived token use and latency are not simulated; live tails are accounted separately.')
    (args.output/'plan.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(event='preflight', calls=verified, tails=plan['expected_tails'])), flush=True)
    handler = logging.FileHandler(args.output/'provider.log', mode='x')
    logging.getLogger('aml.metrics').addHandler(handler)
    logging.getLogger('aml.metrics').setLevel(logging.INFO)
    results = []
    for repetition in range(1, args.repeat+1):
        for index, row in enumerate(rows):
            for enabled in ([False, True] if (index+repetition)%2 else [True, False]):
                result = await execute(row, cases[row['qa_id']], parent['settings'], enabled)
                result.update(qa_id=row['qa_id'], archive_repetition=row['repetition'],
                              repetition=repetition, enabled=enabled)
                with (args.output/'results.jsonl').open('a') as f:
                    f.write(json.dumps(result, ensure_ascii=False)+'\n')
                results.append(result)
                print(json.dumps({k:v for k,v in result.items() if k not in
                      ('diagnostics','requests','fixed_calls')}), flush=True)
                with patch('socket.socket.connect', side_effect=AssertionError('Offline tail audit')):
                    await execute(row, cases[row['qa_id']], parent['settings'], enabled, saved=result)
    events = [dict(p.split('=',1) for p in line.split() if '=' in p)
              for line in (args.output/'provider.log').read_text().splitlines() if 'provider_call ' in line]
    if len(results) != plan['expected_tails']:
        raise ValueError('Incomplete diagnostic')
    if len(events)!=sum(r['live_calls'] for r in results) or sum(int(e['total_tokens']) for e in events)!=sum(r['live_tokens'] for r in results):
        raise ValueError('Provider accounting differs')
    if any(e['model']!='gpt-4o-mini' for e in events):
        raise ValueError('Provider model changed')
    summary = dict(tails=len(results), verified_tails=len(results), live_calls=len(events),
                   live_tokens=sum(r['live_tokens'] for r in results),
                   statuses=dict(Counter(e['status'] for e in events)),
                   internal_error_calls=[c for r in results for c in r['diagnostics']['answer_calls'] if c['status']=='error'],
                   results_sha256=sha(args.output/'results.jsonl'), plan_sha256=sha(args.output/'plan.json'),
                   groups={str(enabled):dict(readers=sum(r['enabled']==enabled for r in results),
                       score=sum(r['score'] for r in results if r['enabled']==enabled),
                       predictions={qid:[r['prediction'] for r in results if r['enabled']==enabled and r['qa_id']==qid]
                                    for qid in cases}) for enabled in (False,True)},
                   limitation=plan['limitation'])
    (args.output/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    (args.output/'exit-code').write_text('0\n')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--repeat', type=int, choices=(1,2,3), default=2)
    asyncio.run(main(p.parse_args()))
