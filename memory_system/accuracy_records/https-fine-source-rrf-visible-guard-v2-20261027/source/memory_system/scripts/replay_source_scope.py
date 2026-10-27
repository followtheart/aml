"""Diagnostic replay of independently classified exact source/quote scopes.

No production gate changes. Freeze all calls preceding independent entailment;
only no_user_assertion can be repaired, and later entailment remains live.
"""
import argparse
import asyncio
import copy
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import answer_choice, budget, config, eval_scoring, llm
from replay_query_context import recorded_response


def read(path):
    return [json.loads(line) for line in path.open() if line.strip()]


def approvals(path):
    """Require affirmative self-report agreement across recorded probe batches."""
    probe = json.loads(path.read_text())
    approved = None
    for run in probe['results']:
        cases = {c['id']: c for c in json.loads(run['prompt'].split('\n', 1)[1])}
        decisions = run['response']['judgments']
        if len(decisions) != len(cases) or {d['id'] for d in decisions} != set(cases):
            raise ValueError('Invalid source-scope judgments')
        batch = {(cases[d['id']]['source'], cases[d['id']]['quote']) for d in decisions
                 if d['scope'] == 'self_report' and d['polarity'] == 'affirmative'}
        approved = batch if approved is None else approved & batch
    return approved or set()


async def main(args):
    if config.FAKE:
        raise ValueError('Live downstream verification required')
    allowed = approvals(args.approvals)
    results = [r for r in read(args.results) if r['qa_id'] in args.questions.split(',')]
    searches = {s['search_id']: s for s in read(args.search_log)
                if s['search_id'] in {r['search_id'] for r in results}}
    questions = {(d['conversation_id'], q['id']): q for d in read(args.prepared) for q in d['qa']}
    original_complete, original_scope = llm.complete_json, answer_choice._self_scope
    with args.output.open('x') as output:
        for rep in range(args.repeat):
            for result in results:
                calls = result['answer_calls']
                boundary = next(i for i, c in enumerate(calls) if c['stage'] == 'eval.choice_entailment')
                for enabled in ([False, True] if rep % 2 == 0 else [True, False]):
                    pending = copy.deepcopy(calls[:boundary])
                    mismatches, overrides, live_calls = [], [], []
                    scope_active = False

                    def scope(source, quote, claim):
                        failure = original_scope(source, quote, claim)
                        if (enabled and not args.use_classifier and scope_active
                                and failure == 'no_user_assertion' and source['role'] == 'user'
                                and quote in source['text'] and (source['text'], quote) in allowed):
                            overrides.append(dict(source_id=source['id'], quote=quote, claim=claim,
                                                  original_failure=failure))
                            return None
                        return failure

                    async def provider(prompt, *positional, **kwargs):
                        nonlocal scope_active
                        stage = kwargs['stage']
                        if pending:
                            archived = pending.pop(0)
                            if (stage != archived['stage'] or hashlib.sha256(prompt.encode()).hexdigest()
                                    != archived['prompt_sha256']):
                                mismatches.append(dict(stage=stage, expected=archived['stage']))
                                raise ValueError('Frozen upstream prompt mismatch')
                            if budget.current.get():
                                budget.current.get().before_call()
                            value = recorded_response(archived)
                            # Scope must remain original while support validation
                            # decides whether its recorded repair branch is needed.
                            # Activate only when recovered citations are checked.
                            if stage == 'eval.choice_citation_recovery':
                                scope_active = True
                            return value
                        live_calls.append(dict(stage=stage, prompt=prompt))
                        value = await original_complete(prompt, *positional, **kwargs)
                        live_calls[-1]['response'] = value
                        return value

                    search = searches[result['search_id']]
                    if search['packet_hash'] != result['packet_hash']:
                        raise ValueError('Packet identity mismatch')
                    with patch.object(answer_choice, '_self_scope', scope), \
                            patch.object(config, 'CHOICE_SOURCE_SCOPE', enabled if args.use_classifier else False), \
                            patch.object(llm, 'complete_json', provider):
                        prediction, score, diagnostics = await eval_scoring.evaluate(
                            questions[(result['conversation_id'], result['qa_id'])],
                            copy.deepcopy(search['returned']))
                    if pending or mismatches:
                        raise ValueError(dict(pending=len(pending), mismatches=mismatches))
                    row = dict(qa_id=result['qa_id'], repetition=rep + 1, enabled=enabled,
                               intervention='runtime_classifier' if args.use_classifier else 'fixed_scope_approval',
                               prediction=prediction, score=score, overrides=overrides,
                               diagnostics=diagnostics, live_calls=live_calls,
                               packet_hash=search['packet_hash'],
                               caveat='Independent exact source/quote scope approvals; all upstream '
                                      'prompts/responses frozen. Entailment and selection live. '
                                      'Upstream time/tokens not reconstructed. No production change.')
                    output.write(json.dumps(row, ensure_ascii=False) + '\n')
                    output.flush()
                    print({k: row[k] for k in ('qa_id', 'repetition', 'enabled', 'prediction', 'score')}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('results', 'search-log', 'prepared', 'approvals', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--questions', required=True)
    parser.add_argument('--repeat', type=int, choices=[1, 2, 3], default=2)
    parser.add_argument('--use-classifier', action='store_true',
                        help='Evaluate the application classifier live instead of fixed scope approvals')
    asyncio.run(main(parser.parse_args()))
