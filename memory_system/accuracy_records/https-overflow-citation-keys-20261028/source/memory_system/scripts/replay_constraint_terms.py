"""Paired diagnostic of constraint content terms; production gate unchanged."""
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
from replay_query_context import records, recorded_response

CONTENT_STOP = answer_choice._CONSTRAINT_CONTENT_STOP


async def main(args):
    if config.FAKE:
        raise ValueError('Live downstream comparison required')
    questions = {(d['conversation_id'], q['id']): q
                 for d in records(args.prepared) for q in d['qa']}
    results = records(args.results)
    changes = {}
    for result in results:
        qa = questions[result['conversation_id'], result['qa_id']]
        constraints = {c['id']: c for c in result['answer_constraint_catalog']}
        pairs = []
        for enabled in (False, True):
            with patch.object(config, 'CHOICE_CONSTRAINT_CONTENT_TERMS', enabled):
                pairs.append({p['pair_id'] for p in answer_choice.constraint_pairs(qa['options'], constraints)})
        changes[(result['conversation_id'], result['qa_id'])] = dict(
            added=sorted(pairs[1] - pairs[0]), removed=sorted(pairs[0] - pairs[1]))
    if args.questions:
        results = [r for r in results if r['qa_id'] in args.questions.split(',')]
    else:
        # Apply the same selection rule to new conversations without consulting
        # scores or expected answers. Unchanged candidate sets need no live calls.
        results = [r for r in results if any(changes[r['conversation_id'], r['qa_id']].values())]
    print(dict(selected_questions=[r['qa_id'] for r in results],
               selection='explicit' if args.questions else 'changed_constraint_pairs'), flush=True)
    wanted = {r['search_id'] for r in results}
    searches = {}
    with args.search_log.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row.get('search_id') in wanted:
                searches[row['search_id']] = row
    original_provider = llm.complete_json
    with args.output.open('x') as output:
        for rep in range(args.repeat):
            for result in results:
                calls = result['answer_calls']
                # Unique-eligible or abstaining archived answers can finish
                # without constraints/selection; any new calls remain live.
                boundary = next((i for i, c in enumerate(calls)
                    if c['stage'] in ('eval.choice_constraints', 'eval.choice_select')), len(calls))
                for enabled in ([False, True] if rep % 2 == 0 else [True, False]):
                    pending = copy.deepcopy(calls[:boundary])
                    violations, live = [], []

                    async def provider(prompt, *positional, **kwargs):
                        if pending:
                            archived = pending.pop(0)
                            if (kwargs['stage'] != archived['stage'] or
                                hashlib.sha256(prompt.encode()).hexdigest() != archived['prompt_sha256']):
                                violations.append(kwargs['stage'])
                                raise ValueError('Frozen prompt mismatch')
                            if budget.current.get():
                                budget.current.get().before_call()
                            return recorded_response(archived)
                        call = dict(stage=kwargs['stage'], prompt=prompt)
                        live.append(call)
                        response = await original_provider(prompt, *positional, **kwargs)
                        call['response'] = response
                        return response

                    search = searches[result['search_id']]
                    if result['packet_hash'] != search['packet_hash']:
                        raise ValueError('Packet mismatch')
                    with patch.object(config, 'CHOICE_CONSTRAINT_CONTENT_TERMS', enabled), \
                            patch.object(llm, 'complete_json', provider):
                        prediction, score, diagnostics = await eval_scoring.evaluate(
                            questions[result['conversation_id'], result['qa_id']],
                            copy.deepcopy(search['returned']))
                    if pending or violations:
                        raise ValueError(dict(pending=len(pending), violations=violations))
                    row = dict(qa_id=result['qa_id'], repetition=rep + 1, enabled=enabled,
                        prediction=prediction, score=score, diagnostics=diagnostics, live_calls=live,
                        packet_hash=result['packet_hash'], content_stop=sorted(CONTENT_STOP),
                        candidate_pair_changes=changes[result['conversation_id'], result['qa_id']],
                        caveat='Frozen upstream prompts/responses; constraints and selection live. '
                               'Upstream elapsed time/tokens not reproduced. No production change.')
                    output.write(json.dumps(row, ensure_ascii=False) + '\n')
                    output.flush()
                    print({k: row[k] for k in ('qa_id', 'repetition', 'enabled',
                                             'prediction', 'score')}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('results', 'search-log', 'prepared', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--questions', help='Optional IDs; default selects all changed candidate sets, without scores')
    parser.add_argument('--repeat', type=int, choices=[1, 2, 3], default=2)
    asyncio.run(main(parser.parse_args()))
