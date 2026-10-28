"""Compare grammatical guidance on exactly reconstructed support repair prompts.

All original calls are first replayed with stage/hash checks and no network.
Only the captured repair requests are then sent live. This does not measure
answer accuracy or change production behavior.
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
from replay_query_context import records, recorded_response


async def main(args):
    if config.FAKE:
        raise ValueError('Real provider required')
    results = [r for r in records(args.results) if r['qa_id'] in args.questions.split(',')]
    if not results:
        raise ValueError('No selected results')
    wanted = {r['search_id'] for r in results}
    searches = {}
    with args.search_log.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row.get('search_id') in wanted:
                searches[row['search_id']] = row
    questions = {(d['conversation_id'], q['id']): q
                 for d in records(args.prepared) for q in d['qa']}
    captured = []
    provider = llm.complete_json
    for result in results:
        pending = copy.deepcopy(result['answer_calls'])
        violations = []

        async def frozen(prompt, *positional, **kwargs):
            call = pending.pop(0) if pending else {}
            if (call.get('stage') != kwargs['stage'] or call.get('prompt_sha256') !=
                    hashlib.sha256(prompt.encode()).hexdigest()):
                violations.append(kwargs['stage'])
                raise ValueError('Frozen prompt mismatch')
            if budget.current.get():
                budget.current.get().before_call()
            if kwargs['stage'] == 'eval.choice_support.repair':
                captured.append(dict(qa_id=result['qa_id'], prompt=prompt,
                                     kwargs=copy.deepcopy(kwargs)))
            return recorded_response(call)

        search = searches[result['search_id']]
        if search['packet_hash'] != result['packet_hash']:
            raise ValueError('Packet mismatch')
        with patch.object(llm, 'complete_json', frozen):
            prediction, score, _ = await eval_scoring.evaluate(
                questions[result['conversation_id'], result['qa_id']],
                copy.deepcopy(search['returned']))
        if pending or violations or prediction != result['prediction'] or score != result['score']:
            raise ValueError('Frozen answer reproduction failed')
    with args.output.open('x') as output:
        for repetition in range(2):
            for index, request in enumerate(captured):
                for enabled in ([False, True] if repetition == 0 else [True, False]):
                    prompt = request['prompt']
                    if enabled:
                        prompt += '\n' + answer_choice.VERIFICATION_SCOPE_GUIDANCE
                    kwargs = dict(request['kwargs'], stage='diagnostic.repair_verification_scope',
                                  attempts=1, timeout=60)
                    with budget.scope(seconds=65, calls=1, tokens=24000):
                        response = await provider(prompt, **kwargs)
                    row = dict(qa_id=request['qa_id'], request_index=index,
                               repetition=repetition + 1, enabled=enabled,
                               prompt=prompt, response=response,
                               caveat='Identical archived repair request except appended guidance; '
                                      'classification probe, not validated downstream accuracy.')
                    output.write(json.dumps(row, ensure_ascii=False) + '\n')
                    output.flush()
                    print(dict(qa_id=row['qa_id'], request=index, repetition=repetition + 1,
                               enabled=enabled), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('results', 'search-log', 'prepared', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--questions', required=True)
    asyncio.run(main(parser.parse_args()))
