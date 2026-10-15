"""Inventory scope-fallback candidates using exact archived reader responses.

No provider calls: the classifier collects candidates but approves nothing.
Reject a replay if any prompt, packet, call sequence or final prediction changes.
"""
import argparse
import asyncio
import copy
import hashlib
import json
from pathlib import Path
import socket
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import budget, config, eval_scoring, llm, source_scope
from replay_query_context import recorded_response, records


async def main(args):
    results = records(args.results)
    wanted = {r['search_id'] for r in results}
    searches = {}
    with args.search_log.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row.get('search_id') in wanted:
                searches[row['search_id']] = row
    questions = {(d['conversation_id'], q['id']): q
                 for d in records(args.prepared) for q in d['qa']}
    with args.output.open('x') as output:
        for result in results:
            pending = copy.deepcopy(result['answer_calls'])
            violations, candidates = [], []

            def forbidden(*args, **kwargs):
                violations.append('network attempted')
                raise RuntimeError('Offline inventory forbids network')

            async def provider(prompt, *args, **kwargs):
                archived = pending.pop(0) if pending else {}
                if (kwargs.get('stage') != archived.get('stage') or
                        hashlib.sha256(prompt.encode()).hexdigest() != archived.get('prompt_sha256')):
                    violations.append(dict(actual=kwargs.get('stage'), expected=archived.get('stage')))
                    raise ValueError('Archived prompt mismatch')
                if budget.current.get():
                    budget.current.get().before_call()
                return recorded_response(archived)

            async def collect(items, sources, diagnostics):
                for sid, quote in items:
                    candidates.append(dict(source_id=sid, quote=quote, source=sources[sid]))
                return set()

            search = searches[result['search_id']]
            if search['packet_hash'] != result['packet_hash']:
                raise ValueError('Packet identity mismatch')
            with patch.object(config, 'CHOICE_SOURCE_SCOPE', True), \
                    patch.object(source_scope, 'classify', collect), \
                    patch.object(llm, 'complete_json', provider), \
                    patch.object(socket.socket, 'connect', forbidden), \
                    patch.object(socket.socket, 'connect_ex', forbidden):
                prediction, score, diagnostics = await eval_scoring.evaluate(
                    questions[(result['conversation_id'], result['qa_id'])],
                    copy.deepcopy(search['returned']))
            if pending or violations or prediction != result['prediction']:
                raise ValueError(dict(qa_id=result['qa_id'], pending=len(pending),
                                      violations=violations, prediction=prediction,
                                      expected=result['prediction']))
            row = dict(qa_id=result['qa_id'], prediction=prediction, score=score,
                       packet_hash=result['packet_hash'], candidates=candidates,
                       frozen_calls=len(result['answer_calls']), network_calls=0,
                       caveat='No-op classifier; all archived reader prompts and final prediction reproduced. '
                              'Archived elapsed time and token use not reconstructed.')
            output.write(json.dumps(row, ensure_ascii=False) + '\n')
            output.flush()
            print(dict(qa_id=row['qa_id'], candidates=len(candidates)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('results', 'search-log', 'prepared', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    asyncio.run(main(parser.parse_args()))
