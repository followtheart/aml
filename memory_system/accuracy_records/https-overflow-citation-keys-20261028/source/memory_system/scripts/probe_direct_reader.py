"""Fixed-packet direct-reader ablation with memory or original-source context.

This diagnostic deliberately omits intermediate witness/support/eligibility
decisions. It retains task instructions and forget constraints. No production
policy changes; archived gold labels are used only after model responses.
"""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import answer_choice, answer_context, budget, config, eval_scoring, llm
from replay_query_context import records


async def main(args):
    if config.FAKE:
        raise ValueError('Real provider required')
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
        for index, result in enumerate(results):
            qa = questions[result['conversation_id'], result['qa_id']]
            if qa['qa_type'] != 'single_choice':
                raise ValueError('Single-choice diagnostic only')
            search = searches[result['search_id']]
            if search['packet_hash'] != result['packet_hash']:
                raise ValueError('Packet identity mismatch')
            packet = search['returned']
            context = answer_context.build(packet)
            original = eval_scoring.answer_prompt(qa, packet)
            marker = '<memories>\n' + context + '\n</memories>'
            if original.count(marker) != 1:
                raise ValueError('Unexpected answer context boundary')
            options = answer_choice.option_map(qa['options'])
            schema = dict(type='object', additionalProperties=False, required=['answer'],
                          properties=dict(answer=dict(type='string', enum=list(options))))
            modes = ['memory', 'sources'] if index % 2 == 0 else ['sources', 'memory']
            for mode in modes:
                prompt = original
                if mode == 'sources':
                    sources = {s['id']: s for s in result['answer_source_catalog']}
                    source_context = json.dumps(dict(
                        original_sources=answer_choice._cards(sources),
                        forget_constraints=result['answer_constraint_catalog']), ensure_ascii=False)
                    prompt = original.replace(marker, '<memories>\n' + source_context + '\n</memories>', 1)
                prompt += '\nReturn the chosen letter in the structured answer field.'
                row = dict(qa_id=result['qa_id'], conversation_id=result['conversation_id'],
                           mode=mode, packet_hash=result['packet_hash'],
                           prompt=prompt, prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                           archived_prediction=result['prediction'], archived_score=result['score'],
                           caveat='Same archived packet; direct reader bypasses intermediate gates. '
                                  'Not a paired rerun of the gated policy or end-to-end accuracy.')
                try:
                    with budget.scope(seconds=65, calls=1, tokens=24000) as usage:
                        response = await llm.complete_json(
                            prompt, schema=schema, stage='diagnostic.direct_reader',
                            attempts=1, timeout=60, max_tokens=128)
                    if set(response) != {'answer'} or response['answer'] not in options:
                        raise ValueError('Invalid selection')
                    row.update(response=response, prediction=response['answer'],
                               score=eval_scoring.choice_score(response['answer'], qa['gold_labels'], qa['qa_type']),
                               calls=usage.calls, tokens=usage.tokens)
                except Exception as exc:
                    row.update(error_type=type(exc).__name__, error=str(exc)[:1000], score=0)
                output.write(json.dumps(row, ensure_ascii=False) + '\n')
                output.flush()
                print({k: row.get(k) for k in ('qa_id', 'mode', 'prediction', 'score', 'error_type')}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('results', 'search-log', 'prepared', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    asyncio.run(main(parser.parse_args()))
