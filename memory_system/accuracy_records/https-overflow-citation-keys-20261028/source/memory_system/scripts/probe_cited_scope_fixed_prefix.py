"""Hold archived pre-scope reader calls fixed; run scope and later calls live.

Selected diagnostic, not fresh HTTPS or a cohort accuracy estimate.
"""
import argparse
import asyncio
import copy
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

from replay_query_context import recorded_response
from summarize_reader_replay import read_complete
from app import budget, config, eval_scoring, evidence_packet, llm


async def main(args):
    results = {r['qa_id']: r for r in read_complete(args.run/'completed/data/results/personamem-v2-32k.jsonl')}
    traces = {r['search_id']: r for r in read_complete(args.run/'server-search-debug.jsonl')}
    questions = {(d['conversation_id'], q['id']): q for d in read_complete(args.prepared) for q in d['qa']}
    cases = []
    for qid in args.questions.split(','):
        row = results[qid]
        packet = traces[row['search_id']]['returned']
        if evidence_packet.digest(packet) != row['packet_hash']:
            raise ValueError('Packet identity mismatch')
        calls = row['answer_calls']
        stop = next(i for i, c in enumerate(calls) if c['stage'] == 'eval.choice_premise_scope')
        prefix = calls[:stop]
        if any(c.get('status') == 'error' for c in prefix):
            raise ValueError('Cannot reconstruct provider errors in fixed prefix')
        cases.append((row, packet, prefix))
    live_provider = llm.complete_json
    with args.output.open('x') as output:
        output.write(json.dumps(dict(event='plan', questions=args.questions, run=str(args.run),
            scope='Archived pre-scope responses; live scope and downstream calls; no DB writes',
            caveat='Selected diagnostic; replay accounts for call count but not original token use or latency.')) + '\n')
        output.flush()
        for case_index, (row, packet, prefix) in enumerate(cases):
            for enabled in ([False, True] if case_index % 2 == 0 else [True, False]):
                pending = copy.deepcopy(prefix)
                mismatches = []
                async def provider(prompt, *positional, **kwargs):
                    if pending:
                        old = pending.pop(0)
                        if old['stage'] != kwargs['stage'] or old['prompt_sha256'] != hashlib.sha256(prompt.encode()).hexdigest():
                            mismatches.append(kwargs['stage'])
                            raise ValueError('Archived prefix prompt mismatch')
                        budget.current.get().before_call()
                        return recorded_response(old)
                    if mismatches:
                        raise ValueError('Prefix mismatch prohibits live continuation')
                    return await live_provider(prompt, *positional, **kwargs)
                with patch.object(config, 'CHOICE_CITED_SCOPE', enabled), patch.object(llm, 'complete_json', provider):
                    prediction, score, diagnostics = await eval_scoring.evaluate(
                        questions[row['conversation_id'], row['qa_id']], copy.deepcopy(packet))
                if pending or mismatches:
                    raise ValueError('Prefix reconstruction failed')
                result = dict(event='result', qa_id=row['qa_id'], enabled=enabled,
                    prediction=prediction, score=score, packet_hash=row['packet_hash'],
                    fixed_prefix_calls=len(prefix), diagnostics=diagnostics)
                output.write(json.dumps(result, ensure_ascii=False) + '\n')
                output.flush()
                print(json.dumps({k: result[k] for k in ['qa_id', 'enabled', 'prediction', 'score']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--questions', required=True)
    parser.add_argument('--output', type=Path, required=True)
    asyncio.run(main(parser.parse_args()))
