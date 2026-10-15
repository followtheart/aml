"""Verify both runtime flag states reproduce every archived no-recovery case.

This is a network-free trajectory check, not a new accuracy measurement.
"""
import argparse
import asyncio
import copy
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

from replay_query_context import records, recorded_response
from replay_source_vector_route import source_route, validate_probe_granularity
from app import budget, config, eval_scoring, evidence_packet, llm


async def main(args):
    probe = json.loads(args.probe.read_text())
    validate_probe_granularity(probe)
    questions = {q['qa_id']: q for q in probe['questions']}
    traces = {t['search_id']: t for t in records(args.run / 'server-search-debug.jsonl')}
    prepared = records(args.prepared)
    if len(prepared) != 1:
        raise ValueError('This diagnostic requires a single-conversation input')
    qa = {q['id']: q for q in prepared[0]['qa']}
    rows = [r for r in records(args.replay) if r['arm'] == 'window']
    if {r['qa_id'] for r in rows} != set(qa) or len(rows) != len(qa):
        raise ValueError('Require one archived window result per prepared question')
    selected = [r for r in rows if not any(c['stage'] == 'eval.choice_citation_recovery'
                                          for c in r['diagnostics']['answer_calls'])]
    with args.output.open('x') as output:
        for archived in selected:
            question = questions[archived['qa_id']]
            trace = traces[question['search_id']]
            packet, _ = source_route(probe, question, trace['returned'], trace['evidence_token_budget'])
            if evidence_packet.digest(packet) != archived['packet_hash']:
                raise ValueError('Packet changed')
            for enabled in (False, True):
                pending = copy.deepcopy(archived['diagnostics']['answer_calls'])
                violations = []

                async def frozen(prompt, *positional, **kwargs):
                    if not pending:
                        violations.append('unexpected_call')
                        raise ValueError('Network forbidden: unexpected provider call')
                    old = pending.pop(0)
                    if (old['stage'] != kwargs['stage'] or
                            old['prompt_sha256'] != hashlib.sha256(prompt.encode()).hexdigest()):
                        violations.append('changed_prompt')
                        raise ValueError('Frozen trajectory changed')
                    budget.current.get().before_call()
                    return recorded_response(old)

                with patch.object(config, 'CHOICE_QUERY_CONTEXT', False), \
                     patch.object(config, 'CHOICE_RECOVERY_CLAIM_ONLY', enabled), \
                     patch.object(llm, 'complete_json', frozen):
                    prediction, score, _ = await eval_scoring.evaluate(qa[archived['qa_id']], copy.deepcopy(packet))
                if pending or violations or (prediction, score) != (archived['prediction'], archived['score']):
                    raise ValueError('Archived no-op case failed exact trajectory verification')
                result = dict(qa_id=archived['qa_id'], enabled=enabled, prediction=prediction,
                              score=score, frozen_verified=True, network_calls=0,
                              caveat='Archived trajectory reproduction, not a live score.')
                output.write(json.dumps(result) + '\n')
                output.flush()
                print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'probe', 'replay', 'prepared', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    asyncio.run(main(parser.parse_args()))
