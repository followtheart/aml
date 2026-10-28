"""Frozen upstream/full-reader diagnostic for subject/modality scope decisions."""
import argparse
import asyncio
import copy
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import budget, config, eval_scoring, evidence_packet, llm
from probe_scope_roles import RoleScopes
from replay_query_context import recorded_response


async def main(args):
    if config.FAKE:
        raise ValueError('Live downstream provider required')
    archived = next(r for r in map(json.loads, args.replay.read_text().splitlines())
                    if r['qa_id'] == args.question and r['shared_sources'])
    qa = next(q for d in map(json.loads, args.prepared.read_text().splitlines())
              for q in d['qa'] if q['id'] == args.question)
    template = next(r for r in map(json.loads, args.scope_template.read_text().splitlines())
                    if r['structured'] and r['valid'])
    role_prefix = template['prompt'].split('\n', 1)[0]
    calls = archived['diagnostics']['answer_calls']
    boundary = next(i for i, c in enumerate(calls) if c['stage'] == 'eval.choice_premise_scope')
    if evidence_packet.digest(archived['packet']) != archived['packet_hash']:
        raise ValueError('Packet identity mismatch')
    original = llm.complete_json

    async def run(enabled, frozen=False):
        pending = copy.deepcopy(calls if frozen else calls[:boundary])
        live, mismatches = [], []

        async def provider(prompt, *positional, **kwargs):
            if pending:
                old = pending.pop(0)
                if old['stage'] != kwargs['stage'] or hashlib.sha256(prompt.encode()).hexdigest() != old['prompt_sha256']:
                    mismatches.append(kwargs['stage'])
                    raise ValueError('Frozen upstream prompt mismatch')
                budget.current.get().before_call()
                return recorded_response(old)
            if frozen:
                mismatches.append('unexpected_extra_call')
                raise ValueError('Unexpected provider call in frozen validation')
            role_call = enabled and kwargs['stage'].startswith('eval.choice_premise_scope')
            actual = prompt
            if role_call:
                actual = role_prefix + '\n' + prompt.split('\n', 1)[1]
                kwargs['schema'] = RoleScopes.model_json_schema()
            record = dict(stage=kwargs['stage'], prompt=actual, original_prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                          role_call=role_call)
            live.append(record)
            response = await original(actual, *positional, **kwargs)
            record['response'] = response
            if role_call:
                parsed = RoleScopes.model_validate(response)
                response = dict(claims=[dict(claim_id=c.claim_id,
                    personal_fact=c.subject != 'generic_audience_or_other'
                    and c.modality != 'proposed_or_conditional') for c in parsed.claims])
                record['converted_response'] = response
            return response

        with patch.object(llm, 'complete_json', provider), budget.scope(seconds=240, calls=20, tokens=128000) as usage:
            prediction, score, diagnostics = await eval_scoring.evaluate(qa, copy.deepcopy(archived['packet']))
        if pending or mismatches:
            raise ValueError(dict(pending=len(pending), mismatches=mismatches))
        return dict(qa_id=args.question, enabled=enabled, prediction=prediction, score=score,
                    packet_hash=archived['packet_hash'], diagnostics=diagnostics, live_calls=live,
                    calls=usage.calls, tokens=usage.tokens)

    checked = await run(False, frozen=True)
    if checked['prediction'] != archived['prediction'] or checked['score'] != archived['score']:
        raise ValueError('Frozen reader result mismatch')
    print(json.dumps(dict(frozen_verified=True, qa_id=args.question)), flush=True)
    with args.output.open('x') as output:
        for repetition in range(args.repeat):
            for enabled in ([False, True] if repetition % 2 == 0 else [True, False]):
                row = await run(enabled)
                row.update(repetition=repetition+1, frozen_verified=True,
                           caveat='Archived packet and calls before scope frozen, upstream time/tokens omitted. '
                                  'Live scope and downstream reader; raw role response converted to existing bool '
                                  'interface. Diagnostic only, not production budget validation or HTTPS accuracy.')
                output.write(json.dumps(row, ensure_ascii=False) + '\n')
                output.flush()
                print(json.dumps({k: row[k] for k in ('qa_id', 'repetition', 'enabled', 'prediction', 'score')}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('replay', 'prepared', 'scope-template', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--question', default='100')
    p.add_argument('--repeat', type=int, choices=(1, 2, 3), default=2)
    asyncio.run(main(p.parse_args()))
