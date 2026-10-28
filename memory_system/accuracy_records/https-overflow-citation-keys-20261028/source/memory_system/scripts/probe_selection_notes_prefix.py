"""Freeze a verified reader prefix to isolate the source-note selector change."""
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
from replay_query_context import records, recorded_response


async def main(args):
    if config.FAKE:
        raise ValueError('Real provider required')
    archived = next(r for r in records(args.replay)
                    if r['qa_id'] == args.question and not r['enabled'])
    baseline = next(r for r in records(args.run / 'completed/data/results/personamem-v2-32k.jsonl')
                    if r['qa_id'] == args.question and r['conversation_id'] == archived['conversation_id'])
    trace = next(t for t in records(args.run / 'server-search-debug.jsonl')
                 if t['search_id'] == baseline['search_id'])
    packet = trace['returned']
    if evidence_packet.digest(packet) != archived['packet_hash']:
        raise ValueError('Packet changed')
    qa = next(q for d in records(args.prepared) if d['conversation_id'] == archived['conversation_id']
              for q in d['qa'] if q['id'] == args.question)
    calls = archived['diagnostics']['answer_calls']
    boundary = (next(i for i, c in enumerate(calls) if c['stage'] == 'eval.choice_support') + 1
                if args.boundary == 'after_support' else
                next(i for i, c in enumerate(calls) if c['stage'] == 'eval.choice_select'))
    settings = {k: v for k, v in archived['versions']['settings'].items() if hasattr(config, k)}
    original = llm.complete_json

    async def run(enabled, frozen=False):
        pending = copy.deepcopy(calls if frozen else calls[:boundary])
        live, violations = [], []

        async def provider(prompt, *positional, **kwargs):
            digest = hashlib.sha256(prompt.encode()).hexdigest()
            if pending:
                old = pending.pop(0)
                if old['stage'] != kwargs['stage'] or old['prompt_sha256'] != digest:
                    violations.append(kwargs['stage'])
                    raise ValueError('Frozen prefix prompt mismatch')
                budget.current.get().before_call()
                return recorded_response(old)
            if frozen:
                violations.append('unexpected_live_call')
                raise ValueError('Network forbidden during reproduction')
            call = dict(stage=kwargs['stage'], prompt=prompt, prompt_sha256=digest)
            live.append(call)
            response = await original(prompt, *positional, **kwargs)
            call['response'] = response
            return response

        with patch.multiple(config, **dict(settings, **{args.setting: enabled})), \
                patch.object(llm, 'complete_json', provider):
            prediction, score, diagnostics = await eval_scoring.evaluate(qa, copy.deepcopy(packet))
        if pending or violations:
            raise ValueError('Frozen trajectory mismatch')
        return dict(qa_id=args.question, enabled=enabled, prediction=prediction, score=score,
                    diagnostics=diagnostics, live_calls=live, packet_hash=archived['packet_hash'])

    reproduced = await run(False, frozen=True)
    if (reproduced['prediction'], reproduced['score']) != (archived['prediction'], archived['score']):
        raise ValueError('Frozen outcome mismatch')
    args.output.with_suffix('.plan.json').write_text(json.dumps(dict(
        qa_id=args.question, run=str(args.run), replay=str(args.replay),
        frozen_verified=True, prefix_calls=boundary,
        prefix_sha256=hashlib.sha256(json.dumps(calls[:boundary], sort_keys=True).encode()).hexdigest(),
        setting=args.setting, boundary=args.boundary,
        caveat='Selected mechanism diagnostic; no cohort accuracy claim. Frozen calls consume call counts but not original elapsed time/tokens.'), indent=2)+'\n')
    with args.output.open('x') as stream:
        for repetition in range(args.repeat):
            for enabled in ([False, True] if repetition % 2 == 0 else [True, False]):
                row = await run(enabled)
                row['repetition'] = repetition + 1
                stream.write(json.dumps(row, ensure_ascii=False)+'\n')
                stream.flush()
                print(json.dumps({k: row[k] for k in ['qa_id', 'enabled', 'repetition', 'prediction', 'score']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['run', 'replay', 'prepared', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--question', required=True)
    parser.add_argument('--setting', choices=['CHOICE_SOURCE_NOTES_SELECTION', 'CHOICE_BOUNDED_SUPPORT_SPANS'],
                        default='CHOICE_SOURCE_NOTES_SELECTION')
    parser.add_argument('--boundary', choices=['selection', 'after_support'], default='selection')
    parser.add_argument('--repeat', type=int, choices=[1, 2, 3], default=2)
    asyncio.run(main(parser.parse_args()))
