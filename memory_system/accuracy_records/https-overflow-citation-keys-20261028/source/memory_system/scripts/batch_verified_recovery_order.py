"""Compare semantically verified early recovery on remaining fixed-prefix cases.

Selection uses stage availability, never scores or answer labels. Each case first
must reproduce its complete archived trajectory without any network calls.
"""
import argparse
import asyncio
import copy
import hashlib
import json
from pathlib import Path

from probe_verified_recovery_order import main as probe
from replay_query_context import records


async def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    # Read the live writer's file once, retaining only complete JSONL records.
    raw = args.replay.read_bytes()
    complete = raw[:raw.rfind(b'\n') + 1]
    snapshot = args.output / 'reader-snapshot.jsonl'
    snapshot.write_bytes(complete)
    selected = [r['qa_id'] for r in records(snapshot)
                if r['arm'] == 'window' and r['qa_id'] not in args.exclude
                and any(c['stage'] == 'eval.choice_premise_scope'
                        for c in r['diagnostics']['answer_calls'])]
    if not selected or len(selected) != len(set(selected)):
        raise ValueError('Need unique eligible question IDs')
    plan = dict(questions=selected, excluded=args.exclude,
                intervention='verified_recovery_before_scope_diagnostic',
                snapshot_sha256=hashlib.sha256(complete).hexdigest(),
                selection='All complete window-arm records with premise scope; no score filter.',
                repetitions=args.repetitions,
                caveat='Availability-selected fixed-prefix diagnostic, not a full HTTPS benchmark.')
    (args.output / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    results = []
    for index, question in enumerate(selected):
        case = copy.copy(args)
        case.question = question
        case.replay = snapshot
        case.order_offset = index % 2
        case.output = args.output / f'{question}.jsonl'
        await probe(case)
        results.extend(records(case.output))
        summary = dict(completed_questions=index + 1, planned_questions=len(selected),
                       arms={name: dict(count=sum(r['enabled'] == enabled for r in results),
                                       score=sum(r['score'] for r in results if r['enabled'] == enabled))
                             for name, enabled in [('control', False), ('recovery_first', True)]})
        (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        print(json.dumps(dict(question=question, **summary)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'probe', 'replay', 'prepared', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--exclude', nargs='*', default=[])
    parser.add_argument('--repetitions', type=int, default=1)
    asyncio.run(main(parser.parse_args()))
