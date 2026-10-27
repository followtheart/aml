"""Replay only supplementary extraction against archived validated outputs.

No database writes and no answer options/gold supplied to extraction.
"""
import argparse
import asyncio
import copy
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import add_pipeline, config, extraction_coverage, memory_debug, persona_source, profile, schemas


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--memory-log', required=True, type=Path)
    parser.add_argument('--request-id', required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--deterministic-coverage', action='store_true',
                        help='Account for persona and forget-rule extraction before retry')
    parser.add_argument('--preverify-user-source', action='store_true',
                        help='Reject assistant-only supplementary facts before semantic calls')
    args = parser.parse_args()
    row = next(json.loads(line) for line in args.memory_log.open()
               if json.loads(line).get('request_id') == args.request_id)
    segments, groups = [], []
    for trace in row['extraction_trace']:
        output = next((e for e in trace['events'] if e['stage'] == 'segment_output'), None)
        if output is None:
            raise ValueError('Archived segment has no validated output')
        if trace['segment_index'] != len(segments):
            raise ValueError('Expected original ordered segments, not retry traces')
        segments.append(trace['message_indices'])
        groups.append(copy.deepcopy(output['facts']))
    original = copy.deepcopy(groups)
    prompt = next(e['prompt'] for e in row['extraction_trace'][0]['events']
                  if e['stage'] == 'extract_request')
    summary = prompt.split('Conversation summary so far (global context, do not extract from it alone):\n', 1)[1].split('\n\nRecent messages (local context):', 1)[0]
    messages = []
    for item in row['source_messages']:
        fields = {k: v for k, v in item.items() if k in schemas.Message.model_fields}
        if isinstance(fields.get('trust_scope'), str):
            fields['trust_scope'] = json.loads(fields['trust_scope'])
        messages.append(schemas.Message(**fields))
    req = schemas.AddRequest(request_id=row['request_id'], user_id=row['user_id'],
                             session_id=row['session_id'], messages=messages)
    if args.deterministic_coverage:
        facts = [f for group in groups for f in group]
        facts = profile.ensure_forget_rules(req, facts, segments)
        facts = persona_source.persona_facts(req, facts, segments)
        groups = [[f for f in facts if f.get('_segment') == i] for i in range(len(segments))]
        original = copy.deepcopy(groups)
    candidates = list(extraction_coverage.uncovered_exchanges(req, segments, groups))
    # Reserve artifact before any billable work; don't overwrite earlier probes.
    with args.output.open('x') as output:
        started = time.monotonic()
        config.MEMORY_DEBUG_LOG = config.MEMORY_DEBUG_LOG or 'diagnostic-capture-only'
        with memory_debug.extraction_scope():
            await extraction_coverage.supplement(req, summary, segments, groups,
                add_pipeline._extract_coverage_segment if args.preverify_user_source
                else add_pipeline._extract_segment_impl)
            trace = copy.deepcopy(memory_debug._extraction.get())
        added = [fact for before, after in zip(original, groups) for fact in after[len(before):]]
        json.dump(dict(source_request=args.request_id, source_log=str(args.memory_log),
                       candidates=candidates, added_facts=added, extraction_trace=trace,
                       elapsed_seconds=time.monotonic() - started,
                       deterministic_coverage=args.deterministic_coverage,
                       preverify_user_source=args.preverify_user_source,
                       scope='Frozen original extraction; live supplementary calls; no DB writes'),
                  output, ensure_ascii=False, indent=2)
        print(json.dumps(dict(candidates=candidates, added_facts=len(added),
                              output=str(args.output))))


if __name__ == '__main__':
    asyncio.run(main())
