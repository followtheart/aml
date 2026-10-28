"""Live extraction coverage probe with attribution and non-adoption controls.

No database writes. Provider inputs contain source messages, not QA metadata.
"""
import argparse
import asyncio
import copy
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import add_pipeline, budget, config, memory_debug, schemas

NOTE = """
Before emitting facts, check each NEW user-supplied passage for incidental
personal details as well as its main business or editing topic. Preserve each
explicit habit, routine, preference, health state, or circumstance as its own
atomic fact, even when it occurs in a subordinate clause explaining another
arrangement. A fact about the main topic does not cover the other propositions
in the same message. This coverage check does not establish user identity:
retain the document narrator and uncertainty under the attribution rules above.
Do not convert proposed, fictional, conditional, negated, or assistant-invented
details into actual user attributes. Do not infer enthusiasm from an activity.
Every additional fact must have its own exact quote and supported triples.
"""


async def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--memory-log', type=Path, required=True)
    p.add_argument('--request-id', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    row = next(json.loads(line) for line in args.memory_log.open()
               if json.loads(line).get('request_id') == args.request_id)
    trace = next(t for t in row['extraction_trace'] if t['segment_index'] == 0)
    prompt = next(e['prompt'] for e in trace['events'] if e['stage'] == 'extract_request')
    summary = prompt.split('Conversation summary so far (global context, do not extract from it alone):\n', 1)[1].split('\n\nRecent messages (local context):', 1)[0]
    by_index = {s['message_index']: s for s in row['source_messages']}
    cases = [dict(name='archived_segment', summary=summary,
                  messages=[dict(role=by_index[i]['role'], content=by_index[i]['content'])
                            for i in trace['message_indices']])]
    controls = [
        ('self_authored', [('user', 'Please polish this email I wrote: Dear Morgan, I can call after nine because I usually begin my day tending my herb garden. Regards, Avery.')]),
        ('third_party', [('user', 'My colleague Avery wrote this email. Please polish it: Dear Morgan, I can call after nine because I usually begin my day tending my herb garden. Regards, Avery.')]),
        ('fictional', [('user', 'I am writing a fictional character named Avery. Please polish this invented email: Dear Morgan, I can call after nine because I usually begin my day tending my herb garden. Regards, Avery.')]),
        ('assistant_only', [('user', 'Can you draft a professional email proposing a call after nine?'), ('assistant', 'Dear Morgan, I can call after nine because I usually begin my day tending my herb garden. Regards, Avery.')]),
        ('conditional', [('user', 'Please polish this email I wrote: Dear Morgan, if I take up gardening, I might arrange calls after nine. I do not currently have a garden. Regards, Avery.')]),
    ]
    cases.extend(dict(name=name, summary='The conversation user is Jordan.',
                      messages=[dict(role=role, content=text) for role, text in msgs])
                 for name, msgs in controls)
    original_note = add_pipeline._DOCUMENT_ATTRIBUTION_NOTE
    config.MEMORY_DEBUG_LOG = 'diagnostic-capture-only'
    with args.output.open('x') as output:
        output.write(json.dumps(dict(event='plan', cases=cases, note=NOTE,
            repeat=2, fake=config.FAKE, source_log=str(args.memory_log),
            source_request=args.request_id, database_writes=False)) + '\n')
        output.flush()
        try:
            for repeat in range(2):
                for case in cases:
                    for arm in (['baseline', 'coverage'] if repeat == 0 else ['coverage', 'baseline']):
                        add_pipeline._DOCUMENT_ATTRIBUTION_NOTE = original_note + (NOTE if arm == 'coverage' else '')
                        req = schemas.AddRequest(request_id='diagnostic', user_id='diagnostic',
                            session_id='diagnostic', messages=[schemas.Message(**m) for m in case['messages']])
                        started = time.monotonic()
                        with budget.scope(seconds=240, calls=30, tokens=240000) as limits:
                            with memory_debug.extraction_scope():
                                facts = await add_pipeline._extract_segment(req, case['summary'], 0,
                                    list(range(len(req.messages))))
                                captured = copy.deepcopy(memory_debug._extraction.get())
                        result = dict(event='result', repeat=repeat, case=case['name'], arm=arm,
                            facts=facts, extraction_trace=captured, calls=limits.calls,
                            elapsed_seconds=time.monotonic() - started)
                        output.write(json.dumps(result, ensure_ascii=False) + '\n')
                        output.flush()
                        print(json.dumps({k: result[k] for k in ['repeat', 'case', 'arm', 'calls']}), flush=True)
        finally:
            add_pipeline._DOCUMENT_ATTRIBUTION_NOTE = original_note


if __name__ == '__main__':
    asyncio.run(main())
