"""Offline exact-response replay of assistant-only supplementary exchanges.

Only completed retries whose final user-source filter kept no facts qualify.
No network calls or synthetic semantic verdicts are permitted.
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
from app import add_pipeline, schemas


async def replay(row, trace):
    messages = []
    for source in row['source_messages']:
        fields = {k: v for k, v in source.items() if k in schemas.Message.model_fields}
        if isinstance(fields.get('trust_scope'), str):
            fields['trust_scope'] = json.loads(fields['trust_scope'])
        messages.append(schemas.Message(**fields))
    req = schemas.AddRequest(request_id=row['request_id'], user_id=row['user_id'],
                             session_id=row['session_id'], messages=messages)
    events = trace['events']
    prompt = next(e['prompt'] for e in events if e['stage'] == 'extract_request')
    response = next(e['response'] for e in events if e['stage'] == 'extract_response')
    summary = prompt.split('Conversation summary so far (global context, do not extract from it alone):\n', 1)[1].split('\n\nRecent messages (local context):', 1)[0]
    calls = []

    async def frozen(actual, *args, **kwargs):
        if actual != prompt or calls:
            raise AssertionError('Frozen prompt mismatch or unexpected semantic call')
        calls.append(kwargs['stage'])
        return copy.deepcopy(response)

    # _extract_segment_impl catches provider exceptions, so also assert the
    # provider call count and capture any mismatch outside that fallback.
    violations = []
    async def checked(*args, **kwargs):
        try:
            return await frozen(*args, **kwargs)
        except AssertionError as exc:
            violations.append(str(exc))
            raise

    with patch('socket.socket.connect', side_effect=AssertionError('Network forbidden')), \
            patch.object(add_pipeline.llm, 'complete_json', checked):
        result = await add_pipeline._extract_coverage_segment(
            req, summary, trace['segment_index'], trace['message_indices'])
    if violations or len(calls) != 1 or any(f.get('type') != 'episode' for f in result):
        raise AssertionError(dict(violations=violations, calls=calls))
    return dict(request_id=req.request_id, indices=trace['message_indices'],
                prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                replay_extraction_calls=len(calls), same_accepted_user_facts=True,
                archived_semantic_requests=sum(e['stage'] == 'semantic_request' for e in events),
                archived_semantic_responses=sum(e['stage'] == 'semantic_response' for e in events))


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--memory-log', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    results = []
    with args.memory_log.open() as stream:
        for line in stream:
            row = json.loads(line)
            roles = {m['message_index']: m['role'] for m in row['source_messages']}
            for trace in row.get('extraction_trace', []):
                events = trace['events']
                final = next((e for e in events if e['stage'] == 'coverage_output'), None)
                grounded = [e['fact'] for e in events if e['stage'] == 'grounding_accepted']
                if final is None or final['facts'] or not grounded:
                    continue
                if any(roles.get(i) == 'user' for f in grounded for i in f['_sources']):
                    continue
                results.append(await replay(row, trace))
    report = dict(source=str(args.memory_log), cases=results, count=len(results),
                  semantic_requests_avoided=sum(r['archived_semantic_requests'] for r in results),
                  caveat='Completed assistant-only retry subset; exact frozen extraction responses, '
                         'no network calls. Semantic requests are not necessarily completed billed calls. '
                         'Does not predict end-to-end score or total budget reallocation.')
    with args.output.open('x') as output:
        json.dump(report, output, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != 'cases'}))


if __name__ == '__main__':
    asyncio.run(main())
