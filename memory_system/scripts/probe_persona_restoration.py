"""Paired full-reader diagnostic: restore the first original persona message.

This bypasses retrieval for diagnosis, never injects QA metadata, and does not
change production. Existing packet items and forget constraints are retained.
"""
import argparse
import asyncio
import copy
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import answer_context, budget, config, eval_scoring, evidence_packet, persona_source


def rows(path):
    with path.open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def restore(packet, message, conversation_id):
    answer_context.build(packet)
    versions = {item.get('packet_hash_version') for item in packet}
    if len(versions) != 1 or next(iter(versions)) not in (2, 3, 4):
        raise ValueError('Requires a provenance-signed packet')
    source = dict(content=message['content'], role=message['role'],
                  timestamp=message.get('timestamp'), message_index=0,
                  request_id=f'diagnostic-original:{conversation_id}:session:0',
                  source_role='persona')
    restored = copy.deepcopy(packet)
    restored.append(dict(id='diagnostic_original_persona', memory_type='fact',
                         content=answer_context.with_evidence('', [source]),
                         sources=[source], packet_hash_version=next(iter(versions))))
    packet_hash = evidence_packet.digest(restored)
    for item in restored:
        item['packet_hash'] = packet_hash
    answer_context.build(restored)
    return restored


async def main(args):
    if config.FAKE:
        raise ValueError('Real provider required')
    config.MEMORY_DEBUG_LOG = config.SEARCH_DEBUG_LOG = ''
    config.CHOICE_WITNESS_SPAN_REFS = args.witness_span_refs
    prepared = {d['conversation_id']: d for d in rows(args.prepared)}
    cases = []
    for specification in args.case:
        directory, qid = specification.rsplit(':', 1)
        run = Path(directory)
        matches = [r for r in rows(run / 'completed/data/results/personamem-v2-32k.jsonl')
                   if r['qa_id'] == qid]
        if len(matches) != 1:
            raise ValueError('Case must identify exactly one result')
        result = matches[0]
        searches = [s for s in rows(run / 'server-search-debug.jsonl')
                    if s.get('search_id') == result['search_id']]
        if len(searches) != 1 or searches[0]['packet_hash'] != result['packet_hash']:
            raise ValueError('Search identity mismatch')
        document = prepared[result['conversation_id']]
        message = document['sessions'][0][0]
        if message['role'] != 'user' or not persona_source.is_persona_message(message['content']):
            raise ValueError('First original message must be an explicit persona declaration')
        qa = next(q for q in document['qa'] if q['id'] == qid)
        original = searches[0]['returned']
        restored = restore(original, message, result['conversation_id'])
        cases.append((specification, result, qa, message, original, restored))
    if args.inspect:
        print(json.dumps(dict(valid=True, cases=len(cases), network_calls=0)))
        return
    with args.output.open('x') as output:
        for repetition in range(args.repeat):
            for specification, result, qa, message, original, restored in cases:
                for enabled in ([False, True] if repetition % 2 == 0 else [True, False]):
                    packet = restored if enabled else original
                    with budget.scope(seconds=240, calls=20, tokens=128000) as usage:
                        prediction, score, diagnostics = await eval_scoring.evaluate(qa, copy.deepcopy(packet))
                    row = dict(case=specification, qa_id=result['qa_id'], repetition=repetition + 1,
                               witness_span_refs=config.CHOICE_WITNESS_SPAN_REFS,
                               restored=enabled, prediction=prediction, score=score,
                               archived_packet_hash=result['packet_hash'],
                               packet_hash=evidence_packet.digest(packet),
                               persona_sha256=hashlib.sha256(message['content'].encode()).hexdigest(),
                               packet=packet, diagnostics=diagnostics, calls=usage.calls, tokens=usage.tokens,
                               caveat='Full reader rerun; original persona bypasses retrieval. '
                                      'No gold/CSV metadata used as evidence; not an end-to-end improvement.')
                    output.write(json.dumps(row, ensure_ascii=False) + '\n')
                    output.flush()
                    print(json.dumps({k: row[k] for k in ('case', 'repetition', 'restored', 'prediction', 'score')}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case', action='append', required=True, help='RUN_DIRECTORY:QA_ID')
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--repeat', type=int, choices=(1, 2, 3), default=2)
    p.add_argument('--inspect', action='store_true')
    p.add_argument('--witness-span-refs', action='store_true',
                   help='Use existing exact source-span witness references in both arms')
    asyncio.run(main(p.parse_args()))
