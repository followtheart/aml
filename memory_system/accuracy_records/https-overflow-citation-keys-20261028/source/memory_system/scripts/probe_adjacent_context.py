"""Offline packet proposal: restore the exact user turn of a retained reply.

Only already eligible fine candidates can supply context. A supplier must own
the identical retained assistant fragments plus exactly their immediately
preceding complete user turn. No query terms or gold labels affect admission.
Uses the production proposal without provider calls; frozen-output results are not a new HTTPS evaluation.
"""
import argparse
import ast
import copy
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import answer_context, evidence_packet, schemas
import audit_https_funnel as audit


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


from app.conversation_context import propose


def main(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    if (args.run/'exit-code').read_text().strip() != '0':
        raise ValueError('Require a completed successful arm')
    result_rows = {r['search_id']: r for r in records(args.run/'completed/data/results/personamem-v2-32k.jsonl')}
    memories = records(args.run/'server-memory-debug.jsonl')
    raw = {}
    for event in memories:
        raw.setdefault(event['user_id'], {}).update({audit.identity(s): s for s in event['source_messages']})
    csv.field_size_limit(32*1024*1024)
    with args.gold_csv.open() as stream:
        gold = list(csv.DictReader(stream))
    rows = []
    for trace in records(args.run/'server-search-debug.jsonl'):
        result = result_rows[trace['search_id']]
        if evidence_packet.digest(trace['returned']) != trace['packet_hash']:
            raise ValueError('Original packet identity differs')
        by_id = {c['id']: c for c in trace['ranked']}
        fine = [by_id[mid] for mid in trace['cascade']['fine']['selected_ids']]
        # Admission is complete before loading this question's reference text.
        proposed, details = propose(trace['returned'], fine, user_id=trace['user_id'], top_k=trace['top_k'],
                                    byte_budget=trace['coverage_manifest']['evidence_token_budget'])
        snippet = gold[int(result['qa_id'])]['related_conversation_snippet']
        try:
            snippet = json.loads(snippet)
        except ValueError:
            snippet = ast.literal_eval(snippet)
        turns = [audit.norm(s['content']) for s in snippet if s['role'] == 'user']
        def coverage(items):
            cards = [dict(s, text=s.get('content',''), id=f'{i}:{j}') for i,item in enumerate(items)
                     for j,s in enumerate(item.get('sources',[]))]
            return sum(c['full'] for c in audit.turn_coverage(turns, cards, raw[trace['user_id']]))
        rows.append(dict(qa_id=result['qa_id'], before_turns=coverage(trace['returned']),
                         after_turns=coverage(proposed), proposal=details))
    summary = dict(questions=len(rows), original_packet_identities_verified=len(rows),
        before_reference_questions=sum(r['before_turns']>0 for r in rows),
        after_reference_questions=sum(r['after_turns']>0 for r in rows),
        before_turns=sum(r['before_turns'] for r in rows), after_turns=sum(r['after_turns'] for r in rows),
        added_context_items=sum(len(r['proposal']['appended_ids']) for r in rows),
        added_bytes=sum(r['proposal']['added_bytes'] for r in rows),
        gained=[r['qa_id'] for r in rows if r['after_turns'] and not r['before_turns']],
        lost=[r['qa_id'] for r in rows if r['before_turns'] and not r['after_turns']])
    args.output.write_text(json.dumps(dict(summary=summary, rows=rows, limitation=__doc__), ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gold-csv', type=Path, default=Path(__file__).resolve().parents[1]/'data/raw/personamem-v2.csv')
    main(parser.parse_args())
