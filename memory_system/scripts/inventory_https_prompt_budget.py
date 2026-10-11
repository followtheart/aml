"""Measure complete fine/listwise pools without provider calls or score-based selection."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from probe_ranking_capacity import replay
from replay_query_context import records


def inventory(run, prepared):
    if (run / 'exit-code').read_text().strip() != '0':
        raise ValueError('Require successful complete HTTPS archive')
    results = records(run / 'completed/data/results/personamem-v2-32k.jsonl')
    searches = {r['search_id']: r for r in records(run / 'server-search-debug.jsonl')}
    expected = {(c['conversation_id'], q['id']) for c in records(prepared) for q in c['qa']}
    actual = [(r['conversation_id'], r['qa_id']) for r in results]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError('Incomplete or duplicate cohort')
    rows = []
    for result in results:
        search = searches[result['search_id']]
        if search['packet_hash'] != result['packet_hash']:
            raise ValueError('Packet identity mismatch')
        settings = search['versions']['settings']
        caps = settings['CASCADE_FINE_LIMIT'], settings['CASCADE_LLM_LIMIT']
        original = replay(search, *caps)
        archived = search['cascade']['listwise']
        if (original['input_ids'] != archived['candidate_ids']
                or original['prompt_bytes'] != archived['input_bytes']
                or original['fine_ids'] != search['cascade']['fine']['selected_ids']):
            raise ValueError('Archived admission not reproduced for ' + result['qa_id'])
        unlimited = copy.deepcopy(search)
        unlimited['versions']['settings']['RERANK_MAX_PROMPT_BYTES'] = 10**9
        full = replay(unlimited, *caps)
        if full['prompt_rejected']:
            raise ValueError('Offline full-pool limit insufficient')
        rows.append(dict(conversation_id=result['conversation_id'], qa_id=result['qa_id'],
            search_id=result['search_id'], baseline_limit=settings['RERANK_MAX_PROMPT_BYTES'],
            baseline_bytes=original['prompt_bytes'], baseline_ids=original['input_ids'],
            full_bytes=full['prompt_bytes'], full_ids=full['input_ids'],
            restored_ids=sorted(set(full['input_ids']) - set(original['input_ids']))))
    return dict(run=str(run), prepared=str(prepared),
        prepared_sha256=hashlib.sha256(prepared.read_bytes()).hexdigest(),
        questions=len(rows), all_baselines_exact=True, max_full_bytes=max(r['full_bytes'] for r in rows),
        affected_questions=sum(bool(r['restored_ids']) for r in rows), rows=rows,
        caveat='Offline prompt-byte admission only. Original fine/LLM caps and source rendering retained. '
               'Does not simulate upstream token/time use or prove downstream source transport or accuracy. '
               'A full-pool prompt size must still be checked against actual runtime admission.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['run', 'prepared', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    report = inventory(args.run, args.prepared)
    with args.output.open('x') as stream:
        stream.write(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'rows'}))
