"""Score measured source ranks after all retrieval choices are frozen.

Raw source rank coverage is an upstream diagnostic, never final delivery or
entailment. Keep this separate from budgeted prepared-carrier packet proposals.
"""
import argparse
import ast
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from audit_https_funnel import norm
from probe_semantic_source_context import ranked_sources
from probe_source_exchange_vectors import exchange_text


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(args):
    probe = args.probe.resolve()
    if (probe / 'exit-code').read_text().strip() != '0':
        raise ValueError('Require successful measurement')
    plan = json.loads((probe / 'plan.json').read_text())
    ranks = json.loads((probe / 'rankings.json').read_text())
    if digest(probe / 'plan.json') != ranks['plan_sha256'] or digest(probe / 'vectors.npz') != ranks['vectors_sha256']:
        raise ValueError('Measured inputs changed')
    vectors = np.load(probe / 'vectors.npz', allow_pickle=False)['vectors']
    if vectors.shape != (len(plan['texts']), plan['dimension']) or not np.isfinite(vectors).all():
        raise ValueError('Vector matrix invalid')
    if not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5):
        raise ValueError('Vectors are not normalized')
    sources, docs = plan['sources'], plan['documents']
    for d in docs:
        if d['mode'] == 'exchange':
            members = plan['exchange_sources'][str(d['source_index'])]
            user = sources[d['source_index']]
            if (members[0] != user or d['text'] != exchange_text(members) or len(members) not in (1, 2)):
                raise ValueError('Exchange text or parent differs')
            if len(members) == 2:
                reply = members[1]
                if (reply['role'] != 'assistant' or reply['user_id'] != user['user_id']
                        or reply['request_id'] != user['request_id'] or reply['session_id'] != user['session_id']
                        or reply['message_index'] != user['message_index'] + 1
                        or (plan.get('pairing_policy', 'common_amu_adjacent') != 'eligible_adjacent'
                            and not set(reply['amu_ids']) & set(user['amu_ids']))):
                    raise ValueError('Invalid adjacent exchange')
        elif d['text'] != sources[d['source_index']]['content'][d['start']:d['end']]:
            raise ValueError('Window is not exact original text')
    # Complete every non-gold selection before opening reference metadata.
    selections = {r['qa_id']: {mode: {str(k): ranked_sources(qs)[:k] for k in (8, 40)}
                  for mode, qs in r['modes'].items()} for r in ranks['rankings']}
    csv.field_size_limit(32 * 1024 * 1024)
    with args.gold_csv.open() as f:
        gold = list(csv.DictReader(f))
    root = Path(plan['source_run'])
    stage = json.loads((root / 'control/funnel/stage-transport.json').read_text())
    baseline = {r['qa_id']: r for r in stage['rows']}
    traces = {t['search_id']: t for t in map(json.loads, (root / 'search/server-search-debug.jsonl').open())}
    rows = []
    for r in ranks['rankings']:
        meta = gold[int(r['qa_id'])]
        if norm(ast.literal_eval(meta['user_query'])['content']) != norm(traces[r['search_id']]['query']):
            raise ValueError('Reference question identity differs')
        snippet = ast.literal_eval(meta['related_conversation_snippet'])
        turns = [norm(s['content']) for s in snippet if s['role'] == 'user']
        matches = [{i for i, s in enumerate(sources) if s['user_id'] == r['user_id'] and turn and turn in norm(s['content'])}
                   for turn in turns]
        before = [x['full'] for x in baseline[r['qa_id']]['visible']['catalog']['turn_coverage']]
        if len(before) != len(matches):
            raise ValueError('Reference turn alignment differs')
        modes = {}
        for mode, queries in r['modes'].items():
            coverage = {k: [bool(set(selected) & origins) for origins in matches]
                        for k, selected in selections[r['qa_id']][mode].items()}
            details = []
            for origins in matches:
                hits = []
                for query in queries:
                    hit = next(((rank, value) for rank, value in enumerate(query['hits'], 1)
                                if value['source_index'] in origins), None)
                    if hit:
                        rank, value = hit
                        hits.append(dict(rank=rank, **value, query=query['query']))
                details.append(sorted(hits, key=lambda h: (h['rank'], -h['score']))[:3])
            modes[mode] = dict(coverage=coverage, best_reference_ranks=details,
                selected_sources=selections[r['qa_id']][mode])
        rows.append(dict(qa_id=r['qa_id'], baseline_final=before,
            eligible_reference_turns=[bool(m) for m in matches],
            upstream_missing=not any(before) and any(matches), modes=modes))
    summary = {}
    for mode in ranks['rankings'][0]['modes']:
        summary[mode] = {}
        for k in ('8', '40'):
            summary[mode][k] = dict(
                source_rank_reference_questions=sum(any(r['modes'][mode]['coverage'][k]) for r in rows),
                source_rank_reference_turns=sum(sum(r['modes'][mode]['coverage'][k]) for r in rows),
                union_with_existing_final_questions=sum(any(r['baseline_final']) or any(r['modes'][mode]['coverage'][k]) for r in rows),
                missing_final_questions_found=[r['qa_id'] for r in rows if r['upstream_missing'] and any(r['modes'][mode]['coverage'][k])])
    events = [dict(p.split('=', 1) for p in line.split() if '=' in p)
              for line in (probe / 'provider.log').read_text().splitlines() if 'provider_call ' in line]
    if (len(events) != ranks['calls'] or sum(int(e['total_tokens']) for e in events) != ranks['tokens']
            or any(e['model'] != plan['model'] or e['kind'] != 'embedding' or e['status'] != 'ok' for e in events)):
        raise ValueError('Provider ledger/model differs')
    result = dict(summary=summary, questions=len(rows), baseline_final_reference_questions=sum(any(r['baseline_final']) for r in rows),
        eligible_reference_questions=sum(any(r['eligible_reference_turns']) for r in rows),
        missing_final=[r for r in rows if r['upstream_missing']], rows=rows,
        provider_usage=dict(calls=len(events), tokens=ranks['tokens'], statuses=dict(Counter(e['status'] for e in events))),
        plan_sha256=ranks['plan_sha256'],rankings_sha256=digest(probe / 'rankings.json'),
        source_sha256=digest(Path(__file__)), limitation=__doc__)
    with (probe / 'rank-audit.json').open('x') as out:
        json.dump(result, out, ensure_ascii=False, indent=2)
        out.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('rows', 'missing_final')}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe', type=Path, required=True)
    parser.add_argument('--gold-csv', type=Path, default=Path('data/raw/personamem-v2.csv'))
    main(parser.parse_args())
