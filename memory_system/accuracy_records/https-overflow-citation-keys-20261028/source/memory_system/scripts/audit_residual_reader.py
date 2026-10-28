"""Inspect novel residual sources, prior reviews, and explicit reader citations.

Citing a newly delivered source is observable usage, not proof that it supports
a personal claim or caused a correct answer. No provider calls are made.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys


def records(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def source_key(card):
    return json.dumps({k: card.get(k) for k in ('role', 'request_id', 'message_index', 'source_event_id',
                      'content_span', 'text', 'timestamp')}, ensure_ascii=False, sort_keys=True)


def citations(value, cards, path=()):
    found = []
    if isinstance(value, dict):
        sid = value.get('source_id')
        if isinstance(sid, str) and sid in cards:
            quote = value.get('quote')
            found.append(dict(path=list(path), source_id=sid, quote=quote,
                              exact_quote=isinstance(quote, str) and bool(quote) and quote in cards[sid]['text']))
        for k, v in value.items():
            found.extend(citations(v, cards, (*path, k)))
    elif isinstance(value, list):
        for i, v in enumerate(value):
            found.extend(citations(v, cards, (*path, i)))
    return found


def main(a):
    root = a.run.resolve()
    if (root / 'exit-code').read_text().strip() != '0':
        raise ValueError('Require completed successful paired trial')
    plan=json.loads((root/'plan.json').read_text())
    for path, digest in plan['source_files'].items():
        if hashlib.sha256((root/'source'/path).read_bytes()).hexdigest()!=digest:
            raise ValueError('Frozen source changed')
    sys.path.insert(0, str(root / 'source/memory_system'))
    from app import answer_choice, evidence_packet
    results = {}
    for arm in ('control', 'treatment'):
        results[arm] = {r['qa_id']: r for r in records(root / arm / 'completed/data/results/personamem-v2-32k.jsonl')}
    if results['control'].keys() != results['treatment'].keys():
        raise ValueError('Question identities differ')
    hashes = {}
    rows = []
    for arm in results:
        by_search = {r['search_id']: r for r in results[arm].values()}
        for t in records(root / arm / 'server-search-debug.jsonl'):
            r = by_search[t['search_id']]
            ctx = t['cascade']['conversation_context']
            n = len(ctx['base_packet_ids']) + len(ctx.get('prefix_reserve_ids', ctx['appended_ids']))
            prefix = t['returned'][:n]
            digest = evidence_packet.digest(prefix)
            hashes[arm, r['qa_id']] = digest
            if arm == 'control':
                continue
            prefix = [dict(p, packet_hash=digest) for p in prefix]
            old_catalog = answer_choice.build_catalog(prefix)
            actual_catalog = answer_choice.build_catalog(t['returned'])
            assert list(actual_catalog[0].values()) == r['answer_source_catalog']
            assert list(actual_catalog[1].values()) == r['answer_constraint_catalog']
            old_keys = {source_key(c) for c in old_catalog[0].values()}
            novel = {sid: c for sid, c in actual_catalog[0].items() if source_key(c) not in old_keys}
            old_constraint_keys = {source_key(c) for c in old_catalog[1].values()}
            novel_constraints = {sid: c for sid, c in actual_catalog[1].items() if source_key(c) not in old_constraint_keys}
            references = []
            for call in r.get('answer_calls', []):
                for citation in citations(call.get('response'), novel | novel_constraints):
                    references.append(dict(stage=call['stage'], status=call.get('status'), **citation))
            reviews = {}
            requests = list(t['cascade'].get('listwise', {}).get('recovery', {}).get('batches', []))
            requests.append(t['cascade'].get('overflow', {}).get('review', {}))
            for request in requests:
                for judgment in request.get('judgments', []):
                    reviews[judgment['candidate_id']] = ('invalid' if not judgment['valid'] else
                                                       'positive' if judgment['useful'] else 'negative')
            ids = ctx['residual']['appended_ids']
            assert [p['id'] for p in t['returned'][n:]] == ids
            rows.append(dict(qa_id=r['qa_id'], control_score=results['control'][r['qa_id']]['score'],
                treatment_score=r['score'], appended_ids=ids, added_bytes=ctx['residual']['added_bytes'],
                prior_reviews={mid: reviews.get(mid, 'not_reviewed') for mid in ids},
                novel_catalog_sources=novel, novel_catalog_constraints=novel_constraints, reader_references=references))
    assert len(rows) == len(results['treatment'])
    for row in rows:
        q = row['qa_id']
        row['same_prefix_between_arms'] = hashes['control', q] == hashes['treatment', q]
    summary = dict(questions=len(rows), appended_items=sum(len(r['appended_ids']) for r in rows),
        added_bytes=sum(r['added_bytes'] for r in rows),
        novel_sources_by_role=dict(Counter(c['role'] for r in rows for c in r['novel_catalog_sources'].values())),
        novel_constraints=sum(len(r['novel_catalog_constraints']) for r in rows),
        prior_reviews=dict(Counter(v for r in rows for v in r['prior_reviews'].values())),
        cited_novel_source_questions=[r['qa_id'] for r in rows if any(c['exact_quote'] for c in r['reader_references'])],
        same_prefix_between_arms=sum(r['same_prefix_between_arms'] for r in rows),
        wins=[r['qa_id'] for r in rows if r['treatment_score'] > r['control_score']],
        losses=[r['qa_id'] for r in rows if r['treatment_score'] < r['control_score']])
    output = root / 'residual-reader-audit.json'
    with output.open('x') as f:
        json.dump(dict(summary=summary, rows=rows, limitation=__doc__,
                       auditor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()), f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    main(p.parse_args())
