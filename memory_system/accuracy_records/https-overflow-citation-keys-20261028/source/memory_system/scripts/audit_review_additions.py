"""Expose changed positive reviews, including non-reference sources, for inspection.

This is an offline comparison, not a semantic judge. Citation validity and
reference identity do not establish useful relevance or factual support.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def audit(directory):
    if (directory / 'exit-code').read_text().strip() != '0':
        raise ValueError('Require a completed successful experiment')
    comparison = json.loads((directory / 'funnel-comparison.json').read_text())
    arms = {}
    for arm in ('control', 'treatment'):
        run = directory / arm
        results = {r['search_id']: r for r in records(run / 'completed/data/results/personamem-v2-32k.jsonl')}
        rows = {}
        for trace in records(run / 'server-search-debug.jsonl'):
            result = results[trace['search_id']]
            if result.get('error_stage'):
                raise ValueError('Cannot compare an errored question')
            qid = result['qa_id']
            if qid in rows:
                raise ValueError('Ambiguous question identity')
            main = trace['cascade']['listwise']
            reviews = [('main', i, r) for i, r in enumerate(main.get('recovery', {}).get('batches', []))]
            reviews.append(('overflow', 0, trace['cascade'].get('overflow', {}).get('review', {})))
            judgments = {}
            for stage, batch, review in reviews:
                cards = {c['candidate_id']: c for c in review.get('sources', [])}
                for judgment in review.get('judgments', []):
                    mid = judgment['candidate_id']
                    key = stage, mid
                    if key in judgments:
                        raise ValueError('Candidate was judged twice in the same stage')
                    judgments[key] = dict(judgment=judgment, card=cards[mid], batch=batch,
                                          request_cards_sha256=fingerprint(review.get('sources', [])))
            rows[qid] = dict(trace=trace, result=result, judgments=judgments)
        if len(rows) != comparison['summaries'][arm]['questions']:
            raise ValueError('Incomplete trace inventory')
        arms[arm] = rows
    if arms['control'].keys() != arms['treatment'].keys():
        raise ValueError('Question sets differ')
    changes = []
    def status(record):
        if record is None:
            return 'not_reviewed'
        j = record['judgment']
        return 'invalid' if not j['valid'] else 'positive' if j['useful'] else 'negative'
    def packet_ids(trace):
        return {mid for item in trace['returned'] for mid in [item['id'], *item.get('equivalent_ids', [])]}
    for qid, right in arms['treatment'].items():
        left = arms['control'][qid]
        ta, tb = left['trace'], right['trace']
        if (ta['query'], ta['options']) != (tb['query'], tb['options']):
            raise ValueError('Question content differs')
        for key in left['judgments'].keys() | right['judgments'].keys():
            a, b = left['judgments'].get(key), right['judgments'].get(key)
            before, after = status(a), status(b)
            if before == after or 'positive' not in (before, after):
                continue
            stage, mid = key
            changes.append(dict(qa_id=qid, stage=stage, candidate_id=mid,
                control=before, treatment=after, query=ta['query'], options=ta['options'],
                same_candidate_sources=(a['card'] == b['card'] if a and b else None),
                same_review_sources=(a['request_cards_sha256'] == b['request_cards_sha256'] if a and b else None),
                control_packet_member=mid in packet_ids(ta), treatment_packet_member=mid in packet_ids(tb),
                control_review=a, treatment_review=b,
                control_score=left['result']['score'], treatment_score=right['result']['score']))
    changes.sort(key=lambda r: (int(r['qa_id']), r['stage'], r['candidate_id']))
    summary = dict(questions=len(arms['control']), changed_positive_reviews=len(changes),
        transitions=dict(Counter(r['control'] + ' -> ' + r['treatment'] for r in changes)),
        new_positive_packet_members=sum(r['treatment'] == 'positive' and r['treatment_packet_member']
                                       and not r['control_packet_member'] for r in changes))
    report = dict(summary=summary, changes=changes, limitation=__doc__)
    (directory / 'review-additions-audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    audit(parser.parse_args().run)
