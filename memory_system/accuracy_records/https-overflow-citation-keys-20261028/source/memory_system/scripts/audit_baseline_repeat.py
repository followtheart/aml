"""Compare completed baseline cohorts without attributing score changes to code.

Catalog equality isolates delivered inputs, not every provider request or hidden
provider state. This is a descriptive paired audit, not a causal experiment.
"""
import argparse
import hashlib
import json
from pathlib import Path


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load(run):
    if (run/'exit-code').read_text().strip() != '0':
        raise ValueError('Require successful completed arm')
    rows = records(run/'completed/data/results/personamem-v2-32k.jsonl')
    traces = {t['search_id']: t for t in records(run/'server-search-debug.jsonl')}
    funnel = json.loads((run/'funnel/stage-transport.json').read_text())
    if funnel['summary']['progress_snapshot'] or any(r.get('error_stage') for r in rows):
        raise ValueError('Incomplete cohort')
    stages = {r['qa_id']: r for r in funnel['rows']}
    result = {}
    for r in rows:
        t = traces[r['search_id']]
        if (r['packet_hash'] != t['packet_hash'] or r['qa_id'] in result
                or r['fake'] or r['transport'] != 'https'):
            raise ValueError('Result/trace identity mismatch')
        result[r['qa_id']] = r, t, stages[r['qa_id']]
    return result


def main(args):
    before, after = load(args.before), load(args.after)
    if before.keys() != after.keys():
        raise ValueError('Cohorts differ')
    rows = []
    for q in sorted(before, key=int):
        a, ta, sa = before[q]
        b, tb, sb = after[q]
        if (a['conversation_id'] != b['conversation_id'] or ta['query'] != tb['query']
                or ta['options'] != tb['options']):
            raise ValueError('Question identity mismatch')
        va, vb = ta['versions']['settings'], tb['versions']['settings']
        def main_input(t):
            ranked = {r['id']: r for r in t['ranked']}
            return [(mid, ranked[mid].get('_rank_text', ranked[mid].get('content')))
                    for mid in t['cascade']['listwise']['candidate_ids']]
        rows.append(dict(qa_id=q, before_score=a['score'], after_score=b['score'],
            before_prediction=a['prediction'], after_prediction=b['prediction'],
            same_query=True, same_plan=ta['plan']==tb['plan'],
            same_main_input=main_input(ta)==main_input(tb),
            same_fine=ta['cascade']['fine']['selected_ids']==tb['cascade']['fine']['selected_ids'],
            same_catalog=all(a.get(k)==b.get(k) for k in ('answer_source_catalog','answer_constraint_catalog')),
            same_answer_policy=a['answer_policy']==b['answer_policy'],
            before_reference_turns=sa['visible']['packet']['full_turns'],
            after_reference_turns=sb['visible']['packet']['full_turns'],
            common_settings_differences={k: [va[k],vb[k]] for k in va.keys() & vb.keys() if va[k]!=vb[k]},
            added_settings={k: vb[k] for k in vb.keys()-va.keys()}))
    summary = dict(questions=len(rows), before_score=sum(r['before_score'] for r in rows),
        after_score=sum(r['after_score'] for r in rows),
        wins=[r['qa_id'] for r in rows if r['after_score']>r['before_score']],
        losses=[r['qa_id'] for r in rows if r['after_score']<r['before_score']],
        same_catalog_score_changes=[r['qa_id'] for r in rows if r['same_catalog'] and r['before_score']!=r['after_score']],
        same_main_input=sum(r['same_main_input'] for r in rows),
        same_fine=sum(r['same_fine'] for r in rows), same_catalog=sum(r['same_catalog'] for r in rows),
        reference_gains=[r['qa_id'] for r in rows if not r['before_reference_turns'] and r['after_reference_turns']],
        reference_losses=[r['qa_id'] for r in rows if r['before_reference_turns'] and not r['after_reference_turns']])
    result = dict(before=str(args.before.resolve()), after=str(args.after.resolve()), summary=summary,
        rows=rows, limitation=__doc__, auditor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with args.output.open('x') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before', type=Path, required=True)
    parser.add_argument('--after', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    main(parser.parse_args())
