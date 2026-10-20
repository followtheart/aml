"""Replay archived CE scores and prepared evidence; never calls a provider.

Gold snippets measure transport only, after selection. No answer accuracy is
estimated. Require exact reproduction of the archived baseline before comparing.
"""
import argparse
import ast
import copy
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app import evidence_selection
import audit_https_funnel as audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run = args.run
    searches = {s['search_id']: s for s in audit.records(run / 'server-search-debug.jsonl')}
    results = list(audit.records(run / 'completed/data/results/personamem-v2-32k.jsonl'))
    memory = list(audit.records(run / 'server-memory-debug.jsonl'))
    csv.field_size_limit(32 * 1024 * 1024)
    with (ROOT / 'data/raw/personamem-v2.csv').open() as stream:
        gold = list(csv.DictReader(stream))
    prepared = {r['conversation_id']: r for r in
                audit.records(ROOT / 'data/prepared/personamem-v2-32k.jsonl')}
    audit.analyze(results, memory, list(searches.values()), prepared, gold)
    output = []
    for result in results:
        search = searches[result['search_id']]
        fine = search['cascade']['fine']
        by_id = {c['id']: copy.deepcopy(c) for c in search['ranked']}
        rows = [by_id[mid] for mid in search['cascade']['coarse']['selected_ids']]
        scores = {d['id']: d['score'] for d in fine['selection']['decisions']}
        for row in rows:
            row['_ce_score'] = scores[row['id']]
        requirements = search['coverage_manifest']['coverage']['requirements']
        def members(reason):
            return {r['candidate_id'] for r in fine['selection']['reservations']
                    if reason in r['reasons']}
        def key(c):
            return (c['_ce_score'] if c['_ce_score'] is not None else -1e30, c.get('_fused', 0))
        kwargs = dict(reserve_ids=members('unscored_rescue'), fused_ids=members('fused_head'))
        before = evidence_selection.select(rows, fine['output_count'], requirements, key, **kwargs)
        if [c['id'] for c in before] != fine['selected_ids']:
            raise ValueError(f"Baseline replay mismatch: qa_id={result['qa_id']}")
        trace = {}
        after = evidence_selection.select(rows, fine['output_count'], requirements, key, **kwargs,
                                         bound_soft_reservations=True, trace=trace)
        # Evaluation labels are read only after both selections finish.
        snippet = gold[int(result['qa_id'])]['related_conversation_snippet']
        try:
            snippet = json.loads(snippet)
        except ValueError:
            snippet = ast.literal_eval(snippet)
        turns = [audit.norm(x['content']) for x in snippet if x['role'] == 'user']
        raw = {audit.identity(s): s for e in memory if e['user_id'] == result['user_id']
               for s in e['source_messages']}
        def coverage(candidates):
            cards = [dict(s, id=f'{i}:{j}', text=s.get('content', ''))
                     for i, c in enumerate(candidates)
                     for j, s in enumerate(c.get('_packet_item', {}).get('sources', []))]
            return [x['full'] for x in audit.turn_coverage(turns, cards, raw)]
        output.append(dict(qa_id=result['qa_id'], before=coverage(before), after=coverage(after),
                           before_ids=[c['id'] for c in before], after_ids=[c['id'] for c in after],
                           selection=trace))
    summary = dict(questions=len(output), baseline_reproduced=True,
                   before_full_turns=sum(sum(r['before']) for r in output),
                   after_full_turns=sum(sum(r['after']) for r in output),
                   gained=[r['qa_id'] for r in output if any(b and not a for a, b in zip(r['before'], r['after']))],
                   lost=[r['qa_id'] for r in output if any(a and not b for a, b in zip(r['before'], r['after']))],
                   caveat='Fixed-candidate, fixed-score fine-selection replay only; no new answer evaluation.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(summary=summary, rows=output), ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
