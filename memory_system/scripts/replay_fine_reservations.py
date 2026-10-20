"""Replay archived CE scores and prepared evidence; never calls a provider.

Gold snippets measure transport only, after selection. No answer accuracy is
estimated. Require exact reproduction of the archived baseline before comparing.
"""
import argparse
import ast
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import audit_https_funnel as audit
from probe_ranking_capacity import replay


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--policy', choices=('legacy', 'bounded', 'no_window'), default='bounded')
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
        by_id = {c['id']: c for c in search['ranked']}
        settings = search['versions']['settings']
        baseline = replay(search, settings['CASCADE_FINE_LIMIT'], settings['CASCADE_LLM_LIMIT'])
        if baseline['fine_ids'] != fine['selected_ids']:
            raise ValueError(f"Baseline replay mismatch: qa_id={result['qa_id']}")
        variant = replay(search, settings['CASCADE_FINE_LIMIT'], settings['CASCADE_LLM_LIMIT'], args.policy)
        before = [by_id[mid] for mid in baseline['fine_ids']]
        after = [by_id[mid] for mid in variant['fine_ids']]
        trace = variant['selection']
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
    summary = dict(questions=len(output), baseline_reproduced=True, policy=args.policy,
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
