"""Offline HTTPS source transport through every ranking stage.

Full normalized user turns are probes, not semantic entailment labels.
Run audit_https_funnel.py first to validate dataset/question identity.
"""
import argparse
import ast
import csv
from datetime import datetime, timezone
import json
from collections import Counter
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_https_funnel as a

ROOT = Path(__file__).resolve().parents[1]
def load(path, partial=False):
    # A live writer can be between writes within the last JSONL record.
    # Only ignore an unterminated final line in explicit progress mode.
    data = path.read_bytes()
    if partial and data and not data.endswith(b'\n'):
        data = data[:data.rfind(b'\n') + 1]
    return [json.loads(line) for line in data.splitlines() if line.strip()]


def reference_set_coverage(raw_coverage, stage_coverage):
    """Recall conditional on availability, plus strict full-reference delivery.

    Missing raw turns cannot be demanded of retrieval. A nonempty available
    subset can be complete while the original full reference set is not.
    Neither metric establishes semantic support for the answer.
    """
    if len(raw_coverage) != len(stage_coverage):
        raise ValueError('Reference turn inventories differ between stages')
    available = {i for i, turn in enumerate(raw_coverage) if turn['full']}
    delivered = {i for i, turn in enumerate(stage_coverage) if turn['full']}
    return dict(available_turns=len(available), delivered_available_turns=len(available & delivered),
        available_set_complete=bool(available) and available <= delivered,
        reference_set_complete=bool(raw_coverage) and len(delivered) == len(raw_coverage))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--gold-csv', type=Path, default=ROOT / 'data/raw/personamem-v2.csv')
    parser.add_argument('--prepared', type=Path, default=ROOT / 'data/prepared/personamem-v2-32k.jsonl')
    parser.add_argument('--allow-partial', action='store_true',
                        help='Write progress-stage-transport.json; never overwrite the final audit')
    args = parser.parse_args()
    RUN = args.run.resolve()
    if not args.allow_partial and not (RUN / 'exit-code').exists():
        parser.error('Run is still active; stage audit requires a completed run')
    results = load(RUN / 'completed/data/results/personamem-v2-32k.jsonl', args.allow_partial)
    if not results:
        parser.error('No completed question available for a stage audit')
    if len({(r['conversation_id'], r['qa_id']) for r in results}) != len(results):
        raise ValueError('Duplicate question identity')
    if any(r.get('fake') is not False or r.get('transport') != 'https' for r in results):
        raise ValueError('Expected non-fake HTTPS results')
    if len({r['run_id'] for r in results}) != 1:
        raise ValueError('Mixed experiment identities')
    search_rows = load(RUN / 'server-search-debug.jsonl', args.allow_partial)
    searches = {s['search_id']: s for s in search_rows}
    if len(searches) != len(search_rows):
        raise ValueError('Duplicate search identity')
    memory = load(RUN / 'server-memory-debug.jsonl', args.allow_partial)
    csv.field_size_limit(32 * 1024 * 1024)
    with args.gold_csv.open() as f:
        gold = list(csv.DictReader(f))
    prepared = {r['conversation_id']: r for r in load(args.prepared)}
    # Reuse the full question/options/history/packet identity checks before
    # joining gold snippets to any stage. A qa_id alone is not sufficient.
    a.analyze(results, memory, list(searches.values()), prepared, gold)
    stages = ['memory', 'fused', 'coarse', 'fine', 'listwise_input', 'listwise_selected', 'selected', 'packet', 'catalog']
    if any('overflow' in s.get('cascade', {}) for s in searches.values()):
        stages[6:6] = ['overflow_input', 'overflow_selected']
    if any('conversation_context' in s.get('cascade', {}) for s in searches.values()):
        stages.insert(stages.index('packet'), 'packet_base')
    rows = []
    for r in results:
        s = searches[r['search_id']]
        if s['user_id'] != r['user_id'] or s['packet_hash'] != r['packet_hash']:
            raise ValueError('Search user or packet mismatch')
        events = [e for e in memory if e['user_id'] == r['user_id']]
        raw = {a.identity(x): x for e in events for x in e['source_messages']}
        memories = {x['id']: x for e in events for x in e['memories']}
        snippet = gold[int(r['qa_id'])]['related_conversation_snippet']
        try: snippet = json.loads(snippet)
        except ValueError: snippet = ast.literal_eval(snippet)
        turns = [a.norm(x['content']) for x in snippet if x['role'] == 'user']
        assistant_turns = [a.norm(x['content']) for x in snippet if x['role'] == 'assistant']
        origins = {key for key, x in raw.items() if x['role'] == 'user' and a.matches(x['content'], turns)}
        mids = {mid for mid, x in memories.items() if any(a.identity(z) in origins for z in x.get('sources', []))}
        c = s['cascade']
        ids = {'memory': set(memories), 'fused': {x['id'] for x in s['fused']},
               'coarse': set(c['coarse']['selected_ids']), 'fine': set(c['fine']['selected_ids']),
               'listwise_input': set(c['listwise']['candidate_ids']),
               'listwise_selected': set(c['listwise']['selected_ids']),
               'selected': set(s['selection']['selected_ids']), 'packet': {x['id'] for x in s['returned']}}
        if 'overflow_input' in stages:
            overflow = c.get('overflow', {})
            ids['overflow_input'] = ids['listwise_selected'] | set(overflow.get('review', {}).get('candidate_ids', []))
            ids['overflow_selected'] = set(overflow.get('selected_ids', c['listwise']['selected_ids']))
        if 'packet_base' in stages:
            ids['packet_base'] = set(c['conversation_context']['base_packet_ids'])
        items = {x['id']: x.get('_packet_item', {}) for x in s['ranked']}
        raw_cards = [dict(x, id=str(i), text=x['content']) for i, x in enumerate(raw.values())]
        raw_coverage = a.turn_coverage(turns, raw_cards, raw)
        raw_assistant_coverage = a.turn_coverage(assistant_turns, raw_cards, raw, role='assistant')
        visible, assistant_visible = {}, {}
        for stage in stages[1:]:
            if stage == 'catalog':
                cards = r.get('answer_source_catalog', []) + r.get('answer_constraint_catalog', [])
            else:
                current = (s['returned'] if stage == 'packet' else
                           [item for item in s['returned'] if item['id'] in ids['packet_base']] if stage == 'packet_base' else
                           [items.get(mid, {}) for mid in ids[stage]])
                cards = [dict(z, id=f'{i}:{j}', text=z.get('content', '')) for i, item in enumerate(current)
                         for j, z in enumerate(item.get('sources', []))]
            coverage = a.turn_coverage(turns, cards, raw)
            visible[stage] = {'turn_coverage': coverage, 'full_turns': sum(x['full'] for x in coverage),
                              'covered_chars': sum(x['covered_chars'] for x in coverage),
                              'total_chars': sum(x['normalized_chars'] for x in coverage),
                              **reference_set_coverage(raw_coverage, coverage)}
            assistant_coverage = a.turn_coverage(assistant_turns, cards, raw, role='assistant')
            assistant_visible[stage] = dict(turn_coverage=assistant_coverage,
                full_turns=sum(x['full'] for x in assistant_coverage),
                covered_chars=sum(x['covered_chars'] for x in assistant_coverage),
                total_chars=sum(x['normalized_chars'] for x in assistant_coverage),
                **reference_set_coverage(raw_assistant_coverage, assistant_coverage))
        route_ids = {x['id'] for route in s['routes'] for x in route.get('candidates', [])}
        carrier_ids = {stage: sorted(mids & group) for stage, group in ids.items()}
        omissions = c.get('omitted', []) + s['coverage_manifest'].get('omitted', []) + s['coverage_manifest'].get('unit_omitted', [])
        rows.append(dict(conversation_id=r['conversation_id'], qa_id=r['qa_id'], score=r['score'], snippet_turns=len(turns), raw_full_turns=sum(x['full'] for x in raw_coverage),
                         raw_turn_coverage=raw_coverage,
                         raw_assistant_turn_coverage=raw_assistant_coverage,
                         assistant_visible=assistant_visible,
                         carrier_ids=carrier_ids, visible=visible, recall_route_carrier_ids=sorted(mids & route_ids),
                         first_carrier_loss=next((stage for stage in stages[:-1] if not carrier_ids[stage]), None),
                         related_omissions=[x for x in omissions if x.get('id') in mids],
                         stage_sizes={stage: len(group) for stage, group in ids.items()}))
    summary = dict(count=len(rows), progress_snapshot=args.allow_partial,
        sampled_at=datetime.now(timezone.utc).isoformat(),
        question_ids=[dict(conversation_id=r['conversation_id'], qa_id=r['qa_id']) for r in rows],
        recall_route_question_count=sum(bool(r['recall_route_carrier_ids']) for r in rows), carrier_question_counts={stage: sum(bool(r['carrier_ids'][stage]) for r in rows) for stage in stages[:-1]},
        visible_full_turn_question_counts={stage: sum(r['visible'][stage]['full_turns'] > 0 for r in rows) for stage in stages[1:]},
        visible_full_turn_counts={stage: sum(r['visible'][stage]['full_turns'] for r in rows) for stage in stages[1:]},
        available_reference_question_count=sum(r['raw_full_turns'] > 0 for r in rows),
        available_set_complete_question_counts={stage: sum(r['visible'][stage]['available_set_complete'] for r in rows)
                                                for stage in stages[1:]},
        strict_reference_set_complete_question_counts={stage: sum(r['visible'][stage]['reference_set_complete'] for r in rows)
                                                       for stage in stages[1:]},
        delivered_available_turn_counts={stage: sum(r['visible'][stage]['delivered_available_turns'] for r in rows)
                                          for stage in stages[1:]},
        raw_full_turn_counts=sum(r['raw_full_turns'] for r in rows),
        total_snippet_turns=sum(r['snippet_turns'] for r in rows),
        assistant_reference_transport=dict(
            raw_full_turns=sum(x['full'] for r in rows for x in r['raw_assistant_turn_coverage']),
            available_reference_questions=sum(any(x['full'] for x in r['raw_assistant_turn_coverage']) for r in rows),
            visible_full_turn_question_counts={stage:sum(r['assistant_visible'][stage]['full_turns']>0 for r in rows)
                                               for stage in stages[1:]},
            delivered_available_turn_counts={stage:sum(r['assistant_visible'][stage]['delivered_available_turns'] for r in rows)
                                             for stage in stages[1:]},
            covered_chars={stage:sum(r['assistant_visible'][stage]['covered_chars'] for r in rows)
                           for stage in stages[1:]},
            caveat='Assistant text transport only, never independent user facts. Full assistant turns can include long generic advice; partial excerpts may suffice. User transport metrics remain separate.'),
        first_carrier_loss=dict(Counter(r['first_carrier_loss'] or 'retained' for r in rows)),
        stage_candidate_counts={stage: sum(r['stage_sizes'][stage] for r in rows) for stage in stages[:-1]},
        transport_transitions=[dict(stage=stage,
            lost=[r['qa_id'] for r in rows if r['visible'][prev]['full_turns'] and not r['visible'][stage]['full_turns']],
            gained=[r['qa_id'] for r in rows if not r['visible'][prev]['full_turns'] and r['visible'][stage]['full_turns']])
            for prev, stage in zip(stages[1:-1], stages[2:])],
        caveat='Carrier IDs measure source linkage, not visible text. Visible stages before packet use fixed prepared units from ranked snapshots; fused coverage therefore includes unit preparation. Complete normalized user turns are measured, including constraint catalog. Dependencies, protected rules and recovery can reintroduce sources, so transitions include gains. These are text transport proxies, not entailment or causal attribution.')
    (RUN / 'funnel').mkdir(exist_ok=True)
    output = RUN / 'funnel' / ('progress-stage-transport.json' if args.allow_partial else 'stage-transport.json')
    output.write_text(json.dumps(dict(summary=summary, rows=rows), ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
