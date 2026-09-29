"""Offline, standard-library-only funnel audit of an archived PersonaMem iteration.

Run from any directory. Writes derived artifacts only; never calls a provider.
Source probes are manually selected related passages, NOT gold entailment labels.
"""
import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run', required=True, help='Archived iteration directory containing completed/')
parser.add_argument('--probe-reference-run', type=Path,
                    help='Archive defining probe source labels when ingestion chunk boundaries differ')
args = parser.parse_args()
RUN = Path(args.run).resolve()
OUT = RUN / 'funnel'


def load(relative):
    return [json.loads(line) for line in (ROOT / relative).read_text(encoding='utf-8').splitlines() if line.strip()]


def identity(source):
    return source.get('request_id'), source.get('message_index')


def distribution(values):
    return dict(Counter(values))


def main():
    paths = ['memory_system/logs/memory-debug.jsonl',
             'memory_system/logs/search-debug.jsonl',
             'memory_system/data/results/personamem-v2-32k.jsonl',
             'memory_system/data/prepared/personamem-v2-32k.jsonl']
    paths = [str(RUN / 'completed' / p.removeprefix('memory_system/')) for p in paths[:3]] + paths[3:]
    memory, searches, results, prepared = map(load, paths)
    result_search_ids = {r['search_id'] for r in results}
    unrelated_searches = [dict(search_id=s['search_id'], query=s.get('query'), fake=s.get('fake'))
                          for s in searches if s['search_id'] not in result_search_ids]
    searches = [s for s in searches if s['search_id'] in result_search_ids]
    by_search = {s['search_id']: s for s in searches}
    assert len(by_search) == len(searches)
    qa = {(d['conversation_id'], q['id']): q for d in prepared for q in d['qa']}
    source_rows = [s for e in memory for s in e['source_messages']]
    raw = {identity(s): s for s in source_rows}
    snapshots = {m['id']: m for e in memory for m in e['memories']}
    linked_sources = {identity(z) for m in snapshots.values() for z in m.get('sources', [])}
    assert len({r['versions']['pipeline_version'] for r in results}) == 1
    prepared_messages = [s for d in prepared if d['conversation_id'] in {r['conversation_id'] for r in results}
                         for session in d['sessions'] for s in session]
    assert [(s['role'], s['content']) for s in prepared_messages] == [
        (s['role'], s['content']) for s in source_rows]
    source_by_label = {f"S{s['request_id'].split(':')[-1]}:{s['message_index']}": s for s in source_rows}
    reference_path = (args.probe_reference_run.resolve() / 'completed/logs/memory-debug.jsonl'
                      if args.probe_reference_run else None)
    reference_sources = ([s for event in load(str(reference_path)) for s in event['source_messages']]
                         if reference_path else source_rows)
    reference_by_label = {
        f"S{s['request_id'].split(':')[-1]}:{s['message_index']}": s for s in reference_sources}
    # A topical source can be present while the gold's stronger personal claim
    # remains unsupported. These probes measure passage transport only.
    probes = {
        0: [('S1:4', 'twenty quiet minutes on the mat')],
        1: [('S1:12', 'very troubling personal circumstances')],
        2: [('S1:0', 'recent incident during the curriculum meeting')],
        3: [('S8:7', 'the real joy came in catching those early, still-glass waves')],
        4: [('S11:9', 'baking bread from scratch')],
        5: [('S7:9', 'shifting a bit to get comfortable'), ('S11:5', 'never really had back problems')],
        6: [('S3:15', 'the increased pollen in the air')],
        8: [('S0:0', 'Facebook for family updates')],
        10: [('S11:11', 'luxury fashion brands')],
        11: [('S0:2', 'Marcus')],
        12: [('S5:9', 'Claire')],
        13: [('S11:3', 'anime')],
        14: [('S2:12', 'comic book artists and sci-fi panels')],
        15: [('S7:1', 'before an old leg injury started acting up again')],
        16: [('S6:0', 'Back when I was a kid I had some mild asthma')],
        17: [('S1:8', 'hybrid technology'), ('S7:11', 'parts from overseas')],
        18: [('S4:7', 'If I have a tiny garden')],
        19: [('S8:1', 'personal mementos')],
        20: [('S4:9', 'European LPs'), ('S6:12', 'European pressings')],
        21: [('S3:19', 'decentralized assets'), ('S2:16', 'stablecoins'),
             ('S6:8', 'layer-2 blockchain'), ('S2:18', 'CBDCs')],
        22: [('S3:17', 'black-and-white movies'), ('S7:3', 'American films')],
        23: [('S4:3', 'water in local lakes'), ('S5:13', 'bacteria advisories')],
        24: [('S2:14', 'spice blends'), ('S8:11', 'spices')],
        25: [('S5:5', 'cholesterol numbers')],
    }
    rows = []
    for r in results:
        q = qa[r['conversation_id'], r['qa_id']]
        s = by_search[r['search_id']]
        gold = q['gold_labels'][0]
        assert s['packet_hash'] == r['packet_hash']
        assert s['query'] == q['question'] and s['options'] == q['options']
        assert s['user_id'] == f"local:personamem-v2:{r['conversation_id']}"
        assert r['score'] == float(r['prediction'] in q['gold_labels'])
        assert {m['id'] for m in s['returned']} == set(r['coverage_manifest']['included_ids'])
        a = next((a for a in r.get('choice_alignment', []) if a['letter'] == gold), None)
        answer_error = dict(stage=r.get('error_stage'), type=r.get('error_type'),
                            detail=r.get('error_detail')) if a is None else None
        traces = []
        for label, quote in probes.get(int(r['qa_id']), []):
            reference = reference_by_label[label]
            assert quote in reference['content'], (label, quote)
            if reference_path:
                # Chunk numbers are not stable source identities. Match the full
                # original message, never a short topical probe or a new label.
                candidates = [z for z in source_rows
                              if z['role'] == reference['role']
                              and z['content'] == reference['content']
                              and z['request_id'].split(':session:')[0]
                              == reference['request_id'].split(':session:')[0]]
                assert len(candidates) == 1, ('ambiguous_or_missing_probe_source', label, len(candidates))
                source = candidates[0]
            else:
                source = source_by_label[label]
            actual_label = f"S{source['request_id'].split(':')[-1]}:{source['message_index']}"
            mids = {mid for mid, m in snapshots.items()
                    if m['user_id'] == s['user_id'] and any(identity(z) == identity(source)
                                                           for z in m.get('sources', []))}
            cascade = s['cascade']
            stage_ids = {
                'fused': {x['id'] for x in s['fused']},
                'coarse': set(cascade['coarse']['selected_ids']),
                'fine': set(cascade['fine']['selected_ids']),
                'listwise_input': set(cascade['listwise']['candidate_ids']),
                'selected': set(s['selection']['selected_ids']),
                'packet': set(r['coverage_manifest']['included_ids']),
            }
            cards = [z for z in r['answer_source_catalog'] if identity(z) == identity(source)]
            card_ids = {z['id'] for z in cards}
            witness_links = [dict(letter=letter, source_id=w['source_id'], quote=w['quote'],
                                  contains_probe=quote in w['quote'])
                             for letter, witnesses in r.get('answer_witness_prefill', {}).items()
                             for w in witnesses if w['source_id'] in card_ids]
            gold_citations = [dict(claim_index=i, source_id=c['source_id'], quote=c['quote'],
                                   contains_probe=quote in c['quote'])
                              for i, claim in enumerate((a or {}).get('claims', []))
                              for c in claim.get('citations', []) if c['source_id'] in card_ids]
            # Dependency expansion can carry this source through a different AMU.
            # Direct snapshot AMU intersections alone are not evidence recall.
            prepared_items = {x['id']: x.get('_packet_item', {}) for x in s['ranked']}
            traces.append(dict(label=actual_label, reference_label=label, quote=quote, role=source['role'],
                recorded_memory_ids=sorted(mids),
                stage_memory_ids={k: sorted(mids & v) for k, v in stage_ids.items()},
                stage_visible_quote_carriers={k: sorted(mid for mid in v if any(
                    identity(z) == identity(source) and quote in z.get('content', '')
                    for z in prepared_items.get(mid, {}).get('sources', [])))
                    for k, v in stage_ids.items()},
                packet_visible_quote_carriers=[m['id'] for m in s['returned'] if any(
                    identity(z) == identity(source) and quote in z.get('content', '')
                    for z in m.get('sources', []))],
                visible_source_ids=[z['id'] for z in cards if quote in z['text']],
                visible_same_event_ids=[z['id'] for z in cards],
                witness_links=witness_links,
                gold_citations=gold_citations,
                hidden_packet_sources=[dict(memory_id=m['id'], reason=z.get('content_omitted'))
                    for m in s['returned'] for z in m.get('sources', [])
                    if identity(z) == identity(source) and not z.get('content')],
                eliminations=[o for o in cascade['omitted'] if o['id'] in mids],
                packet_omissions=[o for o in s['coverage_manifest']['omitted'] if o['id'] in mids]))
        eligible = r.get('answer_eligible_options', [])
        terminal = ('correct' if r['score'] else 'answer_error' if answer_error else
                    'abstained' if not r['prediction'] or r['prediction'] == 'ABSTAIN'
                    else 'gold_excluded' if gold not in eligible else 'selection_error')
        rows.append(dict(qa_id=r['qa_id'], category=r['category'], question=q['question'],
            gold=gold, gold_option=q['options'][ord(gold)-65], prediction=r['prediction'],
            score=r['score'], terminal=terminal, eligible=eligible,
            blocked=r.get('answer_blocked_options', []), gold_alignment=a, answer_error=answer_error,
            gold_option_verifier_source_ids=sorted({c['source_id'] for claim in (a or {}).get('claims', [])
                                                    for c in claim.get('citations', []) if c.get('valid')}),
            gold_option_verifier_skipped=(a or {}).get('validation_status') == 'unresolved',
            support_fallback=r.get('answer_support_fallback'),
            citation_recovery=r.get('answer_citation_recovery'),
            witness=r.get('answer_witness_retrieval', {}), calls=r.get('answer_calls', []), probes=traces))
    fallback = [r for r in results if r.get('answer_support_fallback')]
    gold_eligible = [r for r in rows if r['gold'] in r['eligible']]
    summary = dict(
        probe_reference=(dict(path=str(reference_path), sha256=hashlib.sha256(reference_path.read_bytes()).hexdigest())
                         if reference_path else None),
        input_sha256={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths},
        source_count=len(source_rows), unique_source_count=len(raw),
        linked_source_count=len(set(raw) & linked_sources), source_matches_prepared=True,
        source_roles=distribution(s['role'] for s in source_rows),
        memory_events=len(memory), snapshot_rows=sum(len(e['memories']) for e in memory),
        unique_memory_ids=len(snapshots), memory_types=distribution(m['type'] for m in snapshots.values()),
        conversation_ids=sorted({r['conversation_id'] for r in results}),
        pipeline=results[0]['versions']['search_policy'], answer_policy=results[0]['answer_policy'],
        pipeline_version=results[0]['versions']['pipeline_version'],
        model=results[0]['model'], code_files_checked=len(results[0]['versions']['file_hashes']),
        code_mismatches=[p for p, h in results[0]['versions']['file_hashes'].items()
                         if hashlib.sha256((ROOT / p).read_bytes()).hexdigest() != h],
        unrelated_searches=unrelated_searches,
        aligned_results=len(rows), score=sum(r['score'] for r in results),
        terminal=distribution(r['terminal'] for r in rows),
        gold_eligible=[r['qa_id'] for r in gold_eligible],
        support_fallback_count=len(fallback), fallback_empty=sum(not r['prediction'] for r in fallback),
        fallback_score=sum(r['score'] for r in fallback),
        answer_calls_scope='chosen_attempt',
        adaptive_retry_status=distribution(r.get('answer_adaptive_retry', {}).get('status', 'disabled') for r in results),
        all_attempt_calls=distribution(f"{c['stage']}:{c['status']}:{c.get('error_type', '')}"
            for r in results for a in r.get('answer_attempts', [dict(diagnostics=r)])
            for c in a['diagnostics'].get('answer_calls', [])),
        answer_calls=distribution(f"{c['stage']}:{c['status']}:{c.get('error_type', '')}"
                                  for r in results for c in r['answer_calls']),
        witness_status=distribution(r['answer_witness_retrieval']['status'] for r in results),
        witness_rejections=distribution(w['reason'] for r in results
                                      for w in r['answer_witness_retrieval'].get('rejected_witnesses', [])),
        fused=sum(len(s['fused']) for s in searches), ranked=sum(len(s['ranked']) for s in searches),
        cascade={stage: {key: sum(s['cascade'][stage].get(key, 0) for s in searches)
                         for key in ['input_count', 'output_count', 'scored_count']}
                 for stage in ['coarse', 'cross_encoder', 'fine', 'listwise', 'rule_cap']},
        ce_status=distribution(s['cascade']['cross_encoder']['status'] for s in searches),
        listwise_status=distribution(s['cascade']['listwise']['status'] for s in searches),
        omitted=distribution(f"{o['stage']}:{o['reason']}" for s in searches for o in s['cascade']['omitted']),
        selected=sum(s['selection']['selected_count'] for s in searches),
        kept_rule_occurrences=distribution(mid for s in searches for mid in s['cascade']['rule_cap']['kept_ids']),
        rule_contents={mid: m['content'] for mid, m in snapshots.items() if m['type'] == 'rule'},
        returned=sum(len(s['returned']) for s in searches),
        packet_omitted=sum(len(s['coverage_manifest']['omitted']) for s in searches),
        unit_omitted=sum(len(s['coverage_manifest']['unit_omitted']) for s in searches),
        hidden_source_occurrences=distribution(z.get('content_omitted', 'unspecified')
            for s in searches for m in s['returned'] for z in m.get('sources', []) if not z.get('content')),
        packet_bytes=dict(min=min(s['coverage_manifest']['token_upper_bound'] for s in searches),
                          mean=mean(s['coverage_manifest']['token_upper_bound'] for s in searches),
                          max=max(s['coverage_manifest']['token_upper_bound'] for s in searches)),
        search_degraded=[r['qa_id'] for r in results if r['search_degraded']],
        search_seconds=[(datetime.fromisoformat(s['finished_at']) - datetime.fromisoformat(s['started_at'])).total_seconds()
                        for s in searches],
    )
    all_probes = [p for r in rows for p in r['probes']]
    summary['probe_funnel'] = dict(
        caveat='Related passage transport, not gold entailment; direct AMU lineage and expanded source visibility differ.',
        questions=sum(bool(r['probes']) for r in rows), raw=len(all_probes),
        written=sum(bool(p['recorded_memory_ids']) for p in all_probes),
        direct_lineage={stage: sum(bool(p['stage_memory_ids'][stage]) for p in all_probes)
                        for stage in ['fused', 'coarse', 'fine', 'listwise_input', 'selected', 'packet']},
        final_quote_visible=sum(bool(p['visible_source_ids']) for p in all_probes))
    OUT.mkdir(parents=True, exist_ok=True)
    comparison = None
    (OUT / 'audit.json').write_text(json.dumps(dict(summary=summary, rows=rows, comparison=comparison),
        ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    for r in rows:
        print('Q'+r['qa_id'], 'gold='+r['gold'], 'pred='+repr(r['prediction']), r['terminal'],
              [(p['label'], {k: len(v) for k, v in p['stage_memory_ids'].items()},
                p['visible_source_ids']) for p in r['probes']])


if __name__ == '__main__':
    main()
