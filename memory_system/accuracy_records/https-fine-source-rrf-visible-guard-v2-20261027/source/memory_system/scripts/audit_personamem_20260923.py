"""Offline, standard-library-only audit of the 2026-09-23 PersonaMem run.

Run from any directory. Writes derived artifacts only; never calls a provider.
Source probes are manually selected related passages, NOT gold entailment labels.
"""
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'memory_system/runs/personamem-20260923-audit'


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
    memory, searches, results, prepared = map(load, paths)
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
        assert set(s['selection']['selected_ids']) == set(r['coverage_manifest']['included_ids'])
        a = next(a for a in r['choice_alignment'] if a['letter'] == gold)
        traces = []
        for label, quote in probes.get(int(r['qa_id']), []):
            source = source_by_label[label]
            assert quote in source['content'], (label, quote)
            mids = {mid for mid, m in snapshots.items()
                    if m['user_id'] == s['user_id'] and any(identity(z) == identity(source)
                                                           for z in m.get('sources', []))}
            cascade = s['cascade']
            stage_ids = {
                'fused': {x['id'] for x in s['fused']},
                'coarse': set(cascade['coarse']['selected_ids']),
                'fine': set(cascade['fine']['selected_ids']),
                'listwise_input': set(cascade['listwise']['candidate_ids']),
                'packet': set(r['coverage_manifest']['included_ids']),
            }
            cards = [z for z in r['answer_source_catalog'] if identity(z) == identity(source)]
            traces.append(dict(label=label, quote=quote, role=source['role'],
                recorded_memory_ids=sorted(mids),
                stage_memory_ids={k: sorted(mids & v) for k, v in stage_ids.items()},
                visible_source_ids=[z['id'] for z in cards if quote in z['text']],
                visible_same_event_ids=[z['id'] for z in cards],
                hidden_packet_sources=[dict(memory_id=m['id'], reason=z.get('content_omitted'))
                    for m in s['returned'] for z in m.get('sources', [])
                    if identity(z) == identity(source) and not z.get('content')],
                eliminations=[o for o in cascade['omitted'] if o['id'] in mids]))
        eligible = r['answer_eligible_options']
        terminal = ('correct' if r['score'] else 'empty_answer' if not r['prediction']
                    else 'gold_excluded' if gold not in eligible else 'selection_error')
        rows.append(dict(qa_id=r['qa_id'], category=r['category'], question=q['question'],
            gold=gold, gold_option=q['options'][ord(gold)-65], prediction=r['prediction'],
            score=r['score'], terminal=terminal, eligible=eligible,
            blocked=r['answer_blocked_options'], gold_alignment=a,
            support_fallback=r.get('answer_support_fallback'),
            witness=r['answer_witness_retrieval'], calls=r['answer_calls'], probes=traces))
    fallback = [r for r in results if r.get('answer_support_fallback')]
    gold_eligible = [r for r in rows if r['gold'] in r['eligible']]
    summary = dict(
        input_sha256={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths},
        source_count=len(source_rows), unique_source_count=len(raw),
        linked_source_count=len(set(raw) & linked_sources), source_matches_prepared=True,
        source_roles=distribution(s['role'] for s in source_rows),
        memory_events=len(memory), snapshot_rows=sum(len(e['memories']) for e in memory),
        unique_memory_ids=len(snapshots), memory_types=distribution(m['type'] for m in snapshots.values()),
        conversation_ids=sorted({r['conversation_id'] for r in results}),
        pipeline=results[0]['versions']['search_policy'], answer_policy=results[0]['answer_policy'],
        model=results[0]['model'], code_files_checked=len(results[0]['versions']['file_hashes']),
        code_mismatches=[p for p, h in results[0]['versions']['file_hashes'].items()
                         if hashlib.sha256((ROOT / p).read_bytes()).hexdigest() != h],
        aligned_results=len(rows), score=sum(r['score'] for r in results),
        terminal=distribution(r['terminal'] for r in rows),
        gold_eligible=[r['qa_id'] for r in gold_eligible],
        support_fallback_count=len(fallback), fallback_empty=sum(not r['prediction'] for r in fallback),
        fallback_score=sum(r['score'] for r in fallback),
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
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'audit.json').write_text(json.dumps(dict(summary=summary, rows=rows), ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    for r in rows:
        print('Q'+r['qa_id'], 'gold='+r['gold'], 'pred='+repr(r['prediction']), r['terminal'],
              [(p['label'], {k: len(v) for k, v in p['stage_memory_ids'].items()},
                p['visible_source_ids']) for p in r['probes']])


if __name__ == '__main__':
    main()
