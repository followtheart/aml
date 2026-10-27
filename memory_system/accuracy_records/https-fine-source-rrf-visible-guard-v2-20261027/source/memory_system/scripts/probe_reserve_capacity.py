"""Offline residual reserve capacity: preserve three seats, use remaining bytes.

No new ranking, gold-based admission, source reads, or provider calls. The
original top_k and 5 KiB additional-context bound still apply. More sources
can add noise; transport gains are not answer-quality gains.
"""
import argparse
import ast
import copy
import csv
import json
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import answer_choice, config, evidence_packet, search_pipeline
from app.fine_context_reserve import propose as reserve
from app.fine_context_lexical import propose as lexical
from evaluation_model_guard import check_versions
import audit_https_funnel as audit


def records(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def fill(base, fine, **kwargs):
    first, old = lexical(base, fine, **kwargs)
    remaining = kwargs['reserve_bytes'] - old['added_bytes']
    result, tail = reserve(first, fine, user_id=kwargs['user_id'], top_k=kwargs['top_k'],
                           byte_budget=kwargs['byte_budget'], reserve_bytes=remaining,
                           limit=max(0, kwargs['top_k'] - len(first)))
    assert result[:len(first)] == first
    return result, dict(first=old, tail=tail, added_bytes=old['added_bytes'] + tail['added_bytes'])


def main(a):
    if a.output.exists():
        raise FileExistsError(a.output)
    csv.field_size_limit(32 * 1024 * 1024)
    with a.gold_csv.open() as f:
        gold = list(csv.DictReader(f))
    rows = []
    for run in a.runs:
        if (run / 'exit-code').read_text().strip() != '0':
            raise ValueError('Require completed run')
        rs = {r['search_id']: r for r in records(run / 'completed/data/results/personamem-v2-32k.jsonl')}
        raw = {}
        for event in records(run / 'server-memory-debug.jsonl'):
            raw.setdefault(event['user_id'], {}).update({audit.identity(s): s for s in event['source_messages']})
        for t in records(run / 'server-search-debug.jsonl'):
            check_versions(t['versions'], 'source trace')
            assert evidence_packet.digest(t['returned']) == t['packet_hash']
            ctx = t['cascade']['conversation_context']
            base = [dict(p, packet_hash=ctx['base_packet_hash']) for p in t['returned'][:len(ctx['base_packet_ids'])]]
            assert evidence_packet.digest(base) == ctx['base_packet_hash']
            with patch.multiple(config, **t['versions']['settings']):
                pool = {c['id']: c for c in search_pipeline._foresight_filter(t['ranked'], copy.deepcopy(t['plan']))
                        if c.get('view_status') != 'stale' and '_packet_item' in c}
                fine = [pool[mid] for mid in t['cascade']['fine']['selected_ids']
                        if mid in pool and mid not in t['selection']['selected_ids']]
                kwargs = dict(user_id=t['user_id'], top_k=t['top_k'], byte_budget=t['coverage_manifest']['token_budget'],
                              reserve_bytes=5120, query=t['query'], options=t['options'])
                original, old = lexical(base, fine, **kwargs)
                proposed, trace = fill(base, fine, **kwargs)
                # Reproduce actual source trial's distinct reserve configuration.
                if config.PACKET_FINE_CONTEXT_LEXICAL:
                    actual = original
                elif config.PACKET_FINE_CONTEXT_RESERVE:
                    actual, _ = reserve(base, fine, user_id=t['user_id'], top_k=t['top_k'],
                                        byte_budget=t['coverage_manifest']['token_budget'],
                                        reserve_bytes=5120 if config.PACKET_FINE_CONTEXT_5K else 4096)
                else:
                    actual = base
                digest = evidence_packet.digest(actual)
                assert [dict(p, packet_hash=digest) for p in actual] == t['returned']
            assert proposed[:len(original)] == original
            assert trace['added_bytes'] <= 5120
            qa = rs[t['search_id']]['qa_id']
            snippet = gold[int(qa)]['related_conversation_snippet']
            try:
                snippet = json.loads(snippet)
            except ValueError:
                snippet = ast.literal_eval(snippet)
            scores, catalog_bytes = {}, {}
            for name, packet in (('original', original), ('fill', proposed)):
                cards = [dict(s, text=s.get('content', ''), id=f'{i}:{j}') for i, p in enumerate(packet)
                         for j, s in enumerate(p.get('sources', []))]
                scores[name] = {role: [v['full'] for v in audit.turn_coverage(
                    [audit.norm(s['content']) for s in snippet if s['role'] == role],
                    cards, raw[t['user_id']], role=role)] for role in ('user', 'assistant')}
                digest = evidence_packet.digest(packet)
                catalogs = answer_choice.build_catalog([dict(p, packet_hash=digest) for p in packet])
                catalog_bytes[name] = sum(len(c['text'].encode()) for catalog in catalogs for c in catalog.values())
            rows.append(dict(run=str(run), qa_id=qa, scores=scores, proposal=trace,
                             ids={n: [p['id'] for p in items] for n, items in (('original', original), ('fill', proposed))},
                             catalog_bytes=catalog_bytes))
    summaries = []
    for run in a.runs:
        part = [r for r in rows if r['run'] == str(run)]
        summaries.append(dict(run=str(run), questions=len(part), actual_packets_verified=len(part),
            references={n: sum(any(r['scores'][n]['user']) for r in part) for n in ('original', 'fill')},
            user_turns={n: sum(sum(r['scores'][n]['user']) for r in part) for n in ('original', 'fill')},
            assistant_turns={n: sum(sum(r['scores'][n]['assistant']) for r in part) for n in ('original', 'fill')},
            catalog_bytes={n: sum(r['catalog_bytes'][n] for r in part) for n in ('original', 'fill')},
            gained=[r['qa_id'] for r in part if any(r['scores']['fill']['user']) and not any(r['scores']['original']['user'])],
            turn_regressions=[r['qa_id'] for r in part if any(x and not y for role in ('user', 'assistant')
                for x, y in zip(r['scores']['original'][role], r['scores']['fill'][role]))],
            appended_items=sum(len(r['proposal']['tail']['appended_ids']) for r in part),
            appended_bytes=sum(r['proposal']['tail']['added_bytes'] for r in part)))
    a.output.write_text(json.dumps(dict(summary=summaries, rows=rows, limitation=__doc__), ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gold-csv', type=Path, default=Path(__file__).resolve().parents[1] / 'data/raw/personamem-v2.csv')
    main(p.parse_args())
