"""Pack frozen capacity-probe selections; never infer answer accuracy."""
import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import evidence_packet, answer_context


def pack(search, ids, groups, status=None):
    indexed = {c['id']: c for c in search['ranked']}
    items = []
    for mid in ids:
        c = indexed[mid]
        item = copy.deepcopy(c['_packet_item'])
        kind = c.get('_score_kind', 'unknown')
        if status is not None and c.get('_final') is None and kind != 'constraint':
            kind = 'graph_fallback' if status == 'fallback' else 'listwise'
        item.update(equivalent_ids=c.get('_equivalent_ids', []), score=c.get('_final'),
                    score_kind=kind, _fused=c.get('_fused', 0), _selection_bucket=None)
        items.append(item)
    return evidence_packet.pack_ranked(items, search['top_k'], search['evidence_token_budget'], groups=groups)


def main(args):
    base = args.run / 'completed'
    results = {r['qa_id']: r for r in map(json.loads, (base / 'data/results/personamem-v2-32k.jsonl').read_text().splitlines())}
    searches = {r['search_id']: r for r in map(json.loads, (base / 'logs/search-debug.jsonl').read_text().splitlines())}
    audits = {r['qa_id']: r for r in json.loads((args.run / 'funnel/audit.json').read_text())['rows']}
    for r in results.values():
        s = searches[r['search_id']]
        _, digest, _ = pack(s, s['selection']['selected_ids'], s['coverage_manifest']['requested_evidence_groups'])
        assert digest == s['packet_hash'], r['qa_id']
    output = []
    for row in map(json.loads, args.probe.read_text().splitlines()):
        s = searches[results[row['qa_id']]['search_id']]
        status = row.get('listwise_status', 'ok')
        if 'error' in row:
            # Only recorded structural failures have a verified fallback here.
            assert row.get('error_type') == 'ValueError' and len(row['calls']) == 2
            for call in row['calls']:
                from app.cascade_rerank import validate
                try:
                    validate(call['response'], len(row['admission']['input_ids']))
                except (ValueError, TypeError, KeyError):
                    pass
                else:
                    raise AssertionError('Unexpected valid response')
            ids, groups, status = row['admission']['input_ids'], [], 'fallback'
            indexed = {c['id']: c for c in s['ranked']}
            assert all(indexed[mid].get('_ce_score') is not None for mid in ids)
        else:
            ids, groups = row['selected_ids'], row['groups']
        from app.cascade_rerank import protected as is_protected
        indexed = {c['id']: c for c in s['ranked']}
        protected = [mid for mid in s['selection']['selected_ids'] if is_protected(indexed[mid])]
        assert not set(protected) & set(ids)
        ids = protected + ids
        packed, digest, manifest = pack(s, ids, groups, status)
        context = answer_context.build(packed)
        assert len(context.encode()) <= s['evidence_token_budget']
        output.append(dict(qa_id=row['qa_id'], mode=row['mode'], listwise_status=status,
            packet_hash=digest, selected_ids=ids, packed_ids=[p['id'] for p in packed],
            context_bytes=len(context.encode()), manifest=manifest,
            probes=[dict(label=p['label'], quote=p['quote'], visible=p['quote'] in context)
                    for p in audits[row['qa_id']]['probes']]))
    report = dict(baseline_hashes_verified=len(results), rows=output,
        limitation='Frozen candidate units and live probe ordering, offline packing only. Does not rerun scope filters, dynamic upstream budgets or answer judging.')
    with args.output.open('x') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--probe', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    main(p.parse_args())
