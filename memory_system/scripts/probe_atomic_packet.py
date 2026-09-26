"""Offline fixed-ranking replay for atomic-group packing; no answer score inferred."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import evidence_packet, answer_context


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--traces', required=True)
    ap.add_argument('--candidate', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    spec = importlib.util.spec_from_file_location('app.packet_candidate', args.candidate)
    candidate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(candidate)
    rows = []
    for trace in map(json.loads, Path(args.traces).read_text().splitlines()):
        indexed = {c['id']: c for c in trace['ranked']}
        items = []
        for mid in trace['selection']['selected_ids']:
            c = indexed[mid]
            item = dict(c['_packet_item'], equivalent_ids=c.get('_equivalent_ids', []),
                        score=c.get('_final'), score_kind=c.get('_score_kind', 'unknown'),
                        _fused=c.get('_fused', 0), _selection_bucket=c.get('_selection_bucket'))
            items.append(item)
        manifest = trace['coverage_manifest']
        kwargs = dict(top_k=trace['top_k'], token_budget=trace['evidence_token_budget'],
                      groups=manifest['requested_evidence_groups'])
        old, old_hash, _ = evidence_packet.pack_ranked(items, **kwargs)
        assert old_hash == trace['packet_hash'], trace['query']
        new, new_hash, details = candidate.pack_ranked(items, **kwargs)
        assert candidate.digest(new) == new_hash
        assert len(answer_context.build(new).encode()) <= kwargs['token_budget']
        assert details['evidence_count'] <= kwargs['top_k']
        old_sources = {(s['request_id'], s['message_index']) for m in old for s in m.get('sources', []) if s.get('content')}
        new_sources = {(s['request_id'], s['message_index']) for m in new for s in m.get('sources', []) if s.get('content')}
        rows.append(dict(search_id=trace['search_id'], query=trace['query'],
                         before_hash=old_hash, after_hash=new_hash,
                         before_ids=[m['id'] for m in old], after_ids=[m['id'] for m in new],
                         added_sources=sorted(new_sources-old_sources),
                         removed_sources=sorted(old_sources-new_sources),
                         before_bytes=manifest['token_upper_bound'], after_bytes=details['token_upper_bound'],
                         manifest=details))
    output = Path(args.output)
    output.write_text(json.dumps(rows, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(questions=len(rows), baseline_hashes_matched=len(rows),
                         changed_packets=sum(r['before_hash'] != r['after_hash'] for r in rows),
                         added_sources=sum(len(r['added_sources']) for r in rows),
                         removed_sources=sum(len(r['removed_sources']) for r in rows), provider_calls=0)))

if __name__ == '__main__':
    main()
