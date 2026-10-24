"""Compare production source-seat proposals with the frozen offline diagnostic."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import config, evidence_packet, search_pipeline, source_query_context
from probe_source_query_reserve import nominees, proposal
from probe_reserve_capacity import records


def main(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = []
    for run in args.runs:
        for t in records(run/'server-search-debug.jsonl'):
            with patch.multiple(config, **t['versions']['settings']), patch('socket.socket.connect', side_effect=AssertionError('Offline')):
                ctx = t['cascade']['conversation_context']
                base = [dict(p, packet_hash=ctx['base_packet_hash']) for p in t['returned'][:len(ctx['base_packet_ids'])]]
                assert evidence_packet.digest(base) == ctx['base_packet_hash']
                pool = {c['id']:c for c in search_pipeline._foresight_filter(copy.deepcopy(t['ranked']),copy.deepcopy(t['plan']))
                        if c.get('view_status')!='stale' and '_packet_item' in c}
                fine = [pool[mid] for mid in t['cascade']['fine']['selected_ids']
                        if mid in pool and mid not in t['selection']['selected_ids']]
                diagnostic, old = nominees(t, pool, True)
                champions, actual = source_query_context.nominate(pool, routes=t['routes'], query_specs=t['query_specs'],
                    query=t['query'], options=t['options'], coarse_ids=t['cascade']['coarse']['selected_ids'],
                    fine_ids=t['cascade']['fine']['selected_ids'], selected_ids=t['selection']['selected_ids'],
                    fine_selection=t['cascade']['fine']['selection'])
                assert diagnostic == champions
                for key in ('direct_queries', 'nominations', 'rejected', 'max_ce_rank'):
                    assert actual[key] == old[key], key
                kw = dict(user_id=t['user_id'], top_k=t['top_k'], byte_budget=t['coverage_manifest']['token_budget'],
                          reserve_bytes=5120, query=t['query'], options=t['options'])
                expected, expected_trace = proposal(base, fine, diagnostic, **kw)
                got, got_trace = source_query_context.propose(base, fine, champions=champions, **kw)
                assert got == expected
                for key in ('source_seat','remainder','added_bytes'):
                    assert got_trace[key] == expected_trace[key], key
                assert got_trace['added_bytes'] <= 5120
                rows.append(dict(run=str(run), search_id=t['search_id'], source_ids=got_trace['source_seat']['appended_ids'],
                                 packet_hash=evidence_packet.digest(got)))
    result = dict(packets_verified=len(rows), source_seats=sum(bool(r['source_ids']) for r in rows), rows=rows,
        diagnostic_sha256=hashlib.sha256(Path(__file__).with_name('probe_source_query_reserve.py').read_bytes()).hexdigest(),
        production_sha256=hashlib.sha256(Path(source_query_context.__file__).read_bytes()).hexdigest())
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
