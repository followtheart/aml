"""Pack fixed coarse-review proposals assuming every judgment is positive.

Preserve the actual packet byte-for-byte as a prefix. Append only exact visible
source units using remaining packet capacity and enforce dependency closure.
This is an optimistic bound, not actual model relevance or answer accuracy.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
from unittest.mock import patch


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def records(path):
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main(args):
    run=args.run.resolve();root=run.parent
    plan=json.loads((root/'plan.json').read_text())
    preflight=json.loads(args.preflight.read_text())
    assert preflight['summary']['mode']=='separate-batch'
    assert preflight['summary']['questions']==plan['questions']
    for item in preflight['inputs']:
        assert Path(item['path']).stat().st_size==item['prefix_bytes']
        assert sha(Path(item['path']))==item['prefix_sha256']
    for name,digest in preflight['dependencies_sha256'].items():assert sha(Path(name))==digest,name
    for name,digest in plan['source_files'].items():assert sha(root/'source'/name)==digest,name
    snapshot=root/'source/memory_system'
    sys.path.insert(0,str(snapshot));sys.path.insert(0,str(snapshot/'scripts'))
    from app import fine_context_reserve,fine_evidence,evidence_packet
    from evaluate_recall_context import reference_turns,packet_coverage
    from audit_https_funnel import turn_coverage
    traces={t['search_id']:t for t in records(run/'server-search-debug.jsonl')}
    fixed=[]
    with patch('socket.socket.connect',side_effect=AssertionError('Offline capacity bound')):
        for row in preflight['rows']:
            t=traces[row['search_id']];lookup={c['id']:c for c in t['ranked']}
            packed=t['returned'];assert evidence_packet.digest(packed)==t['packet_hash']
            used=sum(len(p['content'].encode()) for p in packed)+max(0,len(packed)-1)
            byte_limit=t['coverage_manifest']['token_budget']
            assert used==t['coverage_manifest']['token_upper_bound']<=byte_limit
            candidates=[lookup[mid] for mid in row['proposed_ids']]
            candidate_packet,trace=fine_context_reserve.propose(packed,candidates,
                user_id=t['user_id'],top_k=t['top_k'],byte_budget=byte_limit,
                limit=4,reserve_bytes=byte_limit-used)
            before_ids={p['id'] for p in packed};blocked=set()
            while True:
                ids={p['id'] for p in candidate_packet};new_blocked=set()
                for item in candidate_packet[len(packed):]:
                    group,missing=fine_evidence.closure(lookup[item['id']],lookup)
                    if missing or not group<=ids:new_blocked.add(item['id'])
                if not new_blocked:break
                blocked.update(new_blocked)
                candidate_packet=[p for p in candidate_packet if p['id'] not in new_blocked]
            assert candidate_packet[:len(packed)]==packed
            assert len(candidate_packet)<=t['top_k']
            total=sum(len(p['content'].encode()) for p in candidate_packet)+max(0,len(candidate_packet)-1)
            assert total<=byte_limit
            assert all(p['personal_evidence'] is False and not p['coverage_ids']
                       for p in candidate_packet[len(packed):])
            fixed.append((row,t,candidate_packet,dict(added_ids=[p['id'] for p in candidate_packet if p['id'] not in before_ids],
                dependency_blocked_ids=sorted(blocked),added_bytes=total-used,total_bytes=total,byte_limit=byte_limit,
                proposal_trace=trace)))
    # Candidate membership, capacity decisions and dependency checks precede gold.
    manifest=json.loads((run/'isolated-manifest.json').read_text());prepared=Path(manifest['data'])
    assert sha(prepared)==manifest['data_sha256']==preflight['prepared_sha256']
    conversations={c['conversation_id']:c for c in records(prepared)}
    csv.field_size_limit(32*1024*1024)
    with args.gold_csv.open() as stream:gold=list(csv.DictReader(stream))
    raw={}
    with sqlite3.connect((run/'memory.db').as_uri()+'?mode=ro',uri=True) as db:
        db.row_factory=sqlite3.Row
        for source in db.execute('SELECT * FROM source_messages'):
            raw.setdefault(source['user_id'],{})[source['request_id'],source['message_index']]=dict(source)
    rows=[]
    for proposal,t,packet,detail in fixed:
        conv=conversations[proposal['conversation_id']]
        question=next(q for q in conv['qa'] if q['id']==proposal['qa_id'])
        assert t['query']==question['question'] and t['options']==question['options']
        turns=reference_turns(question,conv['source_history'],gold[int(proposal['qa_id'])])
        actual=packet_coverage(turns,t['returned'],raw[t['user_id']])
        assert actual==proposal['actual_final']
        bound=packet_coverage(turns,packet,raw[t['user_id']])
        def flags(items):
            cards=[dict(s,id=f'{i}:{j}',text=s.get('content','')) for i,p in enumerate(items)
                   for j,s in enumerate(p.get('sources',[]))]
            return [m['full'] for m in turn_coverage(turns,cards,raw[t['user_id']])]
        old,new=flags(t['returned']),flags(packet)
        assert len(old)==len(new) and all(not a or b for a,b in zip(old,new))
        rows.append(dict(qa_id=proposal['qa_id'],actual=actual,packed_upper_bound=bound,
                         original_turn_flags=old,proposed_turn_flags=new,**detail))
    summary=dict(questions=len(rows),provider_calls=0,unchanged_packet_prefixes=len(rows),
        added_items=sum(len(r['added_ids']) for r in rows),added_bytes=sum(r['added_bytes'] for r in rows),
        reference_questions={a:sum(r[a]['visible'] for r in rows) for a in ('actual','packed_upper_bound')},
        reference_turns={a:sum(r[a]['full_turns'] for r in rows) for a in ('actual','packed_upper_bound')},
        gained=[r['qa_id'] for r in rows if r['packed_upper_bound']['visible'] and not r['actual']['visible']],
        gained_turn_questions=[r['qa_id'] for r in rows if r['packed_upper_bound']['full_turns']>r['actual']['full_turns']])
    report=dict(summary=summary,rows=rows,limitation=__doc__,inputs={str(p.resolve()):sha(p) for p in
        (args.preflight,prepared,args.gold_csv,run/'memory.db',root/'plan.json',Path(__file__))})
    with args.output.open('x') as stream:json.dump(report,stream,indent=2);stream.write('\n')
    print(json.dumps(summary))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--preflight',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gold-csv',type=Path,default=Path(__file__).resolve().parents[1]/'data/raw/personamem-v2.csv')
    main(parser.parse_args())
