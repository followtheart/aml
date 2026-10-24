"""Inventory missing references and whole-source capacity after complete HTTPS.

Reference IDs are used only for offline diagnostics. A carrier identity is not
proof of full visible reference coverage, relevance, entailment or user fact.
No rule proposed here is permitted to admit candidates using reference IDs.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch


def records(path):return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def main(root):
    if (root/'exit-code').read_text().strip()!='0':raise ValueError('Incomplete run')
    sys.path.insert(0,str(root.resolve()/'source/memory_system'))
    from app import config,evidence_packet,search_pipeline
    from app.fine_context_reserve import propose
    from app.fine_context_lexical import source_scores
    rows=[]
    for arm in ('control','treatment'):
        run=root/arm
        stages=json.loads((run/'funnel/stage-transport.json').read_text())
        if stages['summary']['progress_snapshot']:raise ValueError('Partial funnel')
        by_stage={r['qa_id']:r for r in stages['rows']}
        results={r['search_id']:r for r in records(run/'completed/data/results/personamem-v2-32k.jsonl')}
        for t in records(run/'server-search-debug.jsonl'):
            r=results[t['search_id']];s=by_stage[r['qa_id']]
            if not s['raw_full_turns'] or s['visible']['packet']['full_turns']:continue
            assert evidence_packet.digest(t['returned'])==t['packet_hash']
            carriers=set().union(*(set(ids) for ids in s['carrier_ids'].values()))
            selection=t['cascade']['fine']['selection']
            decisions={d['id']:d for d in selection['decisions']}
            context=t['cascade']['conversation_context']
            remaining=context['reserve_bytes']-context['added_bytes']
            with patch.multiple(config,**t['versions']['settings']),patch('socket.socket.connect',side_effect=AssertionError('Offline')):
                pool={c['id']:c for c in search_pipeline._foresight_filter(copy.deepcopy(t['ranked']),copy.deepcopy(t['plan']))
                      if c.get('view_status')!='stale' and '_packet_item' in c}
                fine=[pool[mid] for mid in t['cascade']['fine']['selected_ids']
                      if mid in pool and mid not in t['selection']['selected_ids']]
                scores,matches=source_scores(fine,user_id=t['user_id'],query=t['query'],options=t['options'])
                lexical=sorted([c['id'] for c in fine if scores.get(c['id'],0)>0],key=lambda mid:-scores[mid])
                targets=[]
                for mid in sorted(carriers&pool.keys()):
                    _,capacity=propose(t['returned'],[pool[mid]],user_id=t['user_id'],top_k=t['top_k'],
                        byte_budget=t['coverage_manifest']['token_budget'],reserve_bytes=remaining,limit=1)
                    targets.append(dict(candidate_id=mid,ce_score=pool[mid].get('_ce_score'),
                        fine_decision=decisions.get(mid),in_fine=mid in t['cascade']['fine']['selected_ids'],
                        originally_selected=mid in t['selection']['selected_ids'],
                        lexical_rank=lexical.index(mid)+1 if mid in lexical else None,
                        lexical_score=scores.get(mid),lexical_matches=matches.get(mid),
                        tail_capacity=capacity,
                        actual_reserve_decisions=[d for d in context['proposed'] if d['candidate_id']==mid]))
            rows.append(dict(arm=arm,qa_id=r['qa_id'],score=r['score'],query=t['query'],
                stage_reference_turns={k:v['full_turns'] for k,v in s['visible'].items()},
                remaining_reserve_bytes=remaining,top_k=t['top_k'],returned_count=len(t['returned']),
                candidates=targets))
    summary={arm:dict(missing_questions=[r['qa_id'] for r in rows if r['arm']==arm],
        fine_present_questions=[r['qa_id'] for r in rows if r['arm']==arm and r['stage_reference_turns']['fine']],
        fine_carrier_can_fit_remaining=[r['qa_id'] for r in rows if r['arm']==arm and
            any(c['in_fine'] and not c['originally_selected'] and c['tail_capacity']['appended_ids'] for c in r['candidates'])])
        for arm in ('control','treatment')}
    output=root/'missing-context-inventory.json'
    with output.open('x') as f:
        json.dump(dict(summary=summary,rows=rows,limitation=__doc__,
            auditor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),f,ensure_ascii=False,indent=2)
        f.write('\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    main(p.parse_args().run)
