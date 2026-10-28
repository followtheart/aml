"""Fixed real CE scores: diagnose bounded new-coarse participation in the fine core.

Offline proposal and main admission replay. It does not replay model judgments,
packet packing, or the reader. References are used only by the scorer.
"""
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import sqlite3
from unittest.mock import patch

from evaluate_recall_context import CSV, packet_coverage, records, reference_turns
from app import cascade_rerank, config, fine_evidence, schemas


def admit(t, fine, selection, *, prioritize_soft=True, with_omissions=False):
    """Replay existing whole-request admission; prove the historical branch first."""
    req=schemas.SearchRequest(user_id=t['user_id'],query=t['query'],options=t['options'])
    plan=copy.deepcopy(t['plan'])
    plan['_coverage_requirements']=[{k:r[k] for k in ('id','text','origin') if k in r}
        for r in t['coverage_manifest']['coverage']['requirements']]
    plan['_fusion_edges']=t['fusion_edges']
    plan['_fusion_triples']=t['fusion_triples']
    plan['_fine_path_groups']=selection['adaptive']['path_groups']
    headroom=t['cascade']['admission_budget']['token_headroom']
    positions={c['id']:len(fine)-i for i,c in enumerate(fine)}
    reservations=[r for r in selection['reservations'] if r['selected']]
    carried={r['candidate_id'] for r in reservations}
    soft={r['candidate_id'] for r in reservations if r['reasons']==['fused_head']}
    mandatory=carried if prioritize_soft else carried-soft
    ordered=cascade_rerank.shortlist(fine,config.CASCADE_LLM_LIMIT,[],
        lambda c:(c['id'] in carried and c['id'] not in soft,positions[c['id']]),carry_ids=carried)
    def fits(request):
        return len(request['prompt'].encode())<=config.RERANK_MAX_PROMPT_BYTES and (
            headroom is None or request['reservation']<=headroom)
    submitted=[]; omissions=[]
    for c in sorted(ordered,key=lambda c:c['id'] not in mandatory):
        request=cascade_rerank.listwise_request(req,plan,submitted+[c],
            shared_sources=None if config.FINE_EVIDENCE_ADAPTIVE else False)
        if fits(request):submitted.append(c)
        else:omissions.append(dict(id=c['id'],stage='listwise',reason=(
            'prompt_budget' if len(request['prompt'].encode())>config.RERANK_MAX_PROMPT_BYTES else 'token_budget')))
    baseline_ids={c['id'] for c in submitted}
    if config.LISTWISE_SHARED_SOURCES and submitted:
        for c in sorted(ordered,key=lambda c:c['id'] not in mandatory):
            if c['id'] in baseline_ids:continue
            trial=sorted(submitted+[c],key=lambda c:-positions[c['id']])
            if fits(cascade_rerank.listwise_request(req,plan,trial)):submitted.append(c)
    submitted,decisions=fine_evidence.admit_additions(submitted,fine,selection['adaptive']['additions'],
        count_limit=config.CASCADE_LLM_LIMIT+config.FINE_EXTRA_LIMIT,byte_limit=config.RERANK_MAX_PROMPT_BYTES,
        token_headroom=headroom,request_for=lambda items:cascade_rerank.listwise_request(req,plan,
            sorted(items,key=lambda c:-positions[c['id']])))
    submitted.sort(key=lambda c:-positions[c['id']])
    submitted_ids={c['id'] for c in submitted}
    omissions=[o for o in omissions if o['id'] not in submitted_ids]
    reasons={mid:d['reason'] for d in decisions if not d['admitted'] for mid in d['group_ids']}
    omissions.extend(dict(id=c['id'],stage='listwise',reason=reasons.get(c['id'],'listwise_limit'))
                     for c in fine if c['id'] not in submitted_ids and c not in ordered)
    request=cascade_rerank.listwise_request(req,plan,submitted)
    if with_omissions:return submitted,len(request['prompt'].encode()),omissions
    return submitted,len(request['prompt'].encode())


def main(args):
    traces = {t['search_id']:t for t in records(args.run/'server-search-debug.jsonl')}
    conversations = {r['conversation_id']:r for r in records(args.prepared)}
    csv.field_size_limit(32*1024*1024)
    with CSV.open() as source:
        gold = list(csv.DictReader(source))
    raw = {}
    with sqlite3.connect((args.run/'memory.db').resolve().as_uri()+'?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        for row in db.execute('SELECT * FROM source_messages'):
            raw.setdefault(row['user_id'], {})[row['request_id'],row['message_index']] = dict(row)
    rows = []
    for result in records(args.run/'completed/data/results/personamem-v2-32k.jsonl'):
        t=traces[result['search_id']]; c=t['cascade']
        if c['cross_encoder']['status']!='ok' or not t['versions']['settings']['COARSE_EVIDENCE_EXTEND']:
            raise ValueError('Requires an extended coarse pool and complete real CE')
        by_id={v['id']:v for v in t['ranked']}
        coarse=[by_id[mid] for mid in c['coarse']['selected_ids']]
        baseline_ids=set(c['coarse']['baseline_ids'])
        extra=sorted([v for v in coarse if v['id'] not in baseline_ids and v.get('_ce_score') is not None],key=fine_evidence.key,reverse=True)
        allowed_core=baseline_ids | {v['id'] for v in extra[:args.new_core_limit]}
        requirements=t['coverage_manifest']['coverage']['requirements']
        with patch.multiple(config, **t['versions']['settings']), patch('socket.socket.connect',side_effect=AssertionError('Offline')):
            if not config.FINE_EVIDENCE_ADAPTIVE or config.FINE_TOPIC_CONTEXT:
                raise ValueError('Requires adaptive fine and unchanged topic context policy')
            fused={v['id'] for v in sorted(coarse,key=lambda v:-(v.get('_fused') or 0))[:config.CASCADE_FUSED_RESERVE]}
            kw=dict(fused_ids=fused,extra_limit=config.FINE_EXTRA_LIMIT,
                    soft_promotion_limit=1 if config.FINE_WEAK_PROMOTION_ONE else None)
            baseline_pool=[v for v in coarse if v['id'] in baseline_ids]
            original_fused={v['id'] for v in sorted(baseline_pool,key=lambda v:-(v.get('_fused') or 0))[:config.CASCADE_FUSED_RESERVE]}
            variant_kw=dict(kw,fused_ids=original_fused)
            historical_allowed=(baseline_ids | {v['id'] for v in extra[:1]}
                                if t['versions']['settings'].get('FINE_COARSE_CORE_QUOTA') else None)
            original,old=fine_evidence.propose(coarse,config.CASCADE_FINE_LIMIT,requirements,
                core_pool_ids=historical_allowed,**(variant_kw if historical_allowed is not None else kw))
            if [v['id'] for v in original]!=c['fine']['selected_ids']:
                raise ValueError('Actual expanded-pool fine did not reproduce: '+result['qa_id'])
            select=fine_evidence.selection.select
            def preserve_core(candidates,count,needs,key,*a,**kwargs):
                return select([v for v in candidates if v['id'] in allowed_core],count,needs,key,*a,**kwargs)
            with patch.object(fine_evidence.selection,'select',side_effect=preserve_core):
                proposed,new=fine_evidence.propose(coarse,config.CASCADE_FINE_LIMIT,requirements,**variant_kw)
            implemented,implemented_trace=fine_evidence.propose(coarse,config.CASCADE_FINE_LIMIT,requirements,
                core_pool_ids=allowed_core,**variant_kw)
            assert [v['id'] for v in implemented]==[v['id'] for v in proposed]
            proposed,new=implemented,implemented_trace
            original_input,original_bytes=admit(t,original,old)
            if ([v['id'] for v in original_input]!=c['listwise']['candidate_ids']
                    or original_bytes!=c['listwise']['input_bytes']):
                raise ValueError('Actual main admission did not reproduce: '+result['qa_id'])
            proposed_input,proposed_bytes=admit(t,proposed,new)
            assert len(proposed)<=old['adaptive']['capacity']
            assert [v['id'] for v in proposed][:len(new['adaptive']['core_ids'])]==new['adaptive']['core_ids']
        conv=conversations[result['conversation_id']]
        q=next(q for q in conv['qa'] if q['id']==result['qa_id'])
        assert t['query']==q['question'] and t['options']==q['options']
        turns=reference_turns(q,conv['source_history'],gold[int(result['qa_id'])])
        def coverage(items):
            return packet_coverage(turns,[v['_packet_item'] for v in items],raw[t['user_id']])
        rows.append(dict(qa_id=result['qa_id'],control=coverage(original),proposed=coverage(proposed),
            control_input=coverage(original_input),proposed_input=coverage(proposed_input),
            control_input_ids=[v['id'] for v in original_input],proposed_input_ids=[v['id'] for v in proposed_input],
            control_input_bytes=original_bytes,proposed_input_bytes=proposed_bytes,
            control_ids=[v['id'] for v in original],proposed_ids=[v['id'] for v in proposed],
            preserved_core_ids=new['adaptive']['core_ids']))
    summary=dict(questions=len(rows),new_core_limit=args.new_core_limit,reproduced_actual_fine=len(rows),reproduced_actual_main_input=len(rows),
        visible={k:sum(r[k]['visible'] for r in rows) for k in ('control','proposed','control_input','proposed_input')},
        turns={k:sum(r[k]['full_turns'] for r in rows) for k in ('control','proposed','control_input','proposed_input')},
        gained=[r['qa_id'] for r in rows if r['proposed']['visible'] and not r['control']['visible']],
        input_gained=[r['qa_id'] for r in rows if r['proposed_input']['visible'] and not r['control_input']['visible']],
        input_lost=[r['qa_id'] for r in rows if r['control_input']['visible'] and not r['proposed_input']['visible']],
        lost=[r['qa_id'] for r in rows if r['control']['visible'] and not r['proposed']['visible']],
        candidates={k:sum(len(r[k+'_ids']) for r in rows) for k in ('control','proposed','control_input','proposed_input')})
    with args.output.open('x') as out:
        json.dump(dict(summary=summary,rows=rows,limitation=__doc__,
            trace_sha256=hashlib.sha256((args.run/'server-search-debug.jsonl').read_bytes()).hexdigest()),out,indent=2)
    print(json.dumps(summary))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--prepared',type=Path,default=Path('memory_system/data/prepared/personamem-v2-32k.jsonl'))
    p.add_argument('--new-core-limit',type=int,choices=[0,1,2,4,8],default=0)
    p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
