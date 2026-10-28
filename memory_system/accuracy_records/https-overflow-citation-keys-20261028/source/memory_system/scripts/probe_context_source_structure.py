"""Offline only: compare structural source trimming with the exact reserve baseline.

No labels enter proposals. Leading-assistant removal is a diagnostic, not a
safe assumption that a user's next message is independent of that assistant.
"""
import argparse
import ast
import copy
import csv
import json
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import config,evidence_packet,search_pipeline
from app.conversation_context import source_key,usable
from app.fine_context_reserve import propose
import audit_https_funnel as audit
from evaluation_model_guard import check_versions

MODES=('original','deduplicate','leading_assistant','reserve_5120')


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def proposal(base,fine,*,user_id,top_k,byte_budget,mode):
    result=copy.deepcopy(base)
    added_bytes=0
    decisions=[]
    for candidate in fine:
        candidate=copy.deepcopy(candidate)
        item=candidate.get('_packet_item',{})
        sources=item.get('sources',[])
        # Do not make an invalid original candidate valid by deleting bad sources.
        if any(not usable(s,user_id) or s.get('role') not in ('user','assistant')
               or s.get('content','') not in item.get('content','') for s in sources):
            continue
        known={source_key(s) for p in result for s in p.get('sources',[])}
        first_user={}
        for s in sources:
            if s['role']=='user':
                rid=s['request_id']
                first_user[rid]=min(first_user.get(rid,s['message_index']),s['message_index'])
        def keep(s):
            if mode=='deduplicate':return source_key(s) not in known
            if mode=='leading_assistant':
                return not (s['role']=='assistant' and s['request_id'] in first_user
                            and s['message_index']<first_user[s['request_id']])
            raise ValueError(mode)
        retained=[s for s in sources if keep(s)]
        removed=[s for s in sources if not keep(s)]
        item['sources']=retained
        new,trace=propose(result,[candidate],user_id=user_id,top_k=top_k,byte_budget=byte_budget,
                          limit=max(0,2-(len(result)-len(base))),reserve_bytes=4096-added_bytes)
        decisions.append(dict(candidate_id=candidate['id'],removed_sources=removed,
                              retained_sources=retained,decision=trace['proposed']))
        result=new
        added_bytes+=trace['added_bytes']
    return result,dict(added_bytes=added_bytes,decisions=decisions)


def main(args):
    if args.output.exists():raise FileExistsError(args.output)
    csv.field_size_limit(32*1024*1024)
    with args.gold_csv.open() as f:gold=list(csv.DictReader(f))
    rows=[]
    for run in args.runs:
        if (run/'exit-code').read_text().strip()!='0':raise ValueError('Incomplete run')
        results={r['search_id']:r for r in records(run/'completed/data/results/personamem-v2-32k.jsonl')}
        raw={}
        for event in records(run/'server-memory-debug.jsonl'):
            raw.setdefault(event['user_id'],{}).update({audit.identity(s):s for s in event['source_messages']})
        for t in records(run/'server-search-debug.jsonl'):
            check_versions(t['versions'],'source trace')
            if evidence_packet.digest(t['returned'])!=t['packet_hash']:raise ValueError('Final hash')
            ctx=t['cascade']['conversation_context']
            base=[dict(p,packet_hash=ctx['base_packet_hash']) for p in t['returned'][:len(ctx['base_packet_ids'])]]
            if evidence_packet.digest(base)!=ctx['base_packet_hash']:raise ValueError('Base hash')
            with patch.multiple(config,**t['versions']['settings']):
                pool={c['id']:c for c in search_pipeline._foresight_filter(t['ranked'],copy.deepcopy(t['plan']))
                      if c.get('view_status')!='stale' and '_packet_item' in c}
                fine=[pool[mid] for mid in t['cascade']['fine']['selected_ids']
                      if mid in pool and mid not in t['selection']['selected_ids']]
                kwargs=dict(user_id=t['user_id'],top_k=t['top_k'],byte_budget=t['coverage_manifest']['token_budget'])
                original,details=propose(base,fine,**kwargs)
                packets={'base':base,'original':original}
                proposals={'original':details}
                for mode in ('deduplicate','leading_assistant'):
                    packets[mode],proposals[mode]=proposal(base,fine,mode=mode,**kwargs)
                packets['reserve_5120'],proposals['reserve_5120']=propose(base,fine,reserve_bytes=5120,**kwargs)
            # Gold is read only after all proposals are fixed.
            qa=results[t['search_id']]['qa_id']
            snippet=gold[int(qa)]['related_conversation_snippet']
            try:snippet=json.loads(snippet)
            except ValueError:snippet=ast.literal_eval(snippet)
            scores={}
            for mode,packet in packets.items():
                cards=[dict(s,text=s.get('content',''),id=f'{i}:{j}') for i,p in enumerate(packet)
                       for j,s in enumerate(p.get('sources',[]))]
                scores[mode]={role:[c['full'] for c in audit.turn_coverage(
                    [audit.norm(s['content']) for s in snippet if s['role']==role],cards,raw[t['user_id']],role=role)]
                    for role in ('user','assistant')}
            source_sets={mode:{source_key(s) for p in packet for s in p.get('sources',[])}
                         for mode,packet in packets.items()}
            rows.append(dict(run=str(run),qa_id=qa,scores=scores,proposals=proposals,
                original_ids=[p['id'] for p in original[len(base):]],
                lost_original_sources={m:len(source_sets['original']-ss) for m,ss in source_sets.items()},
                ids={m:[p['id'] for p in pck[len(base):]] for m,pck in packets.items()}))
    summaries=[]
    for run in args.runs:
        subset=[r for r in rows if r['run']==str(run)]
        for mode in MODES:
            def full(r,m):return all(r['scores'][m]['user']+r['scores'][m]['assistant']) and bool(r['scores'][m]['user']+r['scores'][m]['assistant'])
            summaries.append(dict(run=str(run),mode=mode,questions=len(subset),
                reference_questions=sum(any(r['scores'][mode]['user']) for r in subset),
                complete_all_role_sets=sum(full(r,mode) for r in subset),
                complete_user_sets=sum(bool(r['scores'][mode]['user']) and all(r['scores'][mode]['user']) for r in subset),
                user_turns=sum(sum(r['scores'][mode]['user']) for r in subset),
                assistant_turns=sum(sum(r['scores'][mode]['assistant']) for r in subset),
                gained=[r['qa_id'] for r in subset if any(r['scores'][mode]['user']) and not any(r['scores']['original']['user'])],
                lost=[r['qa_id'] for r in subset if any(r['scores']['original']['user']) and not any(r['scores'][mode]['user'])],
                turn_regressions=[r['qa_id'] for r in subset if any(a and not b for role in ('user','assistant')
                    for a,b in zip(r['scores']['original'][role],r['scores'][mode][role]))],
                lost_original_sources=sum(r['lost_original_sources'][mode] for r in subset),
                added_bytes=sum(r['proposals'][mode]['added_bytes'] for r in subset)))
    args.output.write_text(json.dumps(dict(summary=summaries,rows=rows,limitation=__doc__),ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summaries,ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gold-csv',type=Path,default=Path(__file__).resolve().parents[1]/'data/raw/personamem-v2.csv')
    main(p.parse_args())
