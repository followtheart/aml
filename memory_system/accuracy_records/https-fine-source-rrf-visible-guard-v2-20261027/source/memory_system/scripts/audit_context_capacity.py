"""Isolate reserve capacity or the extra lexical seat with model outputs fixed.

Catalog bytes are measured input text, not billed reader token usage.
"""
import argparse
import ast
import copy
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys
from unittest.mock import patch

import audit_https_funnel as audit
from audit_residual_reader import citations, source_key
from evaluation_model_guard import check_versions


def records(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def main(args):
    root=args.run.resolve()
    if (root/'exit-code').read_text().strip()!='0':raise ValueError('Require successful completed run')
    snapshot=root/'source'
    sys.path.insert(0,str(snapshot/'memory_system'))
    from app import answer_choice,config,evidence_packet,search_pipeline
    from app.fine_context_reserve import propose
    variant=args.lexical or args.fill or args.source_query
    if variant:
        from app.fine_context_lexical import propose as lexical_propose
    if args.fill:
        from app.fine_context_fill import propose as fill_propose
    if args.source_query:
        from app import source_query_context
    modes=(0,1) if variant else (4096,5120)
    before,after=modes
    csv.field_size_limit(32*1024*1024)
    with args.gold_csv.open() as f:gold=list(csv.DictReader(f))
    arms={}
    for arm in ('control','treatment'):
        path=root/arm
        results={r['search_id']:r for r in records(path/'completed/data/results/personamem-v2-32k.jsonl')}
        raw={}
        for event in records(path/'server-memory-debug.jsonl'):
            raw.setdefault(event['user_id'],{}).update({audit.identity(s):s for s in event['source_messages']})
        rows=[]
        for t in records(path/'server-search-debug.jsonl'):
            check_versions(t['versions'],'source trace')
            for name,digest in t['versions']['file_hashes'].items():
                if hashlib.sha256((snapshot/name).read_bytes()).hexdigest()!=digest:raise ValueError('Frozen source mismatch')
            if evidence_packet.digest(t['returned'])!=t['packet_hash']:raise ValueError('Final digest mismatch')
            ctx=t['cascade']['conversation_context'];n=len(ctx['base_packet_ids'])
            base=[dict(p,packet_hash=ctx['base_packet_hash']) for p in t['returned'][:n]]
            if evidence_packet.digest(base)!=ctx['base_packet_hash']:raise ValueError('Base digest mismatch')
            packets={};proposals={}
            with patch.multiple(config,**t['versions']['settings']):
                pool={c['id']:c for c in search_pipeline._foresight_filter(t['ranked'],copy.deepcopy(t['plan']))
                      if c.get('view_status')!='stale' and '_packet_item' in c}
                fine=[pool[mid] for mid in t['cascade']['fine']['selected_ids']
                      if mid in pool and mid not in t['selection']['selected_ids']]
                if args.source_query:
                    champions, nomination=source_query_context.nominate(pool,
                        routes=t['routes'],query_specs=t['query_specs'],query=t['query'],options=t['options'] or (),
                        coarse_ids=t['cascade']['coarse']['selected_ids'],fine_ids=t['cascade']['fine']['selected_ids'],
                        selected_ids=t['selection']['selected_ids'],fine_selection=t['cascade']['fine']['selection'])
                for cap in modes:
                    proposer=(source_query_context.propose if args.source_query and cap else
                              fill_propose if args.fill and cap else lexical_propose if args.source_query or args.fill or args.lexical and cap else propose)
                    packet,trace=proposer(base,fine,user_id=t['user_id'],top_k=t['top_k'],
                        byte_budget=t['coverage_manifest']['token_budget'],reserve_bytes=5120 if variant else cap,
                        **({'query':t['query'],'options':t['options']} if args.source_query or args.fill or args.lexical and cap else {}),
                        **({'champions':champions} if args.source_query and cap else {}))
                    digest=evidence_packet.digest(packet)
                    packets[cap]=[dict(p,packet_hash=digest) for p in packet]
                    proposals[cap]=trace
            actual_cap=(int(t['versions']['settings']['PACKET_SOURCE_QUERY_CONTEXT']) if args.source_query else
                        int(t['versions']['settings']['PACKET_FINE_CONTEXT_FILL']) if args.fill else
                        int(t['versions']['settings']['PACKET_FINE_CONTEXT_LEXICAL']) if args.lexical else
                        5120 if t['versions']['settings']['PACKET_FINE_CONTEXT_5K'] else 4096)
            if packets[actual_cap]!=t['returned']:raise ValueError('Actual packet differs from exact capacity replay')
            r=results[t['search_id']]
            snippet=gold[int(r['qa_id'])]['related_conversation_snippet']
            try:snippet=json.loads(snippet)
            except ValueError:snippet=ast.literal_eval(snippet)
            scores={};catalog_bytes={};catalogs={}
            for cap,packet in packets.items():
                cards=[dict(s,text=s.get('content',''),id=f'{i}:{j}') for i,p in enumerate(packet) for j,s in enumerate(p['sources'])]
                scores[cap]={role:[v['full'] for v in audit.turn_coverage(
                    [audit.norm(s['content']) for s in snippet if s['role']==role],cards,raw[t['user_id']],role=role)]
                    for role in ('user','assistant')}
                catalogs[cap]=answer_choice.build_catalog(packet)
                catalog_bytes[cap]=sum(len(c['text'].encode()) for catalog in catalogs[cap] for c in catalog.values())
            actual_catalog=catalogs[actual_cap]
            if (list(actual_catalog[0].values())!=r['answer_source_catalog'] or
                    list(actual_catalog[1].values())!=r['answer_constraint_catalog']):
                raise ValueError('Catalog does not reproduce actual reader input')
            source_details={}
            if args.source_query:
                deltas=[]
                for index in (0,1):
                    old_keys={source_key(c) for c in catalogs[before][index].values()}
                    new_keys={source_key(c) for c in catalogs[after][index].values()}
                    deltas.append(dict(added={sid:c for sid,c in catalogs[after][index].items() if source_key(c) not in old_keys},
                                       removed={sid:c for sid,c in catalogs[before][index].items() if source_key(c) not in new_keys}))
                refs=[]
                if actual_cap==after:
                    added=deltas[0]['added'] | deltas[1]['added']
                    for call in r.get('answer_calls',[]):
                        refs.extend(dict(stage=call['stage'],**c) for c in citations(call.get('response'),added))
                source_details=dict(source_nomination=nomination,source_seat=proposals[after]['source_seat'],
                    catalog_deltas=dict(sources=deltas[0],constraints=deltas[1]),reader_references=refs,
                    base_packet_hash=ctx['base_packet_hash'],
                    packet_hashes={mode:evidence_packet.digest(packet) for mode,packet in packets.items()})
            rows.append(dict(qa_id=r['qa_id'],score=r['score'],actual_cap=actual_cap,scores=scores,
                ids={cap:p['appended_ids'] for cap,p in proposals.items()},
                added_bytes={cap:p['added_bytes'] for cap,p in proposals.items()},catalog_bytes=catalog_bytes,
                reader_seconds=r['total_seconds']-r['search_seconds'],**source_details))
        summary=dict(questions=len(rows),actual_packets_verified=len(rows),reader_catalogs_verified=len(rows),
            changed_ids=[r['qa_id'] for r in rows if r['ids'][before]!=r['ids'][after]],
            gained=[r['qa_id'] for r in rows if any(r['scores'][after]['user']) and not any(r['scores'][before]['user'])],
            lost=[r['qa_id'] for r in rows if any(r['scores'][before]['user']) and not any(r['scores'][after]['user'])],
            turn_regressions=[r['qa_id'] for r in rows if any(a and not b for role in ('user','assistant')
                for a,b in zip(r['scores'][before][role],r['scores'][after][role]))],
            added_bytes={cap:sum(r['added_bytes'][cap] for r in rows) for cap in modes},
            catalog_bytes={cap:sum(r['catalog_bytes'][cap] for r in rows) for cap in modes},
            user_turns={cap:sum(sum(r['scores'][cap]['user']) for r in rows) for cap in modes},
            assistant_turns={cap:sum(sum(r['scores'][cap]['assistant']) for r in rows) for cap in modes},
            median_reader_seconds=statistics.median(r['reader_seconds'] for r in rows))
        if args.source_query:
            summary.update(source_seat_questions=[r['qa_id'] for r in rows if r['source_seat']['appended_ids']],
                cited_new_source_questions=[r['qa_id'] for r in rows if any(c['exact_quote'] for c in r['reader_references'])],
                reference_questions={cap:sum(any(r['scores'][cap]['user']) for r in rows) for cap in modes})
        arms[arm]=dict(summary=summary,rows=rows)
    result=dict(arms=arms,lexical=args.lexical,fill=args.fill,source_query=args.source_query,
        auditor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        variant_keys=dict(before=before,after=after),limitation=__doc__)
    (root/('source-query-effect-audit.json' if args.source_query else 'residual-effect-audit.json' if args.fill else 'lexical-effect-audit.json' if args.lexical else 'capacity-effect-audit.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({a:d['summary'] for a,d in arms.items()},ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    mode=p.add_mutually_exclusive_group()
    mode.add_argument('--lexical',action='store_true',help='Compare two-seat 5 KiB reserve with the lexical third seat')
    mode.add_argument('--fill',action='store_true',help='Compare lexical reserve with residual capacity in the same 5 KiB')
    mode.add_argument('--source-query',action='store_true',help='Compare lexical reserve with a priority source-query seat')
    p.add_argument('--gold-csv',type=Path,default=Path(__file__).resolve().parents[1]/'data/raw/personamem-v2.csv')
    main(p.parse_args())
