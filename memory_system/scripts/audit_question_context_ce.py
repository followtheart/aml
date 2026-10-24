"""Audit actual question-CE requests, source transport and incremental cost.

Treatment effects hold each treatment packet's own upstream decisions fixed.
Control is audited unchanged; no unobserved CE scores are invented for control.
Reference delivery does not establish semantic support or answer correctness.
Catalog bytes are input text measurements, not billed reader token usage.
"""
import argparse
import ast
from collections import Counter
import copy
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import audit_https_funnel as audit
from audit_residual_reader import citations, source_key
from analyze_overflow_https import provider_stage_costs
from evaluation_model_guard import check_versions


def records(path, partial=False):
    data=path.read_bytes()
    if partial and data and not data.endswith(b'\n'):
        data=data[:data.rfind(b'\n')+1]
    return [json.loads(line) for line in data.splitlines() if line.strip()]


def main(args):
    root=args.run.resolve()
    if not args.allow_partial and (root/'exit-code').read_text().strip()!='0':
        raise ValueError('Require complete successful trial')
    snapshot=root/'source'
    sys.path.insert(0,str(snapshot/'memory_system'))
    from app import answer_choice,config,conversation_context,evidence_packet,question_context_ce,search_pipeline
    plan=json.loads((root/'plan.json').read_text())
    if plan['toggle']!='PACKET_QUESTION_CONTEXT_CE':raise ValueError('Unexpected experiment')
    csv.field_size_limit(32*1024*1024)
    with args.gold_csv.open() as f:gold=list(csv.DictReader(f))
    arms={}
    for arm in ('control','treatment'):
        run=root/arm
        if not args.allow_partial:
            subprocess.run([sys.executable,str(Path(__file__).with_name('audit_adjacent_context.py')),
                            '--run',str(run)],check=True,stdout=subprocess.DEVNULL)
        result_rows=records(run/'completed/data/results/personamem-v2-32k.jsonl',args.allow_partial)
        rs={r['search_id']:r for r in result_rows if not r.get('error_stage')}
        if not args.allow_partial and len(rs)!=len(result_rows):raise ValueError('Incomplete reader cohort')
        raw={}
        for event in records(run/'server-memory-debug.jsonl'):
            raw.setdefault(event['user_id'],{}).update({audit.identity(s):s for s in event['source_messages']})
        rows=[]
        for t in records(run/'server-search-debug.jsonl',args.allow_partial):
            if args.allow_partial and t['search_id'] not in rs:continue
            check_versions(t['versions'],'service trace')
            cfg=t['versions']['settings'];enabled=cfg['PACKET_QUESTION_CONTEXT_CE']
            if enabled!=(arm=='treatment'):raise ValueError('Unexpected arm toggle')
            for name,value in t['versions']['file_hashes'].items():
                if hashlib.sha256((snapshot/name).read_bytes()).hexdigest()!=value:raise ValueError('Frozen source changed')
            r=rs[t['search_id']];ctx=t['cascade']['conversation_context'];manifest=t['coverage_manifest']
            n=len(ctx['base_packet_ids'])
            base=[dict(p,packet_hash=ctx['base_packet_hash']) for p in t['returned'][:n]]
            base_manifest=copy.deepcopy(manifest);base_manifest.pop('conversation_context')
            base_manifest.update(included_ids=ctx['base_packet_ids'],token_upper_bound=ctx['base_token_upper_bound'],
                evidence_count=manifest['evidence_count']-len(ctx['appended_ids']))
            with patch.multiple(config,**cfg),patch('socket.socket.connect',side_effect=AssertionError('Offline audit')):
                pool={c['id']:c for c in search_pipeline._foresight_filter(copy.deepcopy(t['ranked']),copy.deepcopy(t['plan']))
                    if c.get('view_status')!='stale' and '_packet_item' in c}
                fine=[pool[mid] for mid in t['cascade']['fine']['selected_ids'] if mid in pool and mid not in t['selection']['selected_ids']]
                prefix,digest,prefix_manifest,_=conversation_context.attach(base,ctx['base_packet_hash'],base_manifest,fine,
                    user_id=t['user_id'],top_k=t['top_k'],enabled=False,reserve_enabled=True,reserve_extended=True,
                    lexical_enabled=True,query=t['query'],options=t['options'])
                if enabled:
                    actual,actual_digest,actual_manifest=question_context_ce.replay(prefix,digest,prefix_manifest,fine,
                        ctx['question_ce'],user_id=t['user_id'],top_k=t['top_k'],query=t['query'])
                else:actual,actual_digest,actual_manifest=prefix,digest,prefix_manifest
            if actual!=t['returned'] or actual_digest!=t['packet_hash'] or actual_manifest!=manifest:
                raise ValueError('Packet or manifest did not reproduce')
            ce=ctx.get('question_ce',{});state=ce.get('budget_before');calls=tokens=0
            if state:
                if state['max_calls']!=cfg['SEARCH_MAX_CALLS'] or state['max_tokens']!=cfg['SEARCH_MAX_TOKENS']:
                    raise ValueError('Budget configuration mismatch')
                calls=r['coverage_manifest']['provider_calls']-state['calls']
                tokens=r['coverage_manifest']['provider_tokens']-state['tokens']
                expected_calls=1 if ce['status']=='scored' else 0
                if ((ce['status']=='error' and calls not in (0,1)) or
                        (ce['status']!='error' and calls!=expected_calls) or tokens<0):
                    raise ValueError('Optional CE budget accounting mismatch')
            snippet=gold[int(r['qa_id'])]['related_conversation_snippet']
            try:snippet=json.loads(snippet)
            except ValueError:snippet=ast.literal_eval(snippet)
            coverage={};catalogs={};catalog_bytes={}
            for mode,packet in (('prefix',prefix),('actual',actual)):
                cards=[dict(s,text=s.get('content',''),id=f'{i}:{j}') for i,p in enumerate(packet) for j,s in enumerate(p['sources'])]
                coverage[mode]={role:[v['full'] for v in audit.turn_coverage(
                    [audit.norm(s['content']) for s in snippet if s['role']==role],cards,raw[t['user_id']],role=role)]
                    for role in ('user','assistant')}
                catalogs[mode]=answer_choice.build_catalog(packet)
                catalog_bytes[mode]=sum(len(c['text'].encode()) for cat in catalogs[mode] for c in cat.values())
            if (list(catalogs['actual'][0].values())!=r['answer_source_catalog'] or
                    list(catalogs['actual'][1].values())!=r['answer_constraint_catalog']):
                raise ValueError('Reader catalogs do not reproduce')
            deltas=[]
            for index in (0,1):
                before={source_key(c) for c in catalogs['prefix'][index].values()}
                after={source_key(c) for c in catalogs['actual'][index].values()}
                if not before<=after:raise ValueError('An original catalog entry was removed')
                deltas.append({sid:c for sid,c in catalogs['actual'][index].items() if source_key(c) not in before})
            references=[]
            for call in r.get('answer_calls',[]):
                references.extend(dict(stage=call['stage'],**v) for v in citations(call.get('response'),deltas[0]|deltas[1]))
            rows.append(dict(qa_id=r['qa_id'],score=r['score'],status=ce.get('status','disabled'),
                coverage=coverage,catalog_bytes=catalog_bytes,extra_calls=calls,extra_tokens=tokens,
                extra_ids=ce.get('proposal',{}).get('appended_ids',[]),
                extra_bytes=ce.get('proposal',{}).get('added_bytes',0),
                question_ce=ce,catalog_additions=dict(sources=deltas[0],constraints=deltas[1]),
                reader_references=references))
        if not args.allow_partial and len(rows)!=plan['questions']:raise ValueError('Incomplete cohort')
        logged=provider_stage_costs((run/'server.log').read_text()).get(question_context_ce.STAGE,dict(calls=0,tokens=0))
        if not args.allow_partial and (sum(r['extra_calls'] for r in rows)!=logged['calls'] or sum(r['extra_tokens'] for r in rows)!=logged['tokens']):
            raise ValueError('Optional CE cost differs from provider logs')
        summary=dict(questions=len(rows),actual_packets_verified=len(rows),reader_catalogs_verified=len(rows),
            progress_snapshot=args.allow_partial,statuses=dict(Counter(r['status'] for r in rows)),
            extra_calls=sum(r['extra_calls'] for r in rows),extra_tokens=sum(r['extra_tokens'] for r in rows),
            appended_questions=sum(bool(r['extra_ids']) for r in rows),extra_bytes=sum(r['extra_bytes'] for r in rows),
            reference_questions={m:sum(any(r['coverage'][m]['user']) for r in rows) for m in ('prefix','actual')},
            user_turns={m:sum(sum(r['coverage'][m]['user']) for r in rows) for m in ('prefix','actual')},
            assistant_turns={m:sum(sum(r['coverage'][m]['assistant']) for r in rows) for m in ('prefix','actual')},
            catalog_bytes={m:sum(r['catalog_bytes'][m] for r in rows) for m in ('prefix','actual')},
            gained=[r['qa_id'] for r in rows if any(r['coverage']['actual']['user']) and not any(r['coverage']['prefix']['user'])],
            lost=[r['qa_id'] for r in rows if any(r['coverage']['prefix']['user']) and not any(r['coverage']['actual']['user'])],
            turn_regressions=[r['qa_id'] for r in rows if any(x and not y for role in ('user','assistant')
                for x,y in zip(r['coverage']['prefix'][role],r['coverage']['actual'][role]))],
            cited_new_source_questions=[r['qa_id'] for r in rows if any(v['exact_quote'] for v in r['reader_references'])])
        arms[arm]=dict(summary=summary,rows=rows)
    output=root/('question-ce-effect-progress.json' if args.allow_partial else 'question-ce-effect-audit.json')
    output.write_text(json.dumps(dict(arms=arms,limitation=__doc__,
        auditor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({a:r['summary'] for a,r in arms.items()},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--allow-partial',action='store_true',help='Verify completed rows only; write a distinct progress artifact')
    p.add_argument('--gold-csv',type=Path,default=Path(__file__).resolve().parents[1]/'data/raw/personamem-v2.csv')
    main(p.parse_args())
