"""Frozen-input listwise ablations and a human-review benchmark workflow.

export never calls providers. run requires --execute and a real provider.
score uses only explicitly reviewed annotations, never gold-source presence or
model-generated purpose labels as human truth. Run artifacts may contain sources.
"""
import argparse
import asyncio
import copy
import hashlib
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import budget, cascade_rerank, config, listwise_contract, listwise_staged, prompts, run_metadata


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def frozen(run):
    searches = {s['search_id']: s for s in records(run/'server-search-debug.jsonl')}
    for result in records(run/'completed/data/results/personamem-v2-32k.jsonl'):
        search = searches[result['search_id']]
        by_id = {r['id']: r for r in search['ranked']}
        scores = {r['id']: r['score'] for r in search['cascade']['fine']['selection']['decisions']}
        rows = [copy.deepcopy(by_id[mid]) for mid in search['cascade']['listwise']['candidate_ids']]
        for row in rows:
            row['_ce_score'] = scores.get(row['id'])
        plan = copy.deepcopy(search.get('plan', {}))
        plan['_coverage_requirements'] = [{k:r[k] for k in ('id','text','origin') if k in r}
            for r in search['coverage_manifest']['coverage']['requirements']]
        plan['_fusion_edges'] = search.get('fusion_edges', [])
        plan['_fusion_triples'] = search.get('fusion_triples', [])
        plan['_fine_path_groups'] = search['cascade']['fine']['selection'].get('adaptive', {}).get('path_groups', [])
        req = SimpleNamespace(query=search['query'], options=search.get('options') or [])
        yield str(result['qa_id']), req, plan, rows, search


def export(args):
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    examples = []
    for qid, req, plan, rows, search in frozen(args.run):
        lw = search['cascade']['listwise']
        for row in rows:
            examples.append(dict(qa_id=qid,candidate_id=row['id'],query=req.query,options=req.options,
                rank_text=row['_rank_text'],sources=row.get('_packet_item',{}).get('sources',[]),
                context=listwise_staged.listwise_context.diagnose(row),
                annotation=dict(status='pending',reviewer='',useful=None,
                    roles=[],necessity=None,represented_by=[],source_quotes=[],rationale=''),
                # Model outcomes are separate from labels and absent in blind form.
                historical=dict(primary_deleted=row['id'] in lw.get('primary_irrelevant_ids',[]),
                    review_restored=row['id'] in lw.get('recovery',{}).get('restored_ids',[]),
                    final_selected=row['id'] in lw['selected_ids'])))
    blind = [{k:v for k,v in r.items() if k!='historical'} for r in examples]
    for name, data in [('review.jsonl',blind),('historical.jsonl',examples)]:
        target = out/name
        if target.exists():
            raise FileExistsError(f'Refusing to overwrite annotations: {target}')
        target.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in data))
    manifest=dict(examples=len(examples),questions=len({r['qa_id'] for r in examples}),
        reviewed=0, roles=['unique_fact','qualification','counterevidence','temporal_change','bridge','background','duplicate','unrelated'],
        necessity=['required','helpful','redundant','none'],
        instructions='A human reviewer must read the question/options and sources, then set status=reviewed, reviewer, useful, roles, necessity, exact source_quotes and rationale. A duplicate needs a surviving represented_by candidate. Do not infer labels from gold-carrier presence or model outputs.',
        source_run=str(args.run),source_sha256=hashlib.sha256((args.run/'server-search-debug.jsonl').read_bytes()).hexdigest())
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(manifest,ensure_ascii=False))


async def run(args):
    if not args.execute or config.FAKE:
        raise ValueError('Live ablations require --execute and AML_FAKE=0; no fake quality results')
    args.output.mkdir(parents=True,exist_ok=True)
    output=args.output/'predictions.jsonl'
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    selected_q=set(args.qa.split(',')) if args.qa else None
    variants=args.variants.split(',')
    allowed={'legacy_inline','legacy_shared','staged_inline','staged_shared','staged_reverse','staged_shuffle','staged_batch8','staged_batch12'}
    if not set(variants)<=allowed: raise ValueError('Unknown variant')
    metadata=dict(versions=run_metadata.versions(),
        source_run=str(args.run),source_sha256=hashlib.sha256((args.run/'server-search-debug.jsonl').read_bytes()).hexdigest(),
        note='Frozen candidate set and CE scores. Legacy selected_ids are primary only; compare proposed_deleted across variants, not legacy primary against staged final. No answer-model or HTTPS rerun.')
    (args.output/'metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n')
    archived_prompt = (args.run/'source/prompts/05c_listwise_rerank.txt').read_text().strip()
    for qid,req,plan,original,search in frozen(args.run):
        if selected_q and qid not in selected_q: continue
        expected={r['id'] for r in original}
        for variant in variants:
            rows=copy.deepcopy(original)
            if variant=='staged_reverse': rows.reverse()
            if variant=='staged_shuffle': random.Random(0).shuffle(rows)
            assert {r['id'] for r in rows}==expected
            started=time.monotonic()
            result=dict(qa_id=qid,variant=variant,input_ids=[r['id'] for r in rows],
                input_fingerprint=hashlib.sha256(json.dumps(sorted((r['id'],r['_rank_text'],r['_ce_score']) for r in rows),ensure_ascii=False).encode()).hexdigest(),
                model=config.LLM_MODEL)
            with budget.scope(seconds=args.seconds,calls=12,tokens=args.tokens) as limits:
                try:
                    if variant.startswith('legacy'):
                        original_load = prompts.load
                        with patch.object(prompts, 'load', side_effect=lambda name: archived_prompt if name=='05c_listwise_rerank.txt' else original_load(name)):
                            request=cascade_rerank.listwise_request(req,plan,rows,shared_sources=variant.endswith('shared'),keyed=True)
                        response=await listwise_staged.dispatch(request,time.monotonic()+args.seconds,'experiment.listwise.legacy')
                        state=listwise_contract.State(len(rows),keyed=True)
                        state.accept(response)
                        # Measure primary candidate judgments independently of optional group failures.
                        if state.ranking is None or state.irrelevant is None:
                            raise ValueError('Invalid candidate contract')
                        result.update(proposed_deleted=[rows[i]['id'] for i in state.irrelevant],
                            selected_ids=[rows[i]['id'] for i in state.ranking if i not in state.irrelevant],
                            phase='primary_only',errors=state.errors,response=response)
                    else:
                        size=8 if variant.endswith('8') else 12 if variant.endswith('12') else 10
                        selected,_,trace=await listwise_staged.select(req,copy.deepcopy(plan),rows,
                            deadline=time.monotonic()+args.seconds,batch_size=size,shared=variant!='staged_inline')
                        result.update(proposed_deleted=trace['primary_irrelevant_ids'],
                            selected_ids=[r['id'] for r in selected],phase='after_review_and_order',trace=trace)
                    result['status']='complete'
                except Exception as exc:
                    result.update(status='error',error=type(exc).__name__,detail=str(exc)[:300])
                result.update(seconds=time.monotonic()-started,calls=limits.calls,tokens=limits.tokens)
                assert limits.reserved_tokens==limits.reserved_calls==0
            with output.open('a') as stream: stream.write(json.dumps(result,ensure_ascii=False)+'\n')
            print(json.dumps({k:result.get(k) for k in ('qa_id','variant','status','seconds','calls','tokens')},ensure_ascii=False),flush=True)
    (args.output/'completed').write_text('complete\n')


def score(args):
    examples=records(args.labels)
    labels={}
    roles={'unique_fact','qualification','counterevidence','temporal_change','bridge','background','duplicate','unrelated'}
    for r in examples:
        a=r['annotation']
        if a['status']!='reviewed': continue
        if (not a.get('reviewer','').strip() or type(a.get('useful')) is not bool or
            not a.get('rationale','').strip() or not a.get('roles') or not set(a['roles'])<=roles or
            a.get('necessity') not in ('required','helpful','redundant','none')):
            raise ValueError('Incomplete reviewed annotation')
        if a['useful'] and not a.get('source_quotes'):
            raise ValueError('Useful evidence requires exact source quotes')
        for quote in a.get('source_quotes',[]):
            if not isinstance(quote,str) or not quote or not any(quote in s.get('content','') for s in r['sources']):
                raise ValueError('Annotation quote not in source')
        if 'duplicate' in a['roles'] and not a.get('represented_by'):
            raise ValueError('Duplicate annotation requires replacement')
        key=(str(r['qa_id']),r['candidate_id'])
        if key in labels: raise ValueError('Repeated annotation')
        labels[key]=a
    counts={}
    for p in records(args.predictions):
        if p['status']!='complete': continue
        c=counts.setdefault(p['variant'],Counter())
        deleted=set(p['proposed_deleted'])
        for mid in p['input_ids']:
            a=labels.get((str(p['qa_id']),mid))
            if a is None: continue
            c['reviewed']+=1
            if a['useful']:
                c['useful']+=1;c['useful_proposed_deleted']+=mid in deleted
            else:
                c['not_useful']+=1;c['not_useful_retained']+=mid not in deleted
            for role in a['roles']:
                c[role+'_total']+=1;c[role+'_proposed_deleted']+=mid in deleted
    report=dict(reviewed_annotations=len(labels),pending=len(examples)-len(labels),
        status='scored' if labels else 'awaiting_human_review',variants={})
    for variant,c in counts.items():
        report['variants'][variant]=dict(c, useful_false_deletion_rate=c['useful_proposed_deleted']/c['useful'] if c['useful'] else None,
            not_useful_retention_rate=c['not_useful_retained']/c['not_useful'] if c['not_useful'] else None)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    e=sub.add_parser('export');e.add_argument('--run',type=Path,required=True);e.add_argument('--output',type=Path,required=True)
    r=sub.add_parser('run');r.add_argument('--run',type=Path,required=True);r.add_argument('--output',type=Path,required=True)
    r.add_argument('--execute',action='store_true');r.add_argument('--qa',default='')
    r.add_argument('--variants',default='legacy_inline,legacy_shared,staged_inline,staged_shared,staged_reverse,staged_shuffle,staged_batch8,staged_batch12')
    r.add_argument('--seconds',type=int,default=45);r.add_argument('--tokens',type=int,default=200000)
    s=sub.add_parser('score');s.add_argument('--labels',type=Path,required=True);s.add_argument('--predictions',type=Path,required=True);s.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.command=='export':export(args)
    elif args.command=='run':asyncio.run(run(args))
    else:score(args)


if __name__=='__main__':main()
