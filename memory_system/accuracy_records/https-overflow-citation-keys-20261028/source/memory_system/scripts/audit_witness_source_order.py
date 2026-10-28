"""Rebuild CE/source-order requests, validate witnesses, and inspect reference use.

Reference-source quotation is a retrieval proxy, not personal entailment or
answer accuracy. A correct-option reference quotation may still be irrelevant
within that source. Gold enters only offline analysis after all provider calls.
"""
import argparse
import ast
import asyncio
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re

from probe_witness_source_order import prepare,ordered_request,decode,documents_for,records,fingerprint,reservation
from audit_https_funnel import norm


def quote_inventory(witnesses,sources,reference_ids,turns):
    return [dict(letter=letter,source_id=w['source_id'],quote=w['quote'],
            reference_origin=w['source_id'] in reference_ids,
            inside_reference_turn=(w['source_id'] in reference_ids and any(norm(w['quote']) in turn for turn in turns)),
            role=sources[w['source_id']].get('declared') or sources[w['source_id']]['role'])
        for letter,ws in witnesses.items() for w in ws]


async def main(args):
    source=args.diagnostic
    if not source.with_suffix('.exit-code').exists():
        raise ValueError('Diagnostic is still running')
    plan=json.loads(source.with_suffix('.plan.json').read_text())
    recorded=json.loads(source.read_text())
    run=Path(plan['run'])
    for name,value in plan['source_hashes'].items():
        if hashlib.sha256((run/name).read_bytes()).hexdigest()!=value:
            raise ValueError('Historical archive changed')
    jobs,settings=await prepare(run)
    if jobs!=plan['jobs'] or settings!=plan['settings']:
        raise ValueError('Requests did not reconstruct')
    if hashlib.sha256(Path(__file__).with_name('probe_witness_source_order.py').read_bytes()).hexdigest()!=plan['script_sha256']:
        raise ValueError('Diagnostic script changed')
    live=records(source.with_suffix('.live.jsonl'));ce=records(source.with_suffix('.ce.jsonl'))
    if live!=recorded['live'] or ce!=recorded['ce']:
        raise ValueError('Response archive differs')
    if len(live)!=plan['expected_llm_calls'] or len(ce)!=plan['expected_ce_calls']:
        raise ValueError('Incomplete recorded cohort')
    live_index={(r['qa_id'],r['arm']):r for r in live}
    ce_index={(r['qa_id'],r['batch']):r for r in ce}
    if len(live_index)!=len(live) or len(ce_index)!=len(ce):
        raise ValueError('Duplicate request identity')
    metrics=[]
    for line in source.with_suffix('.provider.log').read_text().splitlines():
        if 'provider_call ' in line:
            metrics.append(dict(re.findall(r'(\w+)=([^\s]+)',line)))
    for row in [*live,*ce]:
        stage=(f'diagnostic.witness_order.{row["arm"]}.{row["qa_id"]}' if 'arm'in row else
               f'diagnostic.witness_order.ce.{row["qa_id"]}.{row["batch"]}')
        calls=[m for m in metrics if m['stage']==stage]
        if len(calls)!=row['calls'] or sum(int(m['total_tokens']) for m in calls)!=row['tokens']:
            raise ValueError('Provider accounting differs')
        model='gpt-4o-mini' if 'arm'in row else 'qwen3-rerank'
        if any(m['model']!=model for m in calls):
            raise ValueError('Model requirement violated')
    prepared={(r['conversation_id'],q['id']):q for r in records(args.prepared) for q in r['qa']}
    results={r['qa_id']:r for r in records(run/'completed/data/results/personamem-v2-32k.jsonl')}
    stages={r['qa_id']:r for r in json.loads((run/'funnel/stage-transport.json').read_text())['rows']}
    csv.field_size_limit(32*1024*1024)
    with args.gold_csv.open() as f:gold=list(csv.DictReader(f))
    rows=[]
    for job in jobs:
        q=job['qa_id'];r=results[q];s=stages[q]
        qa=prepared[r['conversation_id'],q]
        if qa['options']!=job['options'] or len(qa['gold_labels'])!=1:
            raise ValueError('Prepared question identity mismatch')
        label=qa['gold_labels'][0]
        if label not in ('A','B','C','D'):
            raise ValueError('Unexpected gold label encoding')
        snippet=gold[int(q)]['related_conversation_snippet']
        try:snippet=json.loads(snippet)
        except ValueError:snippet=ast.literal_eval(snippet)
        turns=[norm(z['content']) for z in snippet if z['role']=='user']
        refs=set().union(*(set(c['source_ids']) for c in s['visible']['catalog']['turn_coverage']))
        scores=[None]*len(job['cards']);failed=False;docs=documents_for(job['cards'])
        for bi,indices in enumerate(job['ce_batches']):
            row=ce_index[q,bi]
            if (row['indices']!=indices or row['query_sha256']!=hashlib.sha256(job['ce_query'].encode()).hexdigest()
                    or row['document_sha256']!=[hashlib.sha256(docs[i].encode()).hexdigest() for i in indices]):
                raise ValueError('CE request identity mismatch')
            if row['status']=='ok':
                if len(row['scores'])!=len(indices):
                    raise ValueError('Missing CE score')
                for i,v in zip(indices,row['scores']):scores[i]=v
            else:failed=True
        treatment,order=(job['request'],list(job['sources'])) if failed else ordered_request(job['request'],job['cards'],scores)
        arms={}
        for arm in ['control','treatment']:
            response=live_index[q,arm];request=job['request'] if arm=='control' else treatment
            expected_order=list(job['sources']) if arm=='control' else order
            if (fingerprint(request)!=response['request_sha256'] or reservation(request)!=response['reservation']
                    or response['source_order']!=expected_order):
                raise ValueError('Witness request identity mismatch')
            validated=decode(response.get('response',{}),job)
            if any(response[k]!=v for k,v in validated.items() if k!='status'):
                raise ValueError('Witnesses failed independent decoding')
            inventory={name:quote_inventory(response[name],job['sources'],refs,turns) for name in ['semantic','final_hints']}
            arms[arm]=dict(status=response['status'],source_order=expected_order,**inventory,
                first_reference_rank=min([expected_order.index(sid)+1 for sid in refs if sid in expected_order],default=None),
                first_source_quotes=sum(w['source_id']==expected_order[0] for w in inventory['semantic']),
                one_repeated_quote_for_all_options=(len(response['semantic'])==len(job['options']) and
                    len({(w['source_id'],w['quote']) for w in inventory['semantic']})==1))
        rows.append(dict(qa_id=q,question=qa['question'],options=job['options'],gold=label,
            historical_score=r['score'],catalog_reference_turns=s['visible']['catalog']['full_turns'],
            reference_source_ids=sorted(refs),ce_scores=scores,ce_fallback=failed,arms=arms))
    summary=dict(provider_statuses=dict(Counter(m['status'] for m in metrics)),validated_requests=len(live)+len(ce),arms={})
    for arm in ['control','treatment']:
        def hit(row,stage):
            return any(w['letter']==row['gold'] and w['inside_reference_turn'] for w in row['arms'][arm][stage])
        summary['arms'][arm]=dict(statuses=dict(Counter(r['arms'][arm]['status'] for r in rows)),
            semantic_quotes=sum(len(r['arms'][arm]['semantic']) for r in rows),
            first_source_quotes=sum(r['arms'][arm]['first_source_quotes'] for r in rows),
            repeated_all_options=sum(r['arms'][arm]['one_repeated_quote_for_all_options'] for r in rows),
            correct_option_reference_semantic=[r['qa_id'] for r in rows if hit(r,'semantic')],
            correct_option_reference_final=[r['qa_id'] for r in rows if hit(r,'final_hints')],
            reference_in_first_three=sum(0<(r['arms'][arm]['first_reference_rank'] or 999)<=3 for r in rows))
    for stage in ['semantic','final']:
        field='correct_option_reference_'+stage
        a,b=[set(summary['arms'][arm][field]) for arm in ['control','treatment']]
        summary[stage+'_gains']=sorted(b-a,key=int);summary[stage+'_losses']=sorted(a-b,key=int)
    with args.output.open('x') as f:
        json.dump(dict(summary=summary,rows=rows,limitation=__doc__,
            auditor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            diagnostic_sha256=hashlib.sha256(source.read_bytes()).hexdigest()),f,ensure_ascii=False,indent=2)
        f.write('\n')
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--diagnostic',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--prepared',type=Path,default=Path('data/prepared/personamem-v2-32k.jsonl'))
    p.add_argument('--gold-csv',type=Path,default=Path('data/raw/personamem-v2.csv'))
    asyncio.run(main(p.parse_args()))
