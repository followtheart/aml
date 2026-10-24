"""Offline bounded retrieval context alongside model-selected evidence.

This is an explicit recall/precision tradeoff, not validated positive review.
No gold, new retrieval, source rehydration, generated facts or displacement is
used for admission. Only already scored, eligible fine candidates can supply
visible original excerpts containing a new complete user turn.
"""
import argparse
import ast
import copy
import csv
import json
import math
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import answer_context,config,evidence_packet,schemas,search_pipeline
from app.conversation_context import source_key,usable
import audit_https_funnel as audit
from evaluation_model_guard import check_versions


from app.fine_context_reserve import propose


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main(args):
    if args.output.exists():raise FileExistsError(args.output)
    if (args.run/'exit-code').read_text().strip()!='0':raise ValueError('Require completed successful arm')
    results={r['search_id']:r for r in records(args.run/'completed/data/results/personamem-v2-32k.jsonl')}
    raw={}
    for event in records(args.run/'server-memory-debug.jsonl'):
        raw.setdefault(event['user_id'],{}).update({audit.identity(s):s for s in event['source_messages']})
    csv.field_size_limit(32*1024*1024)
    with args.gold_csv.open() as stream:gold=list(csv.DictReader(stream))
    rows=[]
    for t in records(args.run/'server-search-debug.jsonl'):
        check_versions(t['versions'],'source trace')
        if evidence_packet.digest(t['returned'])!=t['packet_hash']:raise ValueError('Baseline identity differs')
        baseline=t['returned']
        if args.base_packet:
            context=t['cascade']['conversation_context']
            baseline=[dict(item,packet_hash=context['base_packet_hash'])
                      for item in t['returned'][:len(context['base_packet_ids'])]]
            if ([p['id'] for p in baseline]!=context['base_packet_ids']
                    or evidence_packet.digest(baseline)!=context['base_packet_hash']):
                raise ValueError('Pre-context packet identity differs')
        with patch.multiple(config,**t['versions']['settings']):
            pool={c['id']:c for c in search_pipeline._foresight_filter(t['ranked'],copy.deepcopy(t['plan']))
                  if c.get('view_status')!='stale' and '_packet_item' in c}
            selected=set(t['selection']['selected_ids'])
            fine=[pool[mid] for mid in t['cascade']['fine']['selected_ids'] if mid in pool and mid not in selected]
            packet,details=propose(baseline,fine,user_id=t['user_id'],top_k=t['top_k'],
                byte_budget=t['coverage_manifest']['token_budget'],limit=args.limit)
        # Only after the gold-independent proposal is fixed, score transport.
        qa=results[t['search_id']]['qa_id'];snippet=gold[int(qa)]['related_conversation_snippet']
        try:snippet=json.loads(snippet)
        except ValueError:snippet=ast.literal_eval(snippet)
        turns=[audit.norm(s['content']) for s in snippet if s['role']=='user']
        def coverage(items):
            cards=[dict(s,text=s.get('content',''),id=f'{i}:{j}') for i,item in enumerate(items)
                   for j,s in enumerate(item.get('sources',[]))]
            return sum(c['full'] for c in audit.turn_coverage(turns,cards,raw[t['user_id']]))
        rows.append(dict(qa_id=qa,before_turns=coverage(baseline),after_turns=coverage(packet),proposal=details,
            added_sources=[dict(id=p['id'],sources=p['sources']) for p in packet[len(baseline):]]))
    summary=dict(questions=len(rows),original_packet_identities_verified=len(rows),limit=args.limit,base_packet=args.base_packet,
        before_reference_questions=sum(r['before_turns']>0 for r in rows),
        after_reference_questions=sum(r['after_turns']>0 for r in rows),
        before_turns=sum(r['before_turns'] for r in rows),after_turns=sum(r['after_turns'] for r in rows),
        added_context_items=sum(len(r['proposal']['appended_ids']) for r in rows),
        added_bytes=sum(r['proposal']['added_bytes'] for r in rows),
        gained=[r['qa_id'] for r in rows if r['after_turns'] and not r['before_turns']],
        lost=[r['qa_id'] for r in rows if r['before_turns'] and not r['after_turns']])
    args.output.write_text(json.dumps(dict(summary=summary,rows=rows,limitation=__doc__),ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--limit',type=int,choices=[1,2],default=2)
    p.add_argument('--base-packet',action='store_true')
    p.add_argument('--gold-csv',type=Path,default=Path(__file__).resolve().parents[1]/'data/raw/personamem-v2.csv')
    main(p.parse_args())
