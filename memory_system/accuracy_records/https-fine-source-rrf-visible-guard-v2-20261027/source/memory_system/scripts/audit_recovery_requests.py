"""Audit emitted review requests/quotes using the exact frozen service source.

No provider calls. A valid citation is not proof of semantic relevance.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def digest(value):
    return hashlib.sha256(value).hexdigest()


def main(args):
    run=args.run.resolve()
    if not args.allow_partial and (not (run/'exit-code').exists() or (run/'exit-code').read_text().strip()!='0'):
        raise ValueError('Require a completed successful arm, or explicitly --allow-partial')
    snapshot=run.parent/'source'
    sys.path.insert(0,str(snapshot/'memory_system'))
    from app import config,listwise_recovery as recovery,recovery_spans as spans,schemas
    results={r['search_id']:r for r in records(run/'completed/data/results/personamem-v2-32k.jsonl')}
    rows=[]
    for t in records(run/'server-search-debug.jsonl'):
        if t['search_id'] not in results:
            continue
        for name,expected in t['versions']['file_hashes'].items():
            if digest((snapshot/name).read_bytes())!=expected:
                raise ValueError('Frozen source changed')
        req=schemas.SearchRequest(**{k:t[k] for k in
            ('user_id','query','options','top_k','reference_time','evidence_token_budget')})
        reviews=[('main',i,v) for i,v in enumerate(t['cascade']['listwise'].get('recovery',{}).get('batches',[]))]
        reviews.append(('overflow',0,t['cascade'].get('overflow',{}).get('review',{})))
        with patch.multiple(config,**t['versions']['settings']):
            if not config.LISTWISE_SPAN_REFS:
                raise ValueError('Requires exact part references')
            for stage,batch,v in reviews:
                cards=v.get('sources',[])
                if not cards or not v.get('prompt_sha256'):
                    continue
                request=recovery.request_for(req,cards)
                if (digest(request['prompt'].encode())!=v['prompt_sha256']
                        or len(request['prompt'].encode())!=v['input_bytes']
                        or request['reservation']!=v['request_bound']):
                    raise ValueError('Recorded request did not reproduce')
                built=spans.build(req,cards,role_bound=config.LISTWISE_ROLE_BOUND_REFS,
                    transform_context=config.LISTWISE_TRANSFORM_CONTEXT,abbreviations=config.LISTWISE_ABBREVIATION_PARTS)
                keys={mid:key for key,mid in built['candidate_map'].items()}
                for card in cards:
                    key=keys[card['candidate_id']]
                    for source in card['sources']:
                        text=''.join(built['spans'][pid]['quote'] for pid,owner in built['owners'][key].items()
                                     if owner==source['source_id'])
                        if text!=source['text']:
                            raise ValueError('Source text or ownership not preserved')
                checked=0;cited=[]
                for judgment in v.get('judgments',[]):
                    if 'span_reference_error' in judgment['validation_errors']:
                        if judgment['valid']:
                            raise ValueError('Invalid span accepted')
                        continue
                    card=next(c for c in cards if c['candidate_id']==judgment['candidate_id'])
                    decision={k:judgment[k] for k in recovery.Decision.model_fields}
                    _,verified=recovery.validate(dict(decisions=[decision]),[card],built['targets'])
                    if any(verified[0][k]!=judgment[k] for k in ('valid','validation_errors')):
                        raise ValueError('Runtime validator outcome differs')
                    key=keys[card['candidate_id']]
                    for ref in judgment['citations']:
                        matches=[pid for pid,owner in built['owners'][key].items()
                                 if owner==ref['source_id'] and built['spans'][pid]['quote'].strip()==ref['quote']]
                        if not matches:
                            raise ValueError('Cited text is not an exact owned emitted part')
                        cited.append(dict(candidate_id=card['candidate_id'],part_ids=matches,
                                          source_id=ref['source_id'],quote=ref['quote']))
                    checked+=1
                with patch.object(config,'LISTWISE_ABBREVIATION_PARTS',False):
                    baseline=recovery.request_for(req,cards)
                with patch.object(config,'LISTWISE_ABBREVIATION_PARTS',True):
                    alternative=recovery.request_for(req,cards)
                rows.append(dict(qa_id=results[t['search_id']]['qa_id'],stage=stage,batch=batch,
                    sources_sha256=digest(json.dumps(cards,ensure_ascii=False,sort_keys=True).encode()),
                    request_sha256=digest(json.dumps(request,ensure_ascii=False,sort_keys=True).encode()),
                    system_sha256=digest(request['system'].encode()),
                    schema_sha256=digest(json.dumps(request['schema'],sort_keys=True).encode()),
                    prompt_sha256=v['prompt_sha256'],candidate_ids=[c['candidate_id'] for c in cards],
                    checked_judgments=checked,span_errors=sum('span_reference_error' in j['validation_errors'] for j in v.get('judgments',[])),
                    affected_by_segmentation=baseline!=alternative,
                    segmentation_reservation_delta=alternative['reservation']-baseline['reservation'],
                    judgments=v.get('judgments',[]),cited_parts=cited))
    summary=dict(questions=len(results),audited_requests=len(rows),
        stages=dict(Counter(r['stage'] for r in rows)),
        checked_judgments=sum(r['checked_judgments'] for r in rows),
        span_errors=sum(r['span_errors'] for r in rows),
        affected_requests=sum(r['affected_by_segmentation'] for r in rows),
        affected_questions=sorted({r['qa_id'] for r in rows if r['affected_by_segmentation']},key=int),
        partial=args.allow_partial)
    args.output.write_text(json.dumps(dict(summary=summary,rows=rows,limitation=__doc__),ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--allow-partial',action='store_true')
    main(p.parse_args())
