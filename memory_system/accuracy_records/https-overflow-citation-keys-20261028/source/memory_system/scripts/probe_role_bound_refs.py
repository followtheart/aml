"""Recorded review input/schema preflight; optional one real provider review.

No answer or gold metadata enters requests. Input reproduction is not evidence
of retrieval improvement. Live mode is a fixed-card diagnostic, not HTTPS.
"""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import budget,config,listwise_recovery as recovery,schemas
from app import recovery_spans
import jsonschema


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


async def main(args):
    traces={t['search_id']:t for t in records(args.run/'server-search-debug.jsonl')}
    results=records(args.run/'completed/data/results/personamem-v2-32k.jsonl')
    checks=[]
    valid_judgments_checked=0
    invalid_judgments_rejected=0
    live=None
    for result in results:
        t=traces[result['search_id']]
        req=schemas.SearchRequest(**{k:t[k] for k in ('user_id','query','options','top_k','reference_time','evidence_token_budget')})
        reviews=[('primary',r) for r in t['cascade']['listwise'].get('recovery',{}).get('batches',[])]
        reviews.append(('overflow',t['cascade'].get('overflow',{}).get('review',{})))
        with patch.multiple(config,**t['versions']['settings']):
            for kind,review in reviews:
                cards=review.get('sources',[])
                if not cards or not review.get('prompt_sha256'):
                    continue
                with patch.object(config,'LISTWISE_ROLE_BOUND_REFS',False):
                    before=recovery.request_for(req,cards)
                with patch.object(config,'LISTWISE_ROLE_BOUND_REFS',True):
                    after=recovery.request_for(req,cards)
                if hashlib.sha256(before['prompt'].encode()).hexdigest()!=review['prompt_sha256']:
                    raise ValueError('Recorded prompt did not reproduce')
                if before['reservation']!=review['request_bound']:
                    raise ValueError('Recorded reservation did not reproduce')
                if before['prompt']!=after['prompt']:
                    raise ValueError('Role constraint changed source prompt')
                built=recovery_spans.build(req,cards,role_bound=True)
                candidate_keys={mid:key for key,mid in built['candidate_map'].items()}
                for judgment in review.get('judgments',[]):
                    if 'span_reference_error' in judgment.get('validation_errors',[]):
                        continue
                    key=candidate_keys[judgment['candidate_id']]
                    parts=[]
                    for citation in judgment['citations']:
                        part=next(pid for pid,owner in built['owners'][key].items()
                                  if owner==citation['source_id'] and built['spans'][pid]['quote'].strip()==citation['quote'])
                        parts.append(part)
                    decision=dict(useful=judgment['useful'],target_id=judgment['target_id'],
                                  evidence_kind=judgment['evidence_kind'],citation_parts=parts)
                    validator=jsonschema.Draft202012Validator(built['schema']['properties'][key])
                    valid=validator.is_valid(decision)
                    if judgment['valid']:
                        if not valid:
                            raise ValueError('Role schema rejected previously valid judgment')
                        valid_judgments_checked+=1
                    elif not valid:
                        invalid_judgments_rejected+=1
                checks.append(dict(qa_id=result['qa_id'],stage=kind,candidates=len(cards),prompt_bytes=len(before['prompt'].encode()),
                    baseline_reservation=before['reservation'],treatment_reservation=after['reservation']))
                if kind=='overflow' and args.live_qa==result['qa_id']:
                    if config.FAKE:
                        raise ValueError('Live compatibility check requires real provider')
                    by_id={c['id']:c for c in t['ranked']}
                    pending=[by_id[c['candidate_id']] for c in cards]
                    with patch.object(config,'LISTWISE_ROLE_BOUND_REFS',True),budget.scope(seconds=45,calls=1,tokens=64000) as limits:
                        selected,trace=await recovery._recover_batch(req,pending,[],set(range(len(pending))),stage='diagnostic.role_bound_review')
                    live=dict(qa_id=result['qa_id'],selected_ids=[c['id'] for c in selected],trace=trace,calls=limits.calls,tokens=limits.tokens)
    summary=dict(reproduced_requests=len(checks),source_prompts_unchanged=True,
        previously_valid_judgments_preserved=valid_judgments_checked,
        previously_invalid_judgments_schema_rejected=invalid_judgments_rejected,
        maximum_reservation_growth=max(r['treatment_reservation']-r['baseline_reservation'] for r in checks),
        maximum_treatment_reservation=max(r['treatment_reservation'] for r in checks),
        total_reservation_growth=sum(r['treatment_reservation']-r['baseline_reservation'] for r in checks))
    with args.output.open('x') as out:
        json.dump(dict(summary=summary,checks=checks,live=live,limitation=__doc__),out,indent=2)
        out.write('\n')
    print(json.dumps(dict(summary=summary,live_status=live['trace']['status'] if live else None,
                          live_errors=live['trace']['errors'] if live else None)))
    if args.live_qa and (not live or live['trace']['status']=='error'):
        raise RuntimeError('Provider compatibility probe failed')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--live-qa',help='Optional recorded QA identity for one real overflow review')
    asyncio.run(main(p.parse_args()))
