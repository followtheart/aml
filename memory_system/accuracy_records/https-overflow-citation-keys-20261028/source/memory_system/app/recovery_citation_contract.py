"""Prototype schema restricted to exact citations accepted by existing checks.

This restricts citation syntax/attribution only; it does not establish relevance.
The unchanged post-response validator remains authoritative.
"""
import copy
import hashlib
import json
import time
from . import listwise_recovery, recovery_spans


def apply(built):
    result=copy.deepcopy(built)
    cards={c['candidate_id']:c for c in built['cards']}
    allowed={}
    for key,mid in built['candidate_map'].items():
        choices={}
        for kind in recovery_spans.KINDS:
            if kind=='none':continue
            parts=[]
            for pid,source_id in built['owners'][key].items():
                quote=built['spans'][pid]['quote'].strip()
                if not quote:continue
                ref=listwise_recovery.Citation(source_id=source_id,quote=quote)
                if listwise_recovery._check_citation(ref,kind,cards[mid]['sources']) is None:
                    parts.append(pid)
            if parts:choices[kind]=parts
        allowed[key]=choices
        def branch(kind,parts):
            useful=kind!='none'
            return dict(type='object',additionalProperties=False,
                required=['useful','target_id','evidence_kind','citation_parts'],properties=dict(
                    useful=dict(type='boolean',enum=[useful]),
                    target_id=dict(type='string',enum=list(built['targets']) if useful else ['']),
                    evidence_kind=dict(type='string',enum=[kind]),
                    citation_parts=dict(type='array',minItems=1 if useful else 0,maxItems=2 if useful else 0,
                        items=dict(type='string',enum=parts) if useful else dict(type='string'))))
        result['schema']['properties'][key]=dict(anyOf=[branch('none',[])]+
            [branch(kind,parts) for kind,parts in choices.items()])
    result['allowed_citations']=allowed
    assert result['prompt']==built['prompt'] and result['system']==built['system']
    return result


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,
        separators=(',',':')).encode()).hexdigest()


def prepare(req, cards, request, limits, deadline, *, abbreviations=False,
            role_bound=False, transform_context=False):
    """Apply only after original admission; insufficient headroom keeps old wire."""
    built=recovery_spans.build(req,cards,role_bound=role_bound,
        transform_context=transform_context,abbreviations=abbreviations)
    assert built['prompt']==request['prompt'] and built['schema']==request['schema']
    proposed=apply(built)
    bound=request['reservation']+len(json.dumps(proposed['schema']).encode())-len(json.dumps(request['schema']).encode())
    available=None if limits is None else limits.max_tokens-limits.tokens-limits.reserved_tokens
    calls=None if limits is None else limits.max_calls-limits.calls-limits.reserved_calls
    now=time.monotonic()
    seconds=min(deadline-now if deadline is not None else float('inf'),
        limits.deadline-now-2 if limits is not None else float('inf'))
    reason=('token_budget' if available is not None and bound>available else
            'call_budget' if calls is not None and calls<1 else
            'deadline' if seconds<=0 else None)
    result=dict(request)
    if reason is None:result.update(schema=proposed['schema'],reservation=bound)
    metadata=dict(policy='existing_citation_validator_schema_v1',status='fallback' if reason else 'applied',
        reason=reason,candidate_ids=[c['candidate_id'] for c in cards],
        original_schema_sha256=digest(request['schema']),proposed_schema_sha256=digest(proposed['schema']),
        dispatched_schema_sha256=digest(result['schema']),original_reservation=request['reservation'],
        proposed_reservation=bound,dispatched_reservation=result['reservation'],token_headroom=available,
        call_headroom=calls,deadline_seconds=seconds if seconds!=float('inf') else None,
        source_sha256=digest(cards),prompt_sha256=hashlib.sha256(request['prompt'].encode()).hexdigest())
    return result,metadata
