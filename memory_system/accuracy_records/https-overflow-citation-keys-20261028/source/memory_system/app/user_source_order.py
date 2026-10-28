"""Reorder comparable coarse slots by complete visible user-source CE scores.

Original candidate CE scores, source text, metadata and pinned slots are intact.
Source and whole-candidate scores are never numerically blended. No gold labels.
"""
import hashlib
import json
import math
import re


def original_key(row):
    return (row['_ce_score'] if row.get('_ce_score') is not None else -1e30, row.get('_fused',0))


def normalized(text):
    return re.sub(r'\s+',' ',text).strip()


def identity(source):
    return tuple(source.get(k) for k in ('user_id','request_id','message_index','source_event_id','role'))


def views(coarse,user_id,*,protected_ids=(),document_limit):
    result={};members={};omissions=[];pinned=set(protected_ids)
    for row in sorted(coarse,key=original_key,reverse=True):
        mid=row['id'];score=row.get('_ce_score')
        if mid in pinned or type(score) not in (int,float) or not math.isfinite(score):
            continue
        body=normalized(row.get('_rank_text',row.get('content','')))
        item=row.get('_packet_item',{});keys=[]
        if row.get('view_status')=='stale' or row.get('_user_rule') or item.get('is_constraint'):
            continue
        for source in item.get('sources',[]):
            text=source.get('content')
            span=source.get('content_span')
            if (source.get('role')!='user' or source.get('user_id')!=user_id
                or not source.get('source_event_id') or not source.get('request_id')
                or type(source.get('message_index')) is not int or source['message_index']<0
                or not isinstance(text,str) or not text.strip() or source.get('content_omitted')
                or (source.get('content_excerpted') and not span)
                or (span and not (span.get('start')==0 and span.get('end')==span.get('original_length')==len(text)))
                or normalized(text) not in body or normalized(text) not in normalized(item.get('content',''))):
                continue
            document='Original user message (including its outer task and qualifications):\n'+text
            if len(document.encode())>document_limit:
                omissions.append(dict(candidate_id=mid,reason='whole_user_source_document_budget'));continue
            key=hashlib.sha256(json.dumps(dict(identity=identity(source),text=text),
                ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
            result.setdefault(key,dict(id=key,text=document,source=source))
            if key not in keys:keys.append(key)
        if keys:members[mid]=keys
    return dict(views=result,members=members,omissions=omissions)


def order(coarse,view_data,scores,*,protected_ids=()):
    ordered=sorted(coarse,key=original_key,reverse=True)
    original=[c['id'] for c in ordered]
    if set(scores)!=set(view_data['views']) or any(type(v) not in (int,float) or not math.isfinite(v) for v in scores.values()):
        return dict(status='fallback',reason='incomplete_or_invalid_source_scores',ordered_ids=original,source_scores={})
    pinned=set(protected_ids)
    eligible={mid:max(scores[key] for key in keys) for mid,keys in view_data['members'].items()
              if keys and mid not in pinned}
    positions=[i for i,c in enumerate(ordered) if c['id'] in eligible]
    # Stable original order breaks source-score ties without inventing a scale
    # combining source CE with summary CE or graph fusion scores.
    replacement=sorted((ordered[i] for i in positions),key=lambda c:eligible[c['id']],reverse=True)
    result=list(original)
    for i,c in zip(positions,replacement,strict=True):result[i]=c['id']
    assert set(result)==set(original) and len(result)==len(set(result))
    assert all(result[i]==mid for i,mid in enumerate(original) if i not in positions)
    return dict(status='ordered',reason=None,ordered_ids=result,source_scores=eligible,
                movable_ids=[original[i] for i in positions],pinned_ids=[mid for mid in original if mid not in eligible])


def ordering_key(proposal):
    ranks={mid:len(proposal['ordered_ids'])-i for i,mid in enumerate(proposal['ordered_ids'])}
    return lambda row:(ranks[row['id']],0)


from fractions import Fraction
source_order = order

def fused_order(coarse, view_data, scores, *, protected_ids=()):
    proposal = source_order(coarse, view_data, scores, protected_ids=protected_ids)
    if proposal['status'] != 'ordered':
        return proposal
    # movable_ids is in original whole-candidate CE/fused order. All other
    # original positions remain fixed, including reader-constraint carriers.
    original = proposal['movable_ids']
    movable = set(original)
    source = [mid for mid in proposal['ordered_ids'] if mid in movable]
    whole_rank = {mid: rank for rank, mid in enumerate(original, 1)}
    source_rank = {mid: rank for rank, mid in enumerate(source, 1)}
    fused = {mid: Fraction(1, 60 + whole_rank[mid]) + Fraction(1, 60 + source_rank[mid])
             for mid in original}
    ordered = sorted(original, key=lambda mid: fused[mid], reverse=True)
    slots = [i for i, mid in enumerate(proposal['ordered_ids']) if mid in movable]
    ids = list(proposal['ordered_ids'])
    for i, mid in zip(slots, ordered, strict=True):
        ids[i] = mid
    return dict(proposal, ordered_ids=ids, policy='equal_weight_rrf_k60',
                whole_candidate_ranks=whole_rank, user_source_ranks=source_rank,
                fused_ranks={mid: str(fused[mid]) for mid in original})
