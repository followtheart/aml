"""Default-off late admission of one whole source-scored coarse outsider.

All original fine/main units, graph context, scopes and budgets remain protected.
A source score nominates a candidate for main review, never a personal fact.
"""
import asyncio
import copy
import hashlib
import math
import re
import time
from . import answer_context, budget, config, cross_encoder, fine_evidence, local_candidate_display
from .conversation_context import source_key, usable
from .fine_context_reserve import full_user

def build_views(coarse,user_id,query,*,max_sources=2):
    if max_sources is not None and (type(max_sources) is not int or max_sources < 1):
        raise ValueError('Source cap must be positive or None for all eligible sources')
    views={};members={};omitted=[]
    for candidate in coarse:
        item=candidate.get('_packet_item',{});sources=[s for s in item.get('sources',[]) if s.get('content')]
        if (candidate.get('view_status')=='stale' or item.get('is_constraint') or
                any(not usable(s,user_id) or s['content'] not in item.get('content','') for s in sources)):
            continue
        chosen=[]
        for s in sources:
            if not full_user(s) or s['content'] not in candidate.get('_rank_text',''):continue
            key=hashlib.sha256(source_key(s).encode()).hexdigest()
            if key in chosen:continue
            text=answer_context.with_evidence('Conversation excerpt:',[s])
            if (len(text.encode())>config.CE_MAX_DOCUMENT_BYTES or
                    cross_encoder.request_size(query,[text])>config.CE_MAX_REQUEST_BYTES):
                omitted.append(dict(candidate_id=candidate['id'],source_key=key,reason='input_budget'));continue
            views.setdefault(key,dict(id=key,text=text,source=copy.deepcopy(s)))
            chosen.append(key)
            if max_sources is not None and len(chosen)>=max_sources:break
        members[candidate['id']]=chosen
    batches=[];batch=[]
    for view in views.values():
        if batch and (len(batch)>=config.CE_BATCH_SIZE or cross_encoder.request_size(query,[v['text'] for v in batch+[view]])>config.CE_MAX_REQUEST_BYTES):
            batches.append(batch);batch=[]
        batch.append(view)
    if batch:batches.append(batch)
    return views,members,batches,omitted

def identity(source):
    return tuple(source.get(key) for key in
                 ('user_id', 'request_id', 'message_index', 'source_event_id', 'role'))


def normalized(text):
    return re.sub(r'\s+', ' ', text).strip()


def novel_views(views, members, submitted, query, *, request_size, batch_limit, byte_limit):
    known = [source for candidate in submitted
             for source in candidate.get('_packet_item', {}).get('sources', []) if source.get('content')]
    redundant = {key for key, view in views.items() if any(
        identity(view['source']) == identity(old)
        and normalized(view['source']['content']) in normalized(old['content']) for old in known)}
    filtered_members = {mid: [key for key in keys if key not in redundant] for mid, keys in members.items()}
    used = {key for keys in filtered_members.values() for key in keys}
    filtered = {key: view for key, view in views.items() if key in used}
    batches, batch = [], []
    for key, view in filtered.items():
        if request_size(query, [view['text']]) > byte_limit:
            raise ValueError('Source view exceeds the original CE request limit')
        proposed = batch + [key]
        if batch and (len(proposed) > batch_limit or
                request_size(query, [filtered[mid]['text'] for mid in proposed]) > byte_limit):
            batches.append(batch)
            batch = []
        batch.append(key)
    if batch:
        batches.append(batch)
    return dict(views=filtered, members=filtered_members, batches=batches,
                represented_view_ids=sorted(redundant))


def schedule_bounds(batch_bounds, *, main_reservation, token_headroom, call_headroom,
                    deadline_seconds, main_seconds, ce_seconds):
    total = sum(batch_bounds)
    seconds = ce_seconds if deadline_seconds is None else min(ce_seconds, max(0, deadline_seconds-main_seconds))
    reason = ('no_source_views' if not batch_bounds else
              'token_headroom' if token_headroom is not None and total + main_reservation > token_headroom else
              'call_headroom' if call_headroom is not None and len(batch_bounds)+1 > call_headroom else
              'deadline_headroom' if seconds <= 0 else None)
    return dict(eligible=reason is None, reason=reason, planned_calls=len(batch_bounds),
                ce_input_bound=total, main_reservation=main_reservation,
                reserved_main_calls=1, ce_deadline_seconds=seconds,
                limitation='Conservative frozen-budget preflight, not a live scheduler or guarantee of later review capacity')


def nominate(job, scores):
    if not job['scheduling']['eligible']:
        return dict(status='skipped', reason=job['scheduling']['reason'], addition=None)
    if set(scores) != set(job['views']):
        return dict(status='fallback', reason='incomplete_or_unknown_scores', addition=None)
    if any(type(value) not in (int, float) or not math.isfinite(value) for value in scores.values()):
        return dict(status='fallback', reason='invalid_scores', addition=None)
    ranked = []
    for candidate in job['nominees']:
        keys = candidate['source_view_ids']
        if not keys or any(key not in scores for key in keys):
            return dict(status='fallback', reason='invalid_membership', addition=None)
        ranked.append(dict(id=candidate['id'], score=max(scores[key] for key in keys),
                           ce_rank=candidate['ce_rank']))
    ranked.sort(key=lambda row: (-row['score'], row['ce_rank']))
    if not ranked:
        return dict(status='fallback', reason='no_nominees', addition=None)
    return dict(status='nominated', addition=ranked[0]['id'], ranking=ranked,
                policy='max_novel_complete_user_source_ce_then_original_ce_order',
                factual_support_inferred=False)


def prepare(req, plan, coarse, fine, original, headroom, cascade):
    """Enumerate all affordable whole candidates before source scoring."""
    original_request = cascade.listwise_request(req, plan, original)
    encoded, original_audit = local_candidate_display.request(req, plan, original, cascade)
    visible = {c['id']: fine_evidence.selection.fragments(c) for c in coarse}
    excluded = {c['id'] for c in fine + original}
    feasible, audits, decisions = [], {}, []
    for rank,candidate in enumerate(sorted(coarse,key=fine_evidence.key,reverse=True),1):
        budget.check()
        mid = candidate['id']
        if mid in excluded:
            continue
        score = candidate.get('_ce_score')
        reason, details = None, {}
        if type(score) not in (int,float) or not math.isfinite(score): reason = 'no_finite_ce'
        elif not visible[mid]: reason = 'no_visible_source'
        elif fine_evidence.dependencies(candidate): reason = 'not_singleton'
        elif fine_evidence.selection.represented_by(candidate,original,visible): reason = 'represented_source'
        elif len(original)+1 > config.CASCADE_LLM_LIMIT+config.FINE_EXTRA_LIMIT: reason = 'count_budget'
        else:
            request,audit = local_candidate_display.request(req,plan,original+[candidate],cascade)
            details = dict(trial_bytes=len(request['prompt'].encode()),reservation=request['reservation'])
            if details['trial_bytes'] > config.RERANK_MAX_PROMPT_BYTES: reason = 'prompt_budget'
            elif headroom['token_headroom'] is not None and request['reservation'] > headroom['token_headroom']:
                reason = 'token_budget'
            elif local_candidate_display.missing_graph(original_audit['decoded_graph'],audit['decoded_graph']):
                reason = 'original_graph_context_loss'
            else:
                feasible.append(candidate)
                audits[mid] = dict(**details,ce_rank=rank,display_audit=audit)
        decisions.append(dict(id=mid,ce_rank=rank,reason=reason or 'affordable',**details))
    query = cascade.query_text(req,plan)
    views,members,_,omitted = build_views(feasible,req.user_id,query,max_sources=None)
    novel = novel_views(views,members,original,query,request_size=cross_encoder.request_size,
                        batch_limit=config.CE_BATCH_SIZE,byte_limit=config.CE_MAX_REQUEST_BYTES)
    nominees = [dict(id=c['id'],candidate=c,source_view_ids=novel['members'][c['id']],**audits[c['id']])
                for c in feasible if novel['members'].get(c['id'])]
    reservations = [original_request['reservation']]+[n['reservation'] for n in nominees]
    bounds = [cross_encoder.request_size(query,[novel['views'][key]['text'] for key in batch])
              for batch in novel['batches']]
    scheduling = schedule_bounds(bounds,main_reservation=max(reservations),
        token_headroom=headroom['token_headroom'],call_headroom=headroom['call_headroom'],
        deadline_seconds=headroom['deadline_seconds'],main_seconds=config.RERANK_DEADLINE_SECONDS+2,
        ce_seconds=config.CE_DEADLINE_SECONDS)
    if len(query.encode()) > config.CE_MAX_DOCUMENT_BYTES:
        scheduling.update(eligible=False,reason='query_budget')
    return dict(ce_query=query,decisions=decisions,nominees=nominees,views=novel['views'],
        batches=novel['batches'],batch_bounds=bounds,scheduling=scheduling,
        omitted_source_views=omitted,represented_view_ids=novel['represented_view_ids'],
        original_display_audit=original_audit,encoded_original_bytes=len(encoded['prompt'].encode()))


async def append(req, plan, coarse, fine, original, cascade):
    """Fail closed on preparation/provider/budget failures, including partial scores."""
    details = dict(status='disabled',addition=None,display_ids=False,batches=[],
                   original_main_ids=[c['id'] for c in original])
    if not config.SOURCE_CE_LATE_ADMISSION:
        return original,details
    if config.LISTWISE_STAGED or not original:
        details.update(status='skipped',reason='unsupported_staged_or_empty_main')
        return original,details
    limits = budget.current.get()
    try:
        def remaining():
            return dict(token_headroom=limits.max_tokens-limits.tokens-limits.reserved_tokens if limits else None,
                        call_headroom=limits.max_calls-limits.calls-limits.reserved_calls if limits else None,
                        deadline_seconds=max(0,limits.deadline-time.monotonic()) if limits else None)
        headroom = remaining()
        job = prepare(req,plan,coarse,fine,original,headroom,cascade)
        # Preparation itself consumes time; schedule against current remaining time.
        current = remaining()
        schedule = schedule_bounds(job['batch_bounds'],main_reservation=job['scheduling']['main_reservation'],
            **current,main_seconds=config.RERANK_DEADLINE_SECONDS+2,ce_seconds=config.CE_DEADLINE_SECONDS)
        if not job['scheduling']['eligible']:
            schedule = job['scheduling']
        details.update(status='prepared',scheduling=schedule,decisions=job['decisions'],
            eligible_ids=[c['id'] for c in job['nominees']],query=job['ce_query'],
            source_view_ids=list(job['views']),original_headroom=headroom)
        if not schedule['eligible']:
            details.update(status='skipped',reason=schedule['reason'])
            return original,details
        reserved = False
        scores = {}
        try:
            if limits:
                limits.reserve_calls(1)
                try:
                    limits.reserve_tokens(schedule['main_reservation'])
                except BaseException:
                    limits.release_calls(1)
                    raise
                reserved = True
            async with asyncio.timeout(schedule['ce_deadline_seconds']):
                for index,keys in enumerate(job['batches']):
                    texts = [job['views'][key]['text'] for key in keys]
                    entry = dict(batch=index,view_ids=keys,bound=job['batch_bounds'][index],status='started')
                    details['batches'].append(entry)
                    try:
                        values = await cross_encoder.rerank(job['ce_query'],texts,
                            stage='search.source_nomination',rate_limit_retries=0)
                        if len(values)!=len(keys):
                            raise ValueError('Incomplete source scores')
                        scores.update(zip(keys,values,strict=True))
                        entry.update(status='ok',scores=values)
                    except BaseException as exc:
                        entry.update(status='error',error=type(exc).__name__)
                        raise
        finally:
            if reserved:
                limits.release_calls(1)
                limits.release_tokens(schedule['main_reservation'])
        decision = nominate(job,scores)
        details.update(decision)
        if not decision['addition']:
            return original,details
        nominee = next(c for c in job['nominees'] if c['id']==decision['addition'])
        proposed = original+[nominee['candidate']]
        request,audit = local_candidate_display.request(req,plan,proposed,cascade)
        available = remaining()
        if (len(request['prompt'].encode())>config.RERANK_MAX_PROMPT_BYTES or
                available['token_headroom'] is not None and request['reservation']>available['token_headroom'] or
                available['call_headroom'] is not None and available['call_headroom']<1 or
                available['deadline_seconds'] is not None and available['deadline_seconds']<=2):
            raise budget.BudgetExceeded('Original main reservation unavailable after source scoring')
        if local_candidate_display.missing_graph(job['original_display_audit']['decoded_graph'],audit['decoded_graph']):
            raise ValueError('Original displayed graph changed')
        details.update(status='admitted',display_ids=True,display_audit=audit,
                       proposed_main_ids=[c['id'] for c in proposed])
        return proposed,details
    except Exception as exc:
        details.update(status='fallback',reason=type(exc).__name__,addition=None,display_ids=False)
        return original,details
