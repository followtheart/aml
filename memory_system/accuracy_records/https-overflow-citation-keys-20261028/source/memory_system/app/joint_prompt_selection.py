"""Select complete groups in relevance order within exact request budgets."""
import time
from . import evidence_selection, fine_evidence


def preserves(before, after):
    """Require every existing graph value and list entry to remain in order."""
    if isinstance(before, dict):
        return isinstance(after, dict) and all(k in after and preserves(v, after[k]) for k,v in before.items())
    if isinstance(before, list):
        if not isinstance(after, list):return False
        cursor=iter(after)
        return all(any(v==other for other in cursor) for v in before)
    return before==after


def choose(coarse, pool_ids, original_fine, *, rank_ids, protected_fine_ids, mandatory_main_ids,
           render, graph, token_headroom, byte_limit=24000, count_limit=24, fine_limit=24, deadline=None):
    by_id={c['id']:c for c in coarse};ranks={mid:i+1 for i,mid in enumerate(rank_ids)}
    assert len(by_id)==len(coarse) and set(by_id)==set(ranks)
    assert set(pool_ids)|set(protected_fine_ids)|set(mandatory_main_ids)<=set(by_id)
    ordered=lambda ids:sorted((by_id[mid] for mid in ids),key=lambda c:ranks[c['id']])
    fine_required=set(protected_fine_ids);main_ids=set(mandatory_main_ids)
    # Existing reservations and user constraints precede relevance admission.
    for mid in list(main_ids):
        group,missing=fine_evidence.closure(by_id[mid],by_id)
        if missing:return dict(status='fallback',reason='mandatory_missing_dependency')
        main_ids.update(group)
    fine_required.update(main_ids)
    if len(fine_required)>fine_limit or len(main_ids)>count_limit:
        return dict(status='fallback',reason='mandatory_count_budget')
    selected=ordered(main_ids);request=render(selected);size=len(request['prompt'].encode())
    if size>byte_limit or request['reservation']>token_headroom:
        return dict(status='fallback',reason='mandatory_request_budget')
    eligible=set(pool_ids);visible={mid:evidence_selection.fragments(c) for mid,c in by_id.items()}
    decisions=[];last_rejections=[]
    while len(selected)<count_limit:
        if deadline is not None and time.monotonic()>=deadline:
            return dict(status='fallback',reason='selection_deadline')
        best=None;last_rejections=[];current_graph=graph(selected)
        for mid in sorted(eligible-main_ids,key=ranks.get):
            if deadline is not None and time.monotonic()>=deadline:
                return dict(status='fallback',reason='selection_deadline')
            root=by_id[mid];group,missing=fine_evidence.closure(root,by_id);new=group-main_ids
            if missing or not new:
                last_rejections.append(dict(id=mid,reason='missing_dependency' if missing else 'already_present'));continue
            if not visible[mid] or evidence_selection.represented_by(root,selected,visible):
                last_rejections.append(dict(id=mid,reason='no_novel_visible_evidence'));continue
            if len(main_ids|new)>count_limit or len(fine_required|main_ids|new)>fine_limit:
                last_rejections.append(dict(id=mid,reason='group_count_budget'));continue
            trial=ordered(main_ids|new)
            if not preserves(current_graph,graph(trial)):
                last_rejections.append(dict(id=mid,reason='existing_graph_context_loss'));continue
            candidate_request=render(trial);new_size=len(candidate_request['prompt'].encode())
            if new_size>byte_limit or candidate_request['reservation']>token_headroom:
                last_rejections.append(dict(id=mid,reason='prompt_budget' if new_size>byte_limit else 'token_budget'));continue
            marginal=new_size-size
            # Budget is a feasibility constraint, never a reward for brevity.
            # Start at the earliest rank again after each admission: sharing
            # may make an earlier, previously oversized group feasible.
            best=(mid,new,trial,candidate_request,new_size,marginal)
            break
        if best is None:break
        mid,new,selected,request,size,marginal=best
        main_ids.update(new)
        decisions.append(dict(id=mid,group_ids=sorted(new),marginal_prompt_bytes=marginal,
                              relevance_rank=ranks[mid],prompt_bytes=size))
    # Main is the first budgeted selection. Preserve baseline fine constraints
    # and use remaining fine seats for original then alternative complete groups.
    fine_ids=fine_required|main_ids
    refill=list(dict.fromkeys([c['id'] for c in original_fine]+sorted(eligible,key=ranks.get)))
    for mid in refill:
        if mid in fine_ids:continue
        group,missing=fine_evidence.closure(by_id[mid],by_id)
        if not missing and len(fine_ids|group)<=fine_limit:fine_ids.update(group)
    fine=ordered(fine_ids)
    assert set(main_ids)<=fine_ids and set(protected_fine_ids)<=fine_ids and set(mandatory_main_ids)<=main_ids
    return dict(status='selected',reason=None,fine=fine,submitted=selected,request=request,
                decisions=decisions,rejections=last_rejections,protected_fine_ids=sorted(protected_fine_ids),
                mandatory_main_ids=sorted(mandatory_main_ids))
