"""Source-view pruning and conservative scheduling for affordable nominees."""
import re


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
