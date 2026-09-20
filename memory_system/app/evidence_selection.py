"""Score-ordered selection with witnessed reservations and visible-source coverage."""
import re
from . import budget, personal_evidence as pe


def _text(value):
    return re.sub(r'\s+', ' ', value).strip()


def fragments(row):
    """Only source text actually visible to the ranker can represent evidence."""
    body = _text(row.get('_rank_text', row.get('content', '')))
    result = []
    for source in row.get('_packet_item', {}).get('sources', []):
        text = _text(source.get('content') or '')
        if not text or text not in body:
            continue
        sid = source.get('source_event_id')
        if not sid and source.get('request_id') is not None and source.get('message_index') is not None:
            sid = f"{source['request_id']}:{source['message_index']}"
        if sid:
            result.append(((sid, source.get('role')), text))
    return result


def represented_by(row, selected, cache=None):
    visible = (lambda c: cache[c['id']]) if cache is not None else fragments
    pieces = visible(row)
    if not pieces:
        return []
    qualifiers = ('polarity', 'epistemic_status', 'resolution_status', 'valid_from', 'valid_to',
                  'temporal', 'event_time', 'state', 'knowledge_status')
    compatible = [c for c in selected if all(c.get(k) == row.get(k) for k in qualifiers)]
    # Source equality must not erase an additional recorded graph obligation.
    bridges = set(row.get('_bridge_ids', [])) | set(row.get('_relation_path_ids', []))
    seen_bridges = {mid for c in compatible for mid in
                    [c['id'], *c.get('_bridge_ids', []), *c.get('_relation_path_ids', [])]}
    if not bridges <= seen_bridges:
        return []
    owners = set()
    for source, text in pieces:
        owner = next((c['id'] for c in compatible
                      if any(key == source and text in content for key, content in visible(c))), None)
        if owner is None:
            return []
        owners.add(owner)
    return sorted(owners)


def select(rows, count, requirements, key, reserve_ids=(), *, carry_ids=(), fused_ids=(), trace=None):
    """Keep score order; defer only evidence completely represented by selected units.

    Coverage labels are a soft retrieval diagnostic. Only source-witnessed labels
    can reserve slots. Reservations are explicit and portable to the next cap.
    """
    ordered = sorted(rows, key=key, reverse=True)
    visible = {c['id']: fragments(c) for c in ordered}
    count = max(0, count)
    by_id = {c['id']: c for c in ordered}
    reservations = {}
    for requirement in requirements:
        champion = next((c for c in ordered if requirement['id'] in c.get('_supported_coverage_ids', [])), None)
        if champion is not None:
            entry = reservations.setdefault(champion['id'], dict(requirement_ids=[], reasons=[]))
            entry['requirement_ids'].append(requirement['id'])
            if 'supported_coverage' not in entry['reasons']:
                entry['reasons'].append('supported_coverage')
    for reason, members in (('unscored_rescue', reserve_ids), ('carried_reservation', carry_ids),
                            ('fused_head', fused_ids)):
        for mid in members:
            if mid in by_id:
                entry = reservations.setdefault(mid, dict(requirement_ids=[], reasons=[]))
                entry['reasons'].append(reason)
    reserved = [c for c in ordered if c['id'] in reservations]
    selected = reserved[:count]
    deferred = []
    for c in ordered:
        budget.check()
        if c in selected or c in reserved:
            continue
        owners = represented_by(c, selected, visible)
        if owners:
            deferred.append(c)
        elif len(selected) < count:
            selected.append(c)
    # Distinct claims from the same source remain available when capacity permits.
    for c in deferred:
        if len(selected) >= count:
            break
        selected.append(c)
    chosen_ids = {c['id'] for c in selected}
    if trace is not None:
        trace.update(policy='score_order_visible_evidence', reservations=[
            dict(candidate_id=mid, **entry, selected=mid in chosen_ids)
            for mid, entry in reservations.items()], decisions=[])
        for rank, c in enumerate(ordered, 1):
            others = [x for x in selected if x['id'] != c['id']]
            owners = represented_by(c, others, visible)
            reason = ('+'.join(reservations[c['id']]['reasons']) if c['id'] in reservations
                      else 'covered_sources' if owners else 'rank')
            trace['decisions'].append(dict(id=c['id'], rank=rank, selected=c['id'] in chosen_ids,
                reason=reason if c['id'] in chosen_ids or owners else 'capacity',
                score=c.get('_ce_score'), represented_by=owners,
                visible_sources=sorted({s[0] for s, _ in visible[c['id']]})))
    return [c for c in ordered if c['id'] in chosen_ids]


def rescue(rows, count, query, specs=()):
    """Give failed candidates bounded opportunities across distinct query facets."""
    facets = [query] + [q['text'] for q in specs if q.get('text') and q.get('origin') in
                        ('option_premise', 'option_fallback', 'sub_queries', 'grounded_followup')]
    chosen, remaining = [], list(rows)
    visible = {c['id']: fragments(c) for c in rows}
    for facet in dict.fromkeys(facets):
        if len(chosen) >= count or not remaining:
            break
        needles = pe.terms(facet)
        def rank(c):
            pieces = visible[c['id']]
            texts = [text for _, text in pieces] or [c.get('_rank_text', c['content'])]
            score = max((pe._sentence_score(s.group(), needles)[0]
                         for text in texts for s in pe._sentences(text)), default=0)
            return (score, not bool(represented_by(c, chosen, visible)),
                    bool(c.get('_selection_evidence', {}).get('user_source')),
                    c.get('_fused', 0))
        candidate = max(remaining, key=rank)
        if rank(candidate)[0] > 0:
            chosen.append(candidate)
            remaining.remove(candidate)
    # Finish in score order while preferring evidence not already represented.
    tail = sorted(remaining, key=lambda c: c.get('_fused', 0), reverse=True)
    for duplicate in (False, True):
        for c in tail:
            if len(chosen) >= count:
                return chosen
            if c not in chosen and bool(represented_by(c, chosen, visible)) == duplicate:
                chosen.append(c)
    return chosen
