"""Observable retrieval coverage, never a claim of semantic sufficiency."""
from . import personal_evidence


def source_keys(sources):
    return {s.get('source_event_id') or f"{s['request_id']}:{s['message_index']}"
            for s in sources if s.get('content')}


def matches(text, query):
    wanted = personal_evidence.terms(query)
    hits = wanted & personal_evidence.terms(text)
    return bool(wanted and len(hits) >= min(2, len(wanted)) and len(hits) / len(wanted) >= .5)


def annotate(item, requirements, sources=()):
    text = item.get('_rank_text', item.get('content', ''))
    text += '\n' + '\n'.join(s.get('content', '') for s in sources)
    covered = set(item.get('_coverage_ids', []))
    covered.update(r['id'] for r in requirements if matches(text, r['text']))
    item['_coverage_ids'] = sorted(covered)
    return item


def report(requirements, candidates, packed=()):
    rows = []
    for requirement in requirements:
        rid = requirement['id']
        found = [c['id'] for c in candidates if rid in c.get('_coverage_ids', [])]
        included = [c['id'] for c in packed if rid in c.get('coverage_ids', [])]
        rows.append(dict(requirement, candidate_ids=found, included_ids=included,
                         status='packed' if included else 'candidate_only' if found else 'missing'))
    return dict(basis='query_match_or_grounded_followup; not semantic sufficiency', requirements=rows,
                missing_ids=[r['id'] for r in rows if r['status'] != 'packed'])
