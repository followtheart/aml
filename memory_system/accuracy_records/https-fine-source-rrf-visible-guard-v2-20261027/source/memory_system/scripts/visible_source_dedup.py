"""Diagnostic redundancy coverage, leaving topic reservations and scores intact."""
from app import evidence_selection as es

def coverage_cache(rows):
    result={c['id']:list(es.fragments(c)) for c in rows}
    for c in rows:
        if c.get('type')=='profile':continue
        body=es._text(c.get('_rank_text',c.get('content','')))
        for s in c.get('_packet_item',{}).get('sources',[]):
            if s.get('role')!='user':continue
            text=es._text(s.get('content',''))
            sid=s.get('source_event_id')
            if not sid and s.get('request_id') is not None and s.get('message_index') is not None:sid=f"{s['request_id']}:{s['message_index']}"
            if sid and text and text in body:
                piece=((sid,s['role']),text)
                if piece not in result[c['id']]:result[c['id']].append(piece)
    return result
