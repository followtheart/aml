"""Prevent email introductions from masquerading as complete evidence delivery."""
from audit_https_funnel import norm,turn_coverage,snippet_transport_stage

def main():
    turn=norm('Please polish this email.\n\nI had knee surgery after a softball injury.')
    raw={('req',0):dict(role='user',content=turn)}
    def card(text,index=0,role='user'):
        return dict(id=str(index)+text[:3],text=text,role=role,request_id='req',message_index=index)
    intro=card('Please polish this email.')
    cover=turn_coverage([turn],[intro],raw)
    assert not cover[0]['full'] and 0<cover[0]['fraction']<0.5
    base=dict(observed_add=1,observed_search=True,snippet_turns=1,catalog_hits=[intro['id']],catalog_turn_coverage=cover)
    assert snippet_transport_stage(base)=='partial:source_catalog'
    body=card('I had knee surgery after a softball injury.')
    joined=turn_coverage([turn],[intro,body],raw)
    assert joined[0]['full']
    assert turn_coverage([turn],[intro,body,body],raw)==joined
    wrong=turn_coverage([turn],[card(intro['text'],1),card(body['text'],2)],raw)
    assert wrong[0]['covered_chars']==0 and not wrong[0]['full']
    assert snippet_transport_stage(dict(base,catalog_turn_coverage=wrong))=='fragment:unattributed_source_catalog'
    assert turn_coverage([turn],[card(turn.upper())],raw)[0]['full']
    assert not turn_coverage([turn],[card(turn,role='assistant')],raw)[0]['full']
    repeat=turn_coverage(['yes yes'],[card('yes')],{('req',0):dict(role='user',content='yes yes')})
    assert not repeat[0]['full']
    assert not turn_coverage([''],[intro],raw)[0]['full']
    assert snippet_transport_stage(dict(base,constraint_catalog_hits=['r0'],constraint_turn_coverage=[dict(full=True,covered_chars=20)]))=='delivered:constraint_catalog'
    print('PASS: prefix, joined excerpts, duplicates, provenance, role, repeated text, empty turn and full constraint precedence')
if __name__=='__main__':main()
