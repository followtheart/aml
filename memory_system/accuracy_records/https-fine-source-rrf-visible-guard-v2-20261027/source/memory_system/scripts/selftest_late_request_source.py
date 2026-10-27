import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from late_request_source import augment
cards=[dict(id='s0',role='user',text='An unchanged source.')]
p='Question and claims stay unchanged\nOriginal source cards:\n'+json.dumps(cards)
question='How can I manage sudden anxiety from old memories?'
new,card=augment(p,question);prefix,raw=new.split('\nOriginal source cards:\n');added=json.loads(raw)
assert prefix==p.split('\nOriginal source cards:\n')[0] and added[:-1]==cards
assert added[-1]['text']==question and added[-1]['id']=='q0'
assert card['ephemeral'] and card['source_scope']=='current_request' and card['request_id'] is None
assert cards==[dict(id='s0',role='user',text='An unchanged source.')]
for question in ['', 'Please forget that I like chess.', 'x'*49000]:assert augment(p,question)==(p,None)
assert augment(new,'Another question?')==(new,None)
q='If someone has anxiety, what could help?';changed,card=augment(p,q);assert card['text']==q and 'If someone' in changed
print(json.dumps({'status':'passed','checks':['source_prefix_preserved','literal_question','ephemeral_provenance','input_immutable','empty_skip','forget_skip','byte_limit_skip','collision_skip','hypothetical_not_rewritten'],'caveat':'Hypothetical card is retained as literal data; support still requires independent verification.'}))
