"""Diagnostic current-request card, only at an existing citation-recovery call."""
import json
from app import answer_choice as ac

def augment(prompt,question):
    marker='\nOriginal source cards:\n'
    if not isinstance(question,str) or not question.strip() or ac._constraint_spans(question):return prompt,None
    prefix,raw=prompt.split(marker,1);cards=json.loads(raw)
    if any(c['id']=='q0' for c in cards):return prompt,None
    card=dict(id='q0',role='user',text=question,item_ids=[],request_id=None,message_index=None,timestamp=None,source_scope='current_request',ephemeral=True)
    changed=prefix+marker+json.dumps(cards+ac._cards({'q0':card}),ensure_ascii=False)
    if len(changed.encode())>48000:return prompt,None
    return changed,card
