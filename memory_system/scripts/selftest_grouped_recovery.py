import json
from probe_grouped_recovery import transform,expand
claims=[dict(claim_id='A:0',text='your work',option='A. First advice'),dict(claim_id='B:0',text='your work',option='B. Other advice'),dict(claim_id='C:0',text='your previous work',option='C. Past')]
p='Instruction\nClaims:\n'+json.dumps(claims)+'\nOriginal source cards:\n'+json.dumps([{'id':'s1','text':'unchanged'}])
g,m=transform(p,'grouped');assert m=={'A:0':['A:0','B:0'],'C:0':['C:0']};assert g.split('\nOriginal source cards:\n')[1]==p.split('\nOriginal source cards:\n')[1]
c,cm=transform(p,'claim_only');assert len(cm)==3;assert transform(p,'original')[0]==p
response={'matches':[{'claim_id':k,'premise_type':'occupation','supported':False,'citations':[]} for k in m]};expanded=expand(response,m);assert [x['claim_id'] for x in expanded]==['A:0','B:0','C:0'];assert not any(x['supported'] for x in expanded)
response['matches'][0]['citations'].append({'source_id':'s1','quote':'unchanged','basis':'self_report','subject':'current_user'});expanded=expand(response,m);assert expanded[0]['citations']==expanded[1]['citations']
print(json.dumps({'status':'passed','checks':['exact_text_grouping','distinct_history_retained','source_bytes_unchanged','original_identity','false_support_preserved','citation_expansion']}))
