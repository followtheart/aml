import copy,json
from audit_visible_topic import canonical_plan
base={'comparisons':{'q':{'original':{'ids':['a','b'],'reservations':[{'candidate_id':'a','requirement_ids':['r2','r1'],'reasons':['fused_head','option_topic_coverage'],'selected':True},{'candidate_id':'b','requirement_ids':[],'reasons':['fused_head'],'selected':True}]}}}}
p=copy.deepcopy(base);p['comparisons']['q']['original']['reservations'].reverse();p['comparisons']['q']['original']['reservations'][1]['requirement_ids'].reverse();assert canonical_plan(p)==canonical_plan(base)
for label in ['ids','selected','reason','requirements']:
 p=copy.deepcopy(base);a=p['comparisons']['q']['original']
 if label=='ids':a['ids'].reverse()
 if label=='selected':a['reservations'][0]['selected']=False
 if label=='reason':a['reservations'][0]['reasons']=['fused_head']
 if label=='requirements':a['reservations'][0]['requirement_ids']=['r1']
 assert canonical_plan(p)!=canonical_plan(base),label
p=copy.deepcopy(base);p['comparisons']['q']['original']['reservations'].append(copy.deepcopy(p['comparisons']['q']['original']['reservations'][0]))
try:canonical_plan(p)
except ValueError:pass
else:raise AssertionError('Duplicate accepted')
print(json.dumps({'status':'passed','unordered_metadata_allowed':1,'semantic_changes_distinguished':4,'duplicate_rejected':1}))
