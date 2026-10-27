import copy,json
from inventory_source_admission import classify
base={'coarse':{'selected_ids':['x']},'fine':{'selected_ids':['x'],'selection':{'decisions':[]}},'listwise':{'candidate_ids':['x'],'selected_ids':['x']}}
originals=[{'role':'user','content':'my full statement'}]
candidate={'id':'x','content':'my full statement','_rank_text':'my full statement'}
checks=[]
for stage in ['coarse_limit','fine_limit','listwise_admission','listwise_selection','packet_or_catalog']:
 t=copy.deepcopy(base)
 for name,field,want in [('coarse','selected_ids','coarse_limit'),('fine','selected_ids','fine_limit'),('listwise','candidate_ids','listwise_admission'),('listwise','selected_ids','listwise_selection')]:
  if stage==want:t[name][field]=[]
 assert classify(['my full statement'],originals,{'ranked':[candidate],'cascade':t},{},[])['stage']==stage;checks.append(stage)
assert classify(['my full statement'],[{'role':'assistant','content':'my full statement'}],{}, {},[])['stage']=='no_original_user_turn'
search={'ranked':[dict(candidate,_rank_text='my full')],'cascade':base}
assert classify(['my full statement'],originals,search,{},[])['stage']=='source_rendering'
assert classify(['my full statement'],originals,search,{'s':{'role':'assistant','text':'my full statement'}},[])['stage']=='source_rendering'
assert classify(['my full statement'],originals,search,{'s':{'role':'user','text':'my full'}},[])['stage']=='source_rendering'
assert classify(['my full statement'],originals,search,{'s':{'role':'user','text':'my full statement'}},[])['stage']=='delivered_full_user_turn'
assert classify(['my full statement'],originals,search,{},[{'role':'user','text':'my full statement'}])['stage']=='delivered_constraint'
print(json.dumps(dict(status='passed',checks=11,stages=checks,assistant_and_partial_not_full=True)))
