import copy,json
from probe_option_prefix_admission import prefix_requirements
cases=[dict(id='option:0',text='Since you enjoy drinking water daily, your skin might benefit',origin='option_premise'),dict(id='question',text='Since you enjoy drinking water daily, what else?',origin='question'),dict(id='option:1',text='What do water, meals and sleep share?',origin='option_premise'),dict(id='option:2',text='Since then, things changed',origin='option_premise'),dict(id='option:3',text='Since you enjoy drinking water daily',origin='option_premise')]
snapshot=copy.deepcopy(cases);changed=prefix_requirements(cases)
assert cases==snapshot
assert changed[0]['text']=='Since you enjoy drinking water daily'
assert changed[1:]==cases[1:]
assert changed[0]['id']==cases[0]['id'] and changed[0]['origin']==cases[0]['origin']
print(json.dumps(dict(status='passed',cases=5,input_immutable=True,metadata_preserved=True)))
