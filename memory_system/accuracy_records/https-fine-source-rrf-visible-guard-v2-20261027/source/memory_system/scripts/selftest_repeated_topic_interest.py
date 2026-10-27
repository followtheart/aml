import copy,json
from repeated_topic_interest import supports
base=dict(check_type='premise',premise_type='interest',claim='your interest in Scandinavian crime novels',sources=[dict(id='s1',role='user',quote='What gives Scandinavian crime novels their atmosphere?',context='What gives Scandinavian crime novels their atmosphere?'),dict(id='s2',role='user',quote='Can you give an example of a Scandinavian crime novel?',context='Can you give an example of a Scandinavian crime novel?')])
assert supports(base)
negatives=[]
for kind in ['single','duplicate_text','same_id','assistant','third_party','assignment','negated','hypothetical','qualified','ownership','partial_topic','quote_mismatch']:
 c=copy.deepcopy(base)
 if kind=='single':c['sources']=c['sources'][:1]
 if kind=='duplicate_text':c['sources'][1]=dict(c['sources'][0],id='s2')
 if kind=='same_id':c['sources'][1]['id']='s1'
 if kind=='assistant':c['sources'][1]['role']='assistant'
 if kind in ['third_party','assignment','negated','hypothetical']:
  prefix={'third_party':'My friend asks: ','assignment':'For my homework: ','negated':'I am not interested. ','hypothetical':'Imagine this: '}[kind]
  for s in c['sources']:s['context']=prefix+s['context']
 if kind=='qualified':c['claim']='your interest in Scandinavian crime novels and collecting first editions'
 if kind=='ownership':c['premise_type']='ownership'
 if kind=='partial_topic':c['claim']='your interest in Scandinavian historical novels'
 if kind=='quote_mismatch':c['sources'][1]['context']='unrelated'
 assert not supports(c),kind;negatives.append(kind)
print(json.dumps(dict(status='passed',positive=1,negative=len(negatives),rejected=negatives)))
