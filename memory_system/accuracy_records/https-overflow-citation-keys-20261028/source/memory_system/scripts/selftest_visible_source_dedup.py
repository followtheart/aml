import copy,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import evidence_selection as es
from visible_source_dedup import coverage_cache
text='I enjoy cycling. I teach sociology.'
def row(mid,quote,kind='fact',role='user'):
 return dict(id=mid,type=kind,content=quote,_rank_text=text,evidence=[dict(request_id='r1',message_index=0,quote=quote)],_packet_item={'sources':[dict(request_id='r1',message_index=0,role=role,content=text)]})
a=row('a','I enjoy cycling.');b=row('b','I teach sociology.');before=copy.deepcopy([a,b]);cache=coverage_cache([a,b]);assert not es.represented_by(b,[a]);assert es.represented_by(b,[a],cache)==['a'];assert [a,b]==before
checks=['full_source_coverage','input_immutable']
for label,change in [('profile',{'type':'profile'}),('assistant',{}),('partial_render',{'_rank_text':'I teach sociology.'}),('different_origin',{}),('different_time',{'valid_from':'2025-01-01'}),('negated_state',{'polarity':'negative'}),('relation_obligation',{'_bridge_ids':['missing-relation']})]:
 x,y=copy.deepcopy(a),copy.deepcopy(b);y.update(change)
 if label=='assistant':
  for z in [x,y]:z['_packet_item']['sources'][0]['role']='assistant'
 if label=='different_origin':y['_packet_item']['sources'][0]['request_id']='r2'
 assert not es.represented_by(y,[x],coverage_cache([x,y])),label
 checks.append(label)
print(json.dumps(dict(status='passed',checks=checks)))
