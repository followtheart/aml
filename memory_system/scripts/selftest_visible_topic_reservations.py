import copy,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import evidence_selection as es
from visible_source_dedup import coverage_cache
text='I teach animation. I carry a reusable water bottle daily.'
row=dict(id='a',type='fact',content='I teach animation.',_rank_text=text,evidence=[dict(request_id='r',message_index=0,quote='I teach animation.')],_packet_item={'sources':[dict(request_id='r',message_index=0,role='user',content=text)]})
req=[dict(id='option:0',text='your reusable water bottle',origin='option_premise')]
def choose(r,expanded):
 visible=coverage_cache([r]) if expanded else {r['id']:es.fragments(r)}
 return es.option_champions([r],req,visible,2)
before=copy.deepcopy(row);assert choose(row,False)=={};assert choose(row,True)=={'a':['option:0']};assert row==before
checks=['new_visible_detail_reserved','input_unchanged']
for label in ['assistant','profile','partial','unrelated']:
 r=copy.deepcopy(row)
 if label=='assistant':r['_packet_item']['sources'][0]['role']='assistant'
 if label=='profile':r['type']='profile'
 if label=='partial':r['_rank_text']='I teach animation.'
 if label=='unrelated':r['_rank_text']='I teach animation.';r['_packet_item']['sources'][0]['content']=r['_rank_text']
 assert choose(r,True)==choose(r,False)=={},label;checks.append(label)
print(json.dumps(dict(status='passed',checks=checks,caveat='Reservation is relevance only, not proof of personal habit.')))
