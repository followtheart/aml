"""Blind stratified sampling and lossless text adaptation; no oracle filtering."""
import json
import random
import re
from datetime import datetime, timezone
from pathlib import Path

SEED=20261027
STRATA={'temporal-reasoning':4,'multi-session':4,'knowledge-update':2,'single-session-user':2}


def instant(text):
    clean=re.sub(r'\s*\([^)]*\)','',text)
    return datetime.strptime(clean,'%Y/%m/%d %H:%M').replace(tzinfo=timezone.utc)


def split_original(text, limit=2400):
    chunks=[]
    while len(text)>limit:
        end=text.rfind('\n',limit//2,limit)
        end=end+1 if end>=0 else limit
        piece,text=text[:end],text[end:]
        if not piece.strip() and chunks:chunks[-1]+=piece
        else:chunks.append(piece)
    if text:
        if not text.strip() and chunks:chunks[-1]+=text
        else:chunks.append(text)
    return chunks


def prepare(path):
    rows=json.loads(Path(path).read_text(encoding='utf-8'));rng=random.Random(SEED)
    selected=[];seen=set()
    for kind,count in STRATA.items():
        choices=[r for r in rows if r['question_type']==kind and not str(r['question_id']).endswith('_abs')]
        rng.shuffle(choices)
        for r in choices:
            if seen&set(r['haystack_session_ids']):continue
            selected.append(r);seen.update(r['haystack_session_ids']);count-=1
            if not count:break
        if count:raise ValueError('insufficient disjoint histories')
    plan=[]
    for r in selected:
        sessions=[];source_texts={};gold=[];source_sessions={};originals=[]
        assert len(r['haystack_sessions'])==len(r['haystack_dates'])==len(r['haystack_session_ids'])
        for sid,date,messages in zip(r['haystack_session_ids'],r['haystack_dates'],r['haystack_sessions']):
            session=[];stamp=int(instant(date).timestamp()*1000)
            for index,message in enumerate(messages):
                chunks=split_original(message['content'])
                assert ''.join(chunks)==message['content']
                mids=[]
                for ci,chunk in enumerate(chunks):
                    mid=f'{sid}:{index}:{ci}';mids.append(mid)
                    session.append({'role':message['role'],'speaker_id':message['role'],'content':chunk,
                                    'timestamp':stamp,'message_id':mid})
                    source_texts[mid]=chunk;source_sessions[mid]=sid
                    if message.get('has_answer'):gold.append(mid)
                originals.append({'session_id':sid,'message_index':index,'chunks':mids,
                                  'text':message['content']})
            sessions.append(session)
        qa={'qa_id':'0','question':r['question'],'answer':r['answer'],'question_date':instant(r['question_date']).isoformat(),
            'category':r['question_type'],'scoring':'binary','evidence':gold,'answer_session_ids':r['answer_session_ids']}
        plan.append({'conversation_id':r['question_id'],'sessions':sessions,'session_ids':r['haystack_session_ids'],
                     'qa':[qa],'source_texts':source_texts,'source_sessions':source_sessions,'originals':originals})
    return plan
