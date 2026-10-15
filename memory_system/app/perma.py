"""PERMA checkpoint MCQs with isolated histories and no oracle preference injection."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import re

PROTOCOL = 'perma-mcq-local-v1'
PERIOD_HOURS = {'morning': 6, 'afternoon': 15, 'evening': 19}


def date_ms(text):
    match = re.fullmatch(r'(\d{4}-\d{2}-\d{2})(?: (Morning|Afternoon|Evening))?', text, re.I)
    if not match:
        raise ValueError('Unsupported PERMA date/period')
    dt = datetime.strptime(match[1], '%Y-%m-%d').replace(tzinfo=timezone.utc,
            hour=PERIOD_HOURS.get((match[2] or '').lower(), 0))
    return int(dt.timestamp() * 1000)


def options(text):
    if not isinstance(text, str):
        raise ValueError('PERMA options must be labelled text')
    matches = list(re.finditer(r'(?m)^([A-Z]):[ \t]*', text))
    if not matches or text[:matches[0].start()].strip():
        raise ValueError('Missing/invalid option prefix')
    letters = [m[1] for m in matches]
    if not 2 <= len(letters) <= 26 or letters != [chr(65+i) for i in range(len(letters))]:
        raise ValueError('Non-contiguous or duplicate option labels')
    result=[]
    for i,m in enumerate(matches):
        body=text[m.end():matches[i+1].start() if i+1<len(matches) else len(text)].strip()
        if not body:
            raise ValueError('Empty option')
        result.append(f'{m[1]}. {body}')
    return result


def convert(row, meta, uid, variant, revision):
    task, stage = row['task_id'], row['type']
    if type(stage) is not int or stage not in (1,2,3):
        raise ValueError('Unsupported checkpoint')
    if meta['task_id'] != task or meta['task_type'] != stage or meta['scope'] != 'overall':
        raise ValueError('Task/metadata identity mismatch')
    # Query dates have day precision. The supplied checkpoint includes same-day
    # sessions; preserve those and reject later dates, rather than assuming midnight.
    cutoff = date_ms(meta['question_date']) + 86400000
    opts=options(meta['options']);gold=meta['gold_label']
    if gold not in [chr(65+i) for i in range(len(opts))]:
        raise ValueError('Gold label outside options')
    sessions=[]
    # Upstream evaluation.py explicitly hides memory for type 1.
    if stage != 1:
        for i,pair in enumerate(row['context']):
            if not isinstance(pair,list) or len(pair)!=2 or not isinstance(pair[0],list) or not pair[0]:
                raise ValueError('Malformed context session')
            turns, date=pair;ts=date_ms(date)
            if ts >= cutoff:
                raise ValueError('Context contains a date after the question day')
            messages=[]
            for turn in turns:
                if turn.get('role') not in ('user','assistant') or not isinstance(turn.get('content'),str) or not turn['content'].strip():
                    raise ValueError('Invalid original dialogue turn')
                messages.append(dict(role=turn['role'],content=turn['content'],timestamp=ts))
            sessions.append((ts,i,messages,date))
        if not sessions:
            raise ValueError('Memory-enabled checkpoint has empty history')
    sessions.sort(key=lambda x:(x[0],x[1]))
    ident=f'{uid}:{variant}:{task}:{stage}'
    return dict(conversation_id=ident,dataset='perma',protocol=PROTOCOL,source_user_id=uid,
                source_revision=revision,history_variant=variant,checkpoint=stage,
                cutoff_seconds=cutoff//1000,history_policy='no_memory' if stage==1 else 'provided_checkpoint_context',
                temporal_policy='provided checkpoint; inclusive question day; period hours follow upstream convention',
                sessions=[x[2] for x in sessions],session_ids=[f'session-{x[1]}' for x in sessions],
                source_session_dates=[x[3] for x in sessions],
                qa=[dict(id=ident,question=meta['question'],question_date=meta['question_date'],
                         qa_type='single_choice',scoring='choice',options=opts,gold_labels=[gold],
                         answer=opts[ord(gold)-65],category=f'{"multi" if task.startswith("MD-") else "single"}/checkpoint-{stage}',
                         task_id=task,checkpoint=stage)])


def conversations(index_path):
    index_path=Path(index_path);index=json.loads(index_path.read_text());root=index_path.parent.resolve()
    if index.get('dataset')!='perma':
        raise ValueError('Expected PERMA download index')
    def load(relative):
        path=(root/relative).resolve()
        if not path.is_relative_to(root) or relative not in index['files']:
            raise ValueError('Invalid/unmanifested source path')
        raw=path.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=index['files'][relative]['sha256']:
            raise ValueError('Source hash mismatch: '+relative)
        return json.loads(raw)
    seen=set()
    for relative in index['histories']:
        uid=relative.split('/')[1]
        for row in load(relative)['overall']:
            task,stage=row['task_id'],row['type']
            if not re.fullmatch(r'(SD-[A-Za-z]+|MD)-task-\d+',task):
                raise ValueError('Invalid task ID')
            meta=load(f'evaluation/{uid}/meta/overall/{task}_{stage}.json')
            conv=convert(row,meta,uid,index['variant'],index['revision'])
            if conv['conversation_id'] in seen:
                raise ValueError('Duplicate checkpoint')
            seen.add(conv['conversation_id'])
            yield conv
