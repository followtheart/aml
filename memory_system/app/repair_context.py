"""Scope prior dialogue hints to options with an actual rejected assistant citation."""
import copy
import json

def rejected_ids(value):
    if isinstance(value,str):
        try:value=json.loads(value)
        except (ValueError,TypeError):return []
    if isinstance(value,list):return [sid for child in value for sid in rejected_ids(child)]
    if not isinstance(value,dict):return []
    errors=value.get('errors',[])
    own=[value['source_id']] if isinstance(value.get('source_id'),str) and isinstance(errors,list) and 'not_user_source' in errors else []
    return own+[sid for child in value.values() if isinstance(child,(dict,list)) for sid in rejected_ids(child)]

def targets(pending,sources,failures):
    return {letter:{sid for sid in rejected_ids(failures.get(letter)) if sid in sources
                   and (sources[sid].get('declared') or sources[sid].get('role'))=='assistant'}
            for letter in pending if any(sid in sources and (sources[sid].get('declared') or sources[sid].get('role'))=='assistant' for sid in rejected_ids(failures.get(letter)))}

def triggered_witnesses(witnesses,pending,sources,failures):
    result=copy.deepcopy({letter:items for letter,items in witnesses.items() if letter in pending})
    for letter,bad_ids in targets(pending,sources,failures).items():
        items=[item for item in result.get(letter,[]) if item.get('source_id') in sources and
               (sources[item['source_id']].get('declared') or sources[item['source_id']].get('role')) in ('user','persona')]
        candidates={}
        for bad_id in bad_ids:
            bad=sources[bad_id];index=bad.get('message_index');request=bad.get('request_id')
            if not isinstance(index,int) or not request:continue
            for sid,source in sources.items():
                local_index=source.get('message_index');text=source.get('text','')
                if (source.get('declared') or source.get('role')) not in ('user','persona'):continue
                if source.get('request_id')!=request or not isinstance(local_index,int) or not 1<=index-local_index<=4:continue
                if not text or len(text.encode())>1600:continue
                distance=index-local_index
                if sid not in candidates or distance<candidates[sid][0]:candidates[sid]=(distance,source)
        seen={item['source_id'] for item in items};added=0
        for sid,(distance,source) in sorted(candidates.items(),key=lambda item:(item[1][0],item[0])):
            if sid in seen:continue
            items.append(dict(source_id=sid,role=source.get('declared') or source['role'],quote=source['text'],first_party=True,match_method='visible_prior_dialogue'))
            seen.add(sid);added+=1
            if added==2:break
        result[letter]=items
    return result

def triggered_feedback(feedback,sources,failures):
    affected=targets(feedback['required_options'],sources,failures)
    def clean(value,bad_ids):
        if isinstance(value,dict):return {k:clean(v,bad_ids) for k,v in value.items() if not (k=='quote' and value.get('source_id') in bad_ids)}
        if isinstance(value,list):return [clean(v,bad_ids) for v in value]
        if isinstance(value,str) and value.startswith(('{','[')):
            try:parsed=json.loads(value)
            except (ValueError,TypeError):return value
            result=clean(parsed,bad_ids)
            return json.dumps(result,ensure_ascii=False) if result!=parsed else value
        return value
    result=copy.deepcopy(feedback)
    for letter,bad_ids in affected.items():
        result['invalid_options'][letter]=clean(result['invalid_options'][letter],bad_ids)
    result['previous_invalid_options']=[clean(row,affected[row['letter']]) if row.get('letter') in affected else row for row in result['previous_invalid_options']]
    return result
