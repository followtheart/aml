"""Explicit option keys for witness retrieval; original quote validation applies."""
import copy
import json
from . import choice_witness


def keyed_request(request,options):
    letters=list(choice_witness._options(options))
    result=copy.deepcopy(request)
    schema=result['schema'];defs=schema['$defs']
    entry=copy.deepcopy(defs['OptionWitnesses'])
    entry['properties'].pop('letter');entry['required'].remove('letter');entry.pop('title',None)
    schema=dict(type='object',additionalProperties=False,
        properties={letter:copy.deepcopy(entry) for letter in letters},required=letters,
        **{'$defs':{'Witness':copy.deepcopy(defs['Witness'])}})
    before,separator,shape=result['prompt'].rpartition('Call emit_json_result with this shape:\n')
    if not separator or not shape.strip().startswith('{"options":'):
        raise ValueError('Unknown historical output-shape instruction')
    example={letter:dict(witnesses=[]) for letter in letters}
    example[letters[0]]['witnesses']=[dict(source_id='s0',quote='exact original text')]
    result['prompt']=(before+'Call emit_json_result with exactly these option-letter keys: '+', '.join(letters)+'.\n'
        'Each value contains only witnesses. Include every requested key; no additional keys, '
        'letter fields, or options array. Use an empty witnesses list for an option without useful sources.\n'
        'Shape example (choose actual source IDs and exact quotations):\n'+json.dumps(example,ensure_ascii=False)+'\n')
    result['schema']=schema
    # Everything before the output contract, including full catalogs, is identical.
    assert result['prompt'].startswith(before)
    assert result['system']==request['system'] and result['max_tokens']==request['max_tokens']
    return result


def decode_keyed(payload,options,sources,trace):
    letters=list(choice_witness._options(options))
    if type(payload) is not dict or set(payload)!=set(letters):
        raise ValueError('Return every requested option key exactly once, with no extra keys')
    compiled=[]
    for letter in letters:
        value=payload[letter]
        if type(value) is not dict or set(value)!={'witnesses'}:
            raise ValueError('Each option value must contain only witnesses')
        compiled.append(dict(letter=letter,witnesses=value['witnesses']))
    return choice_witness.validate(dict(options=compiled),options,sources,trace)
