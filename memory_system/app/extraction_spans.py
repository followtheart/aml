"""Optional exact references to original messages in the current extract segment."""
import copy
import json
import re

from . import llm, prompts


def source_parts(batch):
    registry, cards = {}, []
    for index, message in enumerate(batch):
        text, start, pieces = message.content, 0, []
        boundaries = list(re.finditer(r'(?<=[.!?。！？])\s+|\n+', text)) + [None]
        for match in boundaries:
            end = match.end() if match else len(text)
            if end > start:
                part_id = f'new{index}.p{len(pieces)}'
                part = text[start:end]
                pieces.append(dict(id=part_id, text=part))
                if part.strip():
                    registry[part_id] = dict(message_index=index, quote=part.strip())
            start = end
        assert ''.join(p['text'] for p in pieces) == text
        cards.append(dict(id=f'new{index}', role=message.role, parts=pieces))
    return dict(registry=registry, cards=cards)


def request_for(batch, values):
    built = source_parts(batch)
    # Transform the trusted template before interpolating any source text.
    # A quoted template heading in a message must never delimit the request.
    template = prompts.load('01_extract_amu.txt')
    start = template.index('Each NEW message is prefixed with its zero-based index in square brackets.')
    end = template.index('\n\nEach fact must contain its OWN triples array', start)
    template = template[:start] + '''Each NEW message has a newN ID and is preserved in full as consecutive original parts. Read ALL parts of each message together, including subject, negation, conditions and time. Only these NEW parts may support a fact; the summary and recent messages are context, NOT citable evidence and never new observations. Each fact must contain evidence: a nonempty array of objects with ONLY part_id, selected from the provided NEW part IDs. Include every part needed to resolve speaker, pronouns, time and qualifications. The message_index and exact original quote will be recovered by code. Do not copy or paraphrase quotes, invent IDs, or use a nearby part in place of the actual evidence. A question alone is NOT evidence of its answer. A source ID does not prove a fact; unchanged semantic verification follows.
''' + template[end:]
    template = template.replace(
        'at least one evidence.quote for that same\nfact MUST include the exact time phrase and the event it describes.',
        'at least one selected evidence part for that same\nfact MUST include the exact time phrase and the event it describes.')
    template = template.replace('Recent messages (local context):',
                                'Recent messages (context only; not NEW; no citable part IDs):')
    values = dict(values, chunk_messages=json.dumps(built['cards'], ensure_ascii=False))
    schema = copy.deepcopy(llm.STRUCTURED_SCHEMAS['extraction'])
    schema['properties']['facts']['items']['properties']['evidence']['items'] = dict(
        type='object', additionalProperties=False, required=['part_id'],
        properties=dict(part_id=dict(type='string', enum=list(built['registry']))))
    return dict(built, prompt=template.format(**values), schema=schema)


def decode(payload, built):
    if not isinstance(payload, dict) or not isinstance(payload.get('facts'), list):
        raise ValueError('Invalid extraction object')
    result, errors = copy.deepcopy(payload), []
    for index, fact in enumerate(result['facts']):
        if not isinstance(fact, dict):
            errors.append(dict(fact_index=index, reason='non_object_fact'))
            continue
        refs, compiled = fact.get('evidence'), []
        try:
            if not isinstance(refs, list) or not refs:
                raise ValueError('Missing part references')
            for ref in refs:
                if (not isinstance(ref, dict) or set(ref) != {'part_id'}
                        or not isinstance(ref['part_id'], str)
                        or ref['part_id'] not in built['registry']):
                    raise ValueError('Unknown or non-current part')
                compiled.append(copy.deepcopy(built['registry'][ref['part_id']]))
            fact['evidence'] = compiled
        except ValueError as exc:
            # Fail this fact's normal grounding, preserving siblings and privacy.
            fact['evidence'] = [dict(message_index=-1, quote='')]
            errors.append(dict(fact_index=index, reason=str(exc), proposed_evidence=refs))
    return result, errors
