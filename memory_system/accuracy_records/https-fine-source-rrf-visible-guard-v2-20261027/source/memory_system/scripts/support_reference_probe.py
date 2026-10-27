"""Diagnostic reference transport; normal support validation follows decoding."""
import copy
import json

from app import choice_witness


def request(prompt, schema):
    before, rest = prompt.split('<sources>\n', 1)
    raw, after = rest.split('\n</sources>', 1)
    sources = {c['id']: c for c in json.loads(raw)}
    cards, registry = choice_witness.reference_sources(sources)
    for card in cards:
        card['timestamp'] = sources[card['id']].get('timestamp')
    transformed = copy.deepcopy(schema)
    citation = transformed['$defs']['Citation']
    citation['properties'].pop('quote')
    citation['properties']['span_id'] = dict(type='string', enum=list(registry))
    citation['required'] = ['span_id' if name=='quote' else name for name in citation['required']]
    actual = before + '<sources>\n' + json.dumps(cards,ensure_ascii=False) + '\n</sources>' + after
    actual += ('\nOUTPUT CITATION FORMAT FOR THIS REQUEST: use source_id, span_id, basis, subject. '
               'Select span_id from the original source card parts. Do not output quote or regenerate '
               'source text. The program resolves the reference into exact original text before all '
               'normal support checks. Choose a span belonging to the cited source_id. A valid span '
               'is not proof of support. Use unsupported with citations=[] if no source establishes '
               'the premise, preserving any actual personal premise. The reference format replaces '
               'the quote field in earlier output-format instructions and failed examples.')
    return actual, transformed, registry


def decode(payload, registry):
    result = copy.deepcopy(payload)
    options = result['options']
    for option in options.values() if isinstance(options,dict) else options:
        for claim in option['claims']:
            for citation in claim['citations']:
                if set(citation) != {'source_id','span_id','basis','subject'}:
                    raise ValueError('Unexpected reference citation fields')
                span = registry.get(citation['span_id'])
                if span is None or span['source_id'] != citation['source_id']:
                    raise ValueError('Unknown or cross-source reference')
                citation.pop('span_id')
                citation['quote'] = span['quote']
    return result
