"""Deterministic persona ingestion.

Benchmark and product conversations often open with a system-authored persona
card ("[system] You are an AI assistant helping a user with the following
persona: { ...JSON... }"). The LLM extractor treats that blob as one opaque
episode, so none of its attributes (city, job, family, hobbies, devices) become
retrievable profile memories. This module flattens the JSON into atomic
`profile` / `preference` facts with verbatim evidence quotes, marked with
`source_role=persona` so the answer layer can accept them as user-scope facts.

Failures are logged and never fail the Add.
"""
import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Optional, Tuple

from . import config, integrity

log = logging.getLogger("aml.persona")

PERSONA_HEADER = re.compile(r'^\s*\[system\][^{]{0,400}\bpersona\b', re.I | re.S)
_SENSITIVE_PATH = re.compile(r'health|medical|condition|diagnos|sexual|orientation|religio|disabilit|mental|race|ethnic', re.I)
_SKIP_PATH = re.compile(r'speaking_style|tone|conflict_style|short_persona|persona$|traits|tech_comfort|type$|region$|income', re.I)
_MAX_FACTS = 40


def is_persona_message(text: str) -> bool:
    return bool(text) and bool(PERSONA_HEADER.match(text)) and '{' in text


def parse(text: str) -> Optional[Dict]:
    """Decode the first JSON object embedded in the persona message."""
    start = text.find('{')
    if start < 0:
        return None
    try:
        value, _ = json.JSONDecoder().raw_decode(text[start:])
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def _leaves(value: Any, path: Tuple[str, ...] = ()) -> Iterator[Tuple[Tuple[str, ...], Any]]:
    # A named person block (spouse, child, pet) stays one unit so that its age
    # and occupation are never attached to the wrong sibling.
    if isinstance(value, dict) and _person(path) and isinstance(value.get('name'), str):
        yield path + ('_person',), value
        return
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _leaves(item, path + (str(key),))
    elif isinstance(value, list):
        for item in value:
            yield from _leaves(item, path)
    elif value is not None and value != '':
        yield path, value


def _humanize(key: str) -> str:
    return key.replace('_', ' ').strip()


def _person(path: Tuple[str, ...]) -> Optional[str]:
    """Owner of a nested person block (spouse / children / partner)."""
    for relation in ('spouse', 'partner', 'children', 'child', 'parents', 'siblings', 'pets'):
        if relation in path:
            return relation
    return None


def _sentence(path: Tuple[str, ...], value: Any) -> Optional[Tuple[str, str, List[Dict]]]:
    """Render (type, content, triples) for one flattened persona attribute."""
    key = path[-1]
    joined = '.'.join(path)
    relation = _person(path)
    if key == '_person' and relation:
        who = {'spouse': 'spouse', 'partner': 'partner', 'children': 'child', 'child': 'child',
               'parents': 'parent', 'siblings': 'sibling', 'pets': 'pet'}[relation]
        name = str(value['name']).strip()
        extras = [f"{_humanize(k)} {v}" for k, v in value.items()
                  if k != 'name' and v not in (None, '') and not isinstance(v, (dict, list))]
        detail = f" ({', '.join(extras)})" if extras else ''
        return ('profile', f"The user's {who} is {name}{detail}.",
                [dict(subject='user', relation=f'has_{who}', object=name)])
    text = str(value).strip()
    if not text or _SKIP_PATH.search(joined):
        return None
    if relation:
        who = relation.rstrip('s') if relation != 'children' else 'child'
        return ('profile', f"The user's {who} {_humanize(key)} is {text}.", [])
    if key == 'name' and len(path) == 1:
        return ('profile', f"The user's name is {text}.", [dict(subject='user', relation='named', object=text)])
    if key == 'age':
        return ('profile', f"The user is {text} years old.", [])
    if key == 'gender':
        return ('profile', f"The user's gender is {text}.", [])
    if key == 'political_affiliation':
        return ('profile', f"The user identifies politically as {text}.", [])
    if key == 'race_ethnicity':
        return ('profile', f"The user's ethnic background is {text}.", [])
    if path[0] == 'location':
        label = {'city': 'lives in the city of', 'state': 'lives in the state of', 'country': 'lives in'}.get(key, f'lives in ({_humanize(key)})')
        return ('profile', f"The user {label} {text}.", [dict(subject='user', relation='lives_in', object=text)])
    if path[0] == 'occupation':
        if key == 'title':
            return ('profile', f"The user works as a {text}.", [dict(subject='user', relation='works_as', object=text)])
        if key == 'employer':
            return ('profile', f"The user works for {text}.", [dict(subject='user', relation='works_for', object=text)])
        if key == 'years_in_role':
            return ('profile', f"The user has been in their current role for {text} years.", [])
        return ('profile', f"The user's occupation {_humanize(key)} is {text}.", [])
    if path[0] == 'education':
        if key == 'degree':
            return ('profile', f"The user holds a {text}.", [dict(subject='user', relation='has_degree', object=text)])
        if key == 'institution':
            return ('profile', f"The user studied at {text}.", [dict(subject='user', relation='studied_at', object=text)])
        return ('profile', f"The user's education {_humanize(key)} is {text}.", [])
    if key == 'marital_status':
        return ('profile', f"The user is {text.lower()}.", [dict(subject='user', relation='marital_status', object=text)])
    if key == 'religious_background':
        return ('profile', f"The user's religious background: {text}", [])
    if re.search(r'hobb|interest', joined, re.I):
        return ('preference', f"The user enjoys {text[:1].lower() + text[1:]}.",
                [dict(subject='user', relation='interested_in', object=text)])
    if key == 'devices':
        return ('profile', f"The user uses a {text.lower()}.", [dict(subject='user', relation='uses_device', object=text)])
    if key == 'social_media':
        return ('preference', f"The user uses {text}.", [dict(subject='user', relation='uses_platform', object=text)])
    if path[0] in ('values_beliefs', 'values', 'beliefs'):
        return ('preference', f"The user's view on {_humanize(key)}: {text}", [])
    if path[0] in ('personality',):
        return ('profile', f"The user's personality: {text}", [])
    return ('profile', f"The user's {' '.join(_humanize(p) for p in path)} is {text}.", [])


def _quote(text: str, path: Tuple[str, ...], value: Any) -> Optional[str]:
    """Verbatim source span for the attribute: the JSON line that carries it."""
    if isinstance(value, dict):
        return _quote(text, path[:-1] + ('name',), value.get('name'))
    encoded = json.dumps(value, ensure_ascii=False)
    key = json.dumps(path[-1], ensure_ascii=False)
    for pattern in (re.escape(key) + r'\s*:\s*' + re.escape(encoded), re.escape(encoded)):
        match = re.search(pattern, text)
        if match:
            return match.group(0)
    plain = str(value)
    return plain if plain in text else None


def _entities(value: Any) -> List[str]:
    if isinstance(value, dict):
        return [str(value.get('name'))[:80]]
    return [str(value)[:80]] if isinstance(value, str) and len(value) <= 80 else []


def persona_facts(req, facts: List[Dict], segments: List[List[int]]) -> List[Dict]:
    """Append deterministic profile facts for each persona message in the request."""
    if not config.PERSONA_SOURCE_EXTRACTION:
        return facts
    segment_of = {i: s for s, indices in enumerate(segments or []) for i in indices}
    existing = {' '.join(str(f.get('content', '')).lower().split()) for f in facts}
    for index, message in enumerate(req.messages):
        if message.role != 'user' or not is_persona_message(message.content):
            continue
        data = parse(message.content)
        if not data:
            log.info("persona message without parseable JSON source_index=%d", index)
            continue
        reference = (datetime.fromtimestamp(message.timestamp / 1000, tz=timezone.utc).isoformat()
                     if message.timestamp else None)
        temporal = integrity.resolve_time(None, reference)
        produced = 0
        for path, value in _leaves(data):
            if produced >= _MAX_FACTS:
                break
            rendered = _sentence(path, value)
            if not rendered:
                continue
            kind, content, triples = rendered
            if ' '.join(content.lower().split()) in existing:
                continue
            quote = _quote(message.content, path, value)
            if not quote:
                continue
            sensitivity = 'sensitive' if _SENSITIVE_PATH.search('.'.join(path)) else 'normal'
            facts.append({
                'content': content, 'retrieval_key': content[:200], 'type': kind,
                'entities': _entities(value),
                'keywords': ['persona', _humanize(path[-1])], 'time_expression': None,
                'state': None, 'triples': triples if sensitivity == 'normal' else [],
                'sensitivity': sensitivity, 'epistemic_status': 'asserted',
                'temporal': temporal, 'event_time': None,
                'evidence': [{'message_index': index, 'quote': quote,
                              'request_id': req.request_id, 'source_role': 'persona'}],
                '_sources': [index], '_segment': segment_of.get(index, 0),
                '_deterministic': True, '_persona': True,
            })
            existing.add(' '.join(content.lower().split()))
            produced += 1
        log.info("persona attributes extracted source_index=%d count=%d", index, produced)
    return facts
