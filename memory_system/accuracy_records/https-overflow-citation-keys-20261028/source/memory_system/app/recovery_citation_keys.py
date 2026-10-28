"""Candidate-owned citation keys, with the existing validator authoritative.

No reference labels, answer choices selected as gold, or provider calls here.
Target text is ordinary search input; the model still decides relevance.
"""
import copy
import hashlib
import json
import time

from . import listwise_recovery, recovery_spans


OUTPUT_GUIDANCE = '''For each candidate, return target_id and citation_keys only.
If useful, choose the target ID and one or two keys from that candidate's enum.
A key is candidate:evidence_kind@source.part. It binds its candidate, kind and part. Two keys MUST share a kind.
An allowed key does not establish relevance. Select meaningful statements with subjects and qualifiers.
Use user_context for actual first-person circumstances including negative/past circumstances; topic_interest
for user questions; counterevidence for user denials/qualifications; third_party_context for others/fictional
context; assistant_context for relevant assistant material. This proves no personal fact.
Never change kinds just to force retention. If not useful, return target_id="" and citation_keys=[].
Do not generate quotes or new keys. Return every candidate key exactly once.
Call emit_json_result with the candidate-keyed object only.
'''


def build(original):
    result = copy.deepcopy(original)
    cards = {c['candidate_id']: c for c in original['cards']}
    lookup = {}
    properties = {}
    for candidate, mid in original['candidate_map'].items():
        choices = {}
        for kind in recovery_spans.KINDS:
            if kind == 'none':
                continue
            for part, source_id in original['owners'][candidate].items():
                quote = original['spans'][part]['quote'].strip()
                if not quote:
                    continue
                ref = listwise_recovery.Citation(source_id=source_id, quote=quote)
                if listwise_recovery._check_citation(ref, kind, cards[mid]['sources']) is None:
                    choices[candidate + ':' + kind + '@' + part] = dict(kind=kind, part=part)
        lookup[candidate] = choices
        items = dict(type='string', enum=list(choices)) if choices else dict(type='string')
        properties[candidate] = dict(type='object', additionalProperties=False,
            required=['target_id', 'citation_keys'], properties={
                'target_id': dict(type='string', enum=[''] + list(original['targets'])),
                'citation_keys': dict(type='array', maxItems=2 if choices else 0, items=items)})
    prefix, output = original['prompt'].split('For each candidate, return useful,', 1)
    _, payload = output.split('Call emit_json_result with the candidate-keyed object only.\n', 1)
    # Same semantic instructions before output syntax, same complete source payload.
    result.update(prompt=prefix + OUTPUT_GUIDANCE + payload,
                  schema=dict(type='object', additionalProperties=False,
                              required=list(properties), properties=properties),
                  citation_key_map=lookup)
    assert json.loads(payload) == json.loads('{' + original['prompt'].split('\n{', 1)[1])
    return result


def translate(value, key, built):
    if not isinstance(value, dict) or set(value) != {'target_id', 'citation_keys'}:
        raise ValueError('Incorrect citation-key decision structure')
    target = value['target_id']
    keys = value['citation_keys']
    if not isinstance(target, str) or target not in ['', *built['targets']]:
        raise ValueError('Unknown target')
    if not isinstance(keys, list) or len(keys) > 2:
        raise ValueError('Invalid citation-key count')
    if bool(target) != bool(keys):
        raise ValueError('Inconsistent negative or positive decision')
    if any(not isinstance(k, str) or k not in built['citation_key_map'][key] for k in keys):
        raise ValueError('Unknown or unowned citation key')
    selected = [built['citation_key_map'][key][k] for k in keys]
    kinds = {x['kind'] for x in selected}
    if len(kinds) > 1:
        raise ValueError('Mixed evidence kinds')
    return dict(useful=bool(keys), target_id=target,
                evidence_kind=selected[0]['kind'] if selected else 'none',
                citation_parts=[x['part'] for x in selected])


def decode_partial(payload, built):
    if not isinstance(payload, dict) or set(payload) - set(built['candidate_map']):
        raise ValueError('Unknown candidate key')
    restored, judgments = set(), []
    for key, mid in built['candidate_map'].items():
        sub = dict(built, candidate_map={key: mid},
                   cards=[c for c in built['cards'] if c['candidate_id'] == mid])
        try:
            if key not in payload:
                raise ValueError('Missing candidate key')
            translated = translate(payload[key], key, built)
            ids, rows = recovery_spans.decode({key: translated}, sub)
            restored.update(ids)
            judgments.extend(rows)
        except (ValueError, TypeError, KeyError) as exc:
            judgments.append(dict(candidate_id=mid, useful=False, target_id='', target_quote='',
                evidence_kind='none', citations=[], valid=False,
                validation_errors=['citation_key_error'], error_detail=str(exc),
                proposed_decision=payload.get(key)))
    return restored, judgments


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':')).encode()).hexdigest()


def prepare(req, cards, request, limits, deadline, prompt_limit, *, abbreviations=False,
            role_bound=False, transform_context=False):
    """Keep original admission; fallback without changing any admitted candidate."""
    original = recovery_spans.build(req, cards, role_bound=role_bound,
        transform_context=transform_context, abbreviations=abbreviations)
    assert (original['prompt'], original['schema'], original['system']) == (
        request['prompt'], request['schema'], request['system'])
    proposed = build(original)
    bound = request['reservation'] + len(proposed['prompt'].encode()) - len(request['prompt'].encode())
    bound += len(json.dumps(proposed['schema']).encode()) - len(json.dumps(request['schema']).encode())
    available = None if limits is None else limits.max_tokens-limits.tokens-limits.reserved_tokens
    calls = None if limits is None else limits.max_calls-limits.calls-limits.reserved_calls
    now = time.monotonic()
    seconds = min(deadline-now if deadline is not None else float('inf'),
        limits.deadline-now-2 if limits is not None else float('inf'))
    reason = ('prompt_budget' if len(proposed['prompt'].encode()) > prompt_limit else
              'token_budget' if available is not None and bound > available else
              'call_budget' if calls is not None and calls < 1 else
              'deadline' if seconds <= 0 else None)
    selected = dict(request)
    if reason is None:
        selected.update(prompt=proposed['prompt'], schema=proposed['schema'], reservation=bound)
    marker = dict(policy='owned_citation_keys_v1', status='fallback' if reason else 'applied', reason=reason,
        candidate_ids=[c['candidate_id'] for c in cards], source_sha256=digest(cards),
        original_schema_sha256=digest(request['schema']), proposed_schema_sha256=digest(proposed['schema']),
        dispatched_schema_sha256=digest(selected['schema']),
        original_prompt_sha256=hashlib.sha256(request['prompt'].encode()).hexdigest(),
        proposed_prompt_sha256=hashlib.sha256(proposed['prompt'].encode()).hexdigest(),
        dispatched_prompt_sha256=hashlib.sha256(selected['prompt'].encode()).hexdigest(),
        citation_key_map_sha256=digest(proposed['citation_key_map']),
        original_reservation=request['reservation'], proposed_reservation=bound,
        dispatched_reservation=selected['reservation'], prompt_limit_bytes=prompt_limit,
        token_headroom=available, call_headroom=calls,
        deadline_seconds=seconds if seconds != float('inf') else None)
    return selected, marker, proposed if reason is None else None
