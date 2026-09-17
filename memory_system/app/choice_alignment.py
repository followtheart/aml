"""Choice judgements grounded in verbatim spans of the verified packet."""
import json
import re
from . import answer_context, llm, personal_evidence, profile, prompts

VERSION = 'cited-option-claims-v2'
_MATCH_ORDER = {'strong': 2, 'weak': 1, 'none': 0}


def option_letter(option, index):
    match = re.match(r'\s*\(?([A-Z])[.)\s]', option)
    return match.group(1) if match else chr(65 + index)


def evidence_pool(memories):
    # Enforce the same immutable packet used by the answer model.
    visible = answer_context.build(memories)
    pool, forgotten = {}, {}
    for index, memory in enumerate(memories):
        eligibility = memory.get('personal_evidence') if memory.get('packet_hash_version') == 2 else None
        if not (eligibility if eligibility is not None else personal_evidence.personal(memory)):
            continue
        if not memory.get('packet_hash') and memory.get('content', '') not in visible:
            continue  # Legacy contexts may have been clipped before answering.
        mid = memory.get('id') or f'memory_{index}'
        if mid in pool or mid in forgotten:
            raise ValueError('Duplicate alignment evidence id')
        item = {'id': mid, 'memory_type': memory.get('memory_type', 'fact'),
                'text': memory.get('content', '')}
        is_forget = profile.is_forget_rule({'type': item['memory_type'],
                                           'content': item['text'].split(answer_context.SOURCE_MARKER, 1)[0]})
        (forgotten if is_forget else pool)[mid] = item
    return pool, forgotten


def rank_key(entry):
    return (bool(entry.get('forbidden')), -_MATCH_ORDER.get(entry.get('match'), 0),
            bool(entry.get('unsupported_claims')))


def rank_alignment(entries):
    """Stable display ordering only; equal keys express no winning option."""
    return sorted(entries, key=rank_key)


def unique_supported_choice(entries):
    eligible = [e for e in entries if not e.get('forbidden') and e.get('match') == 'strong'
                and e.get('citation_valid') and not e.get('unsupported_claims')
                and not e.get('validation_errors')]
    return eligible[0] if len(eligible) == 1 else None


def _citation(mid, span, pool):
    return bool(mid in pool and isinstance(span, str) and span.strip()
                and span in pool[mid]['text'])


def validate_entries(raw, options, pool, forgotten):
    expected = {option_letter(option, i): option for i, option in enumerate(options)}
    if not isinstance(raw, list) or len(raw) != len(expected):
        raise ValueError('Alignment must judge every option exactly once')
    by_letter = {}
    for entry in raw:
        if not isinstance(entry, dict):
            raise ValueError('Invalid alignment entry')
        letter = entry.get('letter')
        if letter not in expected or letter in by_letter:
            raise ValueError('Unknown or duplicate alignment option')
        option = expected[letter]
        errors = []
        claim = entry.get('option_claim', '')
        claim_valid = isinstance(claim, str) and bool(claim.strip()) and claim in option
        if claim and not claim_valid:
            errors.append('claim_not_in_option')
        kind = 'persona' if claim_valid else entry.get('kind', 'generic')
        if kind not in ('persona', 'generic'):
            raise ValueError('Invalid alignment kind')
        match = entry.get('match', 'none')
        if match not in _MATCH_ORDER:
            raise ValueError('Invalid alignment match')
        mid, span = entry.get('evidence_id', ''), entry.get('evidence', '')
        valid = _citation(mid, span, pool)
        if match != 'none' and (not valid or not claim_valid):
            errors.append('unsupported_citation' if not valid else 'missing_option_claim')
            match = 'none'
        unsupported = entry.get('unsupported_claims', [])
        if not isinstance(unsupported, list) or any(not isinstance(s, str) or not s.strip() or s not in option for s in unsupported):
            errors.append('invalid_unsupported_claims')
            unsupported, match = [], 'none'
        if unsupported and match == 'strong':
            match = 'weak'
        constraint_id, constraint_span = entry.get('constraint_id', ''), entry.get('constraint_span', '')
        constraint_valid = _citation(constraint_id, constraint_span, forgotten)
        forbidden = bool(entry.get('forbidden')) and constraint_valid and claim_valid
        if entry.get('forbidden') and not forbidden:
            errors.append('unsupported_constraint')
        by_letter[letter] = {
            'letter': letter, 'kind': kind, 'match': match, 'forbidden': forbidden,
            'option_claim': claim if claim_valid else '',
            'evidence_id': mid if valid else '', 'evidence': span if valid else '',
            'citation_valid': valid, 'unsupported_claims': unsupported,
            'constraint_id': constraint_id if forbidden else '',
            'constraint_span': constraint_span if forbidden else '',
            'validation_errors': errors}
    return [by_letter[letter] for letter in expected]


async def align(qa, memories, diagnostics=None):
    options = qa.get('options') or []
    if not options:
        return None, None
    pool, forgotten = evidence_pool(memories)
    if diagnostics is not None:
        diagnostics.update(alignment_version=VERSION, alignment_input_ids=list(pool),
                           alignment_constraint_ids=list(forgotten))
    # Even an empty pool needs classification of unsupported persona premises.
    prompt = prompts.render('11_choice_align.txt', question=qa['question'],
                            options='\n'.join(options),
                            profile_memories=json.dumps(list(pool.values()), ensure_ascii=False),
                            forgotten_traits=json.dumps(list(forgotten.values()), ensure_ascii=False))
    result = await llm.complete_json(prompt, '{"options": []}',
                                     schema=llm.STRUCTURED_SCHEMAS['choice_align'], stage='eval.choice_align')
    entries = validate_entries(result.get('options'), options, pool, forgotten)
    lines = []
    for e in entries:
        line = f"- {e['letter']}: {e['kind']}, match={e['match']}"
        if e['forbidden']:
            line += f", FORBIDDEN (constraint {e['constraint_id']}: {e['constraint_span']})"
        if e['option_claim']:
            line += f"; option claim: {e['option_claim']}"
        if e['evidence']:
            line += f"; evidence {e['evidence_id']}: {e['evidence']}"
        if e['unsupported_claims']:
            line += '; unsupported premises: ' + '; '.join(e['unsupported_claims'])
        if e['validation_errors']:
            line += '; invalid judgement: ' + ', '.join(e['validation_errors'])
        lines.append(line)
    ranked = rank_alignment(entries)
    groups = []
    for entry in ranked:
        if not groups or rank_key(groups[-1][0]) != rank_key(entry):
            groups.append([])
        groups[-1].append(entry)
    lines.append('Evidence tiers (ties are unordered; not an answer): ' + ' > '.join(
        ' = '.join(e['letter'] for e in group) for group in groups))
    if all(e['match'] == 'none' for e in entries if not e['forbidden']):
        lines.append('No option has supported personal evidence. Prefer advice without invented personal history; do not break ties by letter.')
    return '\n'.join(lines), entries
