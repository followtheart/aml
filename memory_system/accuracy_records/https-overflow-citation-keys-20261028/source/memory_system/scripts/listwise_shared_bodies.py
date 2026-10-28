"""Lossless prototype for repeated candidate-body strings, not source evidence.

Operates only on the existing shared-source display. Original candidate IDs,
indices, source references and source texts are unchanged. Every transformed
body must decode character-for-character; fallback cards remain untouched.
"""
from collections import Counter
import copy
import json
import re


NOTE = ('Candidate body_parts concatenate literal strings and integer references '
        'to shared_candidate_body_texts, in order, to reconstruct the complete body. '
        'These repeated body strings are not source evidence or independent confirmation.\n')


def encoded(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))


def pieces(text):
    # Keep support labels and all their whitespace literally. No provenance,
    # dependency or statement equivalence is inferred from matching text.
    return re.split(r'(\s*\[support: [^\]\n]+\]\s*)', text)


def rewrite(rendered):
    intro, raw = rendered.split('\n', 1)
    original = json.loads(raw)
    if list(original) != ['candidates', 'shared_sources']:
        raise ValueError('Unexpected shared-source payload')
    result = copy.deepcopy(original)
    splits = {i: pieces(card['body']) for i, card in enumerate(original['candidates']) if 'body' in card}
    counts = Counter(piece for parts in splits.values() for i, piece in enumerate(parts)
                     if i % 2 == 0 and piece)
    # Count only complete primary/support body strings, not arbitrary words or
    # source quotations. Conservative local cost gate; whole-layout gate below
    # includes the actual instruction/table/key overhead.
    useful = {text for text, count in counts.items() if count > 1
              and (count - 1) * len(encoded(text).encode()) > count * 12 + 16}
    shared, lookup = {}, {}
    for index, parts in splits.items():
        card = result['candidates'][index]
        if not any(i % 2 == 0 and part in useful for i, part in enumerate(parts)):
            continue
        joined = []
        for i, part in enumerate(parts):
            if i % 2 == 0 and part in useful:
                if part not in lookup:
                    lookup[part] = len(lookup)
                    shared[str(lookup[part])] = part
                joined.append(lookup[part])
            elif part:
                joined.append(part)
        card.pop('body')
        card['body_parts'] = joined
    if not shared:
        return rendered
    result['shared_candidate_body_texts'] = shared
    restored = copy.deepcopy(result)
    restored.pop('shared_candidate_body_texts')
    for card in restored['candidates']:
        if 'body_parts' in card:
            card['body'] = ''.join(shared[str(x)] if type(x) is int else x for x in card.pop('body_parts'))
    if restored != original:
        raise ValueError('Repeated-body display changed original content')
    proposed = intro + '\n' + NOTE + encoded(result)
    return proposed if len(proposed.encode()) < len(rendered.encode()) else rendered


def renderer(original):
    def render(rows):
        return rewrite(original(rows))
    return render
