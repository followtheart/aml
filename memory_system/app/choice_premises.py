"""Conservative option-text normalization; never manufacture a personal premise."""
from difflib import SequenceMatcher
import re


_WORDS = re.compile(r"\w+(?:['’]\w+)?", re.UNICODE)
_PROTECTED = re.compile(r"\b(?:not|no|never|without|only|always|daily|every|used|past|previously|"
                        r"currently|now|if|unless|would|could|may|might|you|your|they|their|he|she|"
                        r"we|our|i|my|friend|spouse|mother|father|had|have|has|own|owned)\b|n['’]t\b", re.I)
_ADVICE = re.compile(r'\b(?:could|should|might|try|consider|start|perhaps|maybe)\b', re.I)
_FACT = re.compile(r"\b(?:since|because|given|already|own|owned|have|had|has|history of|diagnos\w*|"
                   r"not|no|never|cannot|daily|weekly|monthly|every|often|usually|always|used to|"
                   r"previously|last year|your|you['’]re|you are)\b|n['’]t\b", re.I)
# Inside the advice complement, "your" merely addresses the reader ("arrange
# your shelves"); it is not a fact marker unless the claim itself contains it.
_FACT_TAIL = re.compile(r"\b(?:since|because|given|already|own|owned|have|had|has|history of|diagnos\w*|"
                        r"not|no|never|cannot|daily|weekly|monthly|every|often|usually|always|used to|"
                        r"previously|last year|you['’]re|you are)\b|n['’]t\b", re.I)
# "so guests get a sense of your history": a purpose clause describes the
# intended effect of the advice, not an established personal fact.
_PURPOSE = re.compile(r'\bso\s+(?:that\s+)?(?:\w+\s+){1,3}?(?:can\s+|will\s+|would\s+)?'
                      r'(?:get|gets|see|sees|have|has|feel|feels|know|knows|sense|senses)\b', re.I)
_STRONG_CLAIM = re.compile(r"\b(?:own|owned|have|had|has|history of|diagnos\w*|daily|weekly|monthly|every|"
                           r"often|usually|always|used to|previously|last year|you['’]re|you are)\b", re.I)
_BACKGROUND = re.compile(r'\byour (?:day[- ]to[- ]day activities|routine|needs)\b', re.I)
_HISTORY = re.compile(r'\b(?:last|before|after|when|once|formerly|earlier|ago|yesterday)\b', re.I)
_ASSERTION = re.compile(r'\b(?:you|yourself|I|we|they|he|she)\b', re.I)
_IMPERATIVE = re.compile(r'^(?:connect|take|give|listen|acknowledge|remind|arrange|mix|add|choose|keep)\b', re.I)


def option_spans(letter, option):
    """Stable clause IDs select original text; the full option retains its scope."""
    offset = re.match(r'\s*\(?[A-Z][.)]\s*', option)
    start = offset.end() if offset else 0
    rows = [dict(id=f'{letter}:0', text=option[start:], start=start, end=len(option))]
    for match in re.finditer(r'[^,;.!?\n]+(?:[,;.!?]+|$)', option[start:]):
        text = match.group().strip()
        if not text or text == rows[0]['text']:
            continue
        left = start + match.start() + len(match.group()) - len(match.group().lstrip())
        rows.append(dict(id=f'{letter}:{len(rows)}', text=text, start=left, end=left + len(text)))
    return rows


def canonical_span(claim, option):
    """Recover a unique near-verbatim span, rejecting polarity/subject/time edits.

    Fuzzy matching is only for a single typographical edit in a long word. Short
    function words, insertions, omissions and changed tense remain repair errors.
    """
    exact = re.search(re.escape(claim), option, re.I)
    if exact:
        return exact.group(), False
    words = list(_WORDS.finditer(claim))
    tokens = list(_WORDS.finditer(option))
    matches = []
    wanted = [w.group().casefold().replace('’', "'") for w in words]
    for start in range(len(tokens) - len(words) + 1) if words else ():
        window = tokens[start:start + len(words)]
        value = option[window[0].start():window[-1].end()]
        actual = [w.group().casefold().replace('’', "'") for w in window]
        mismatches = [(a, b) for a, b in zip(wanted, actual) if a != b]
        if not mismatches:
            # Formatting only; punctuation that changes clause boundaries is not formatting.
            if re.sub(r"[\s’']", '', claim.casefold()) == re.sub(r"[\s’']", '', value.casefold()):
                matches.append(value)
            continue
        if len(mismatches) != 1 or SequenceMatcher(None, claim.casefold(), value.casefold()).ratio() <= .9:
            continue
        a, b = mismatches[0]
        if (min(len(a), len(b)) < 6 or _PROTECTED.search(a) or _PROTECTED.search(b)
                or re.findall(_PROTECTED, claim.casefold()) != re.findall(_PROTECTED, value.casefold())
                or a.endswith(('ed', 'ing')) != b.endswith(('ed', 'ing'))):
            continue
        operations = [(tag, i2-i1, j2-j1) for tag, i1, i2, j1, j2 in
                      SequenceMatcher(None, a, b).get_opcodes() if tag != 'equal']
        # Replacement edits can be different words (praying/playing). Permit
        # only a missing/doubled interior character, not tense/suffix edits.
        if (len(operations) == 1 and operations[0][0] in ('insert', 'delete')
                and max(operations[0][1:]) == 1 and a[0] == b[0] and a[-2:] == b[-2:]):
            matches.append(value)
    return (matches[0], True) if len(matches) == 1 else (None, False)


def is_suggestion(claim, option):
    match = re.search(re.escape(claim), option, re.I)
    if not match:
        return False
    # Universal background wording carries no distinguishing personal fact.
    if _BACKGROUND.fullmatch(claim.strip()) and _ADVICE.search(option[:match.end()]):
        return True
    # Future modal advice can include "before bed" or "when needed". Keep
    # actual past ability, possession, negation and causal personal qualifiers.
    modal = re.match(r'^(?:you\s+)?(?:could|should|might)\s+', claim, re.I)
    if modal:
        tail = claim[modal.end():]
        past = re.search(r'\b(?:last|formerly|earlier|ago|yesterday|used to)\b|'
                         r'\bbefore\s+(?:the|your|my|his|her)\b', tail, re.I)
        if not past and not _FACT_TAIL.search(tail) and not re.search(r'\b(?:since|because|given)\b',
                option[:match.start()].split(',')[-1], re.I):
            return True
    # Only discard advice complements. A declarative subject or a historical
    # qualifier stays a premise even when embedded inside a recommendation.
    if _HISTORY.search(claim):
        return False
    if _ASSERTION.search(claim) and not re.match(r'^you\s+(?:could|should|might)\s+(?:try|consider|start)\b', claim, re.I):
        return False
    start = max(option.rfind(mark, 0, match.start()) for mark in ('. ', ';', '\n'))
    prefix = option[start + 1:match.start()]
    if (_IMPERATIVE.match(claim) and not _ASSERTION.search(prefix) and not _FACT.search(claim)
            and (not prefix.strip(' ABCDEFGHIJKLMNOPQRSTUVWXYZ.)') or re.search(r'[,;]\s*$', prefix))):
        return True
    # An imperative at the start of an option is also future advice.
    advice = list(_ADVICE.finditer(prefix + claim))
    if not advice:
        return False
    tail = (prefix + claim)[advice[-1].end():]
    if _FACT.search(claim) or _FACT_TAIL.search(tail):
        # A short "your X" purpose complement ("so guests get a sense of your
        # history") is still the advice's intended effect, not prior history.
        purpose = _PURPOSE.search(tail[:len(tail) - len(claim)])
        if not (purpose and re.fullmatch(r'(?:your|their)\s+\w+', claim.strip(), re.I)
                and not _STRONG_CLAIM.search(claim) and not _FACT_TAIL.search(tail)):
            return False
    # A preceding premise is allowed, but an embedded "because/since" is not.
    if re.search(r'\b(?:since|because|given)\b', prefix, re.I) and ',' not in prefix:
        return False
    return True
