"""Bounded evidence queries; option text is a hypothesis, never a user fact."""
import re

from . import choice_premises, personal_evidence, rerank_query

MAX_QUERY_CHARS = 240


_EMBEDDED_ROLE=re.compile(r'\byour\s+(?:[\w-]+\s+){1,5}as\s+an?\s+[\w-]+',re.I)
_ROLE_SCOPE=re.compile(r'\b(?:if|unless|not|never|no|without|would|could|may|might|imagine|suppose|pretend|hypothetical|fictional)\b|n[’\x27]t\b',re.I)
def _literal_role(option):
 if rerank_query._premise(option):return None
 # A trailing condition may qualify an earlier role. Preserve the whole
 # option when bounded; otherwise leave the planner unchanged.
 if _ROLE_SCOPE.search(option):
  whole=choice_premises.option_spans('X',option)[0]
  return whole if _EMBEDDED_ROLE.search(whole['text']) and len(whole['text'])<=240 else None
 for span in sorted(choice_premises.option_spans('X',option), key=lambda s: len(s['text'])):
  text=span['text'];match=_EMBEDDED_ROLE.search(text)
  if not match or len(text)>240:continue
  # Do not detach a role from any preceding conditional/negative frame.
  if _ROLE_SCOPE.search(option[:span['start']]):continue
  start=span['start']+match.start()
  if not _ROLE_SCOPE.search(option[:start]):
   tail=option[start:span['end']]
   boundary=re.search(r'[—,;.!?\n]',tail)
   end=start+(boundary.start() if boundary else len(tail))
   while end>start and option[end-1].isspace():end-=1
   return dict(span,id=span['id']+':role',start=start,end=end,text=option[start:end])
  return span
 return None


def _preserve_embedded_role(option, proposed):
    """Keep omitted role wording as a query hypothesis, never as evidence."""
    span = _literal_role(option)
    if span is None:
        return proposed
    match = _EMBEDDED_ROLE.search(span['text'])
    words = personal_evidence.terms(span['text'][match.start():])
    existing = personal_evidence.terms(proposed) if isinstance(proposed, str) else set()
    if words and len(existing & words) < len(words) / 2:
        return span['text']
    return proposed


def compact_option(text):
    """Conservative fallback when a planner omits or malforms option queries."""
    text = re.sub(r'^\s*[A-Z0-9]+[.)、]\s*', '', text).strip()
    # Keep the premise and its negation/conditional wording, not the advice tail.
    premise = re.match(r'^(?:since|given|because|as|if)\b[^\n]*?[,;—](?=\s*(?:you|consider|try|I\b|we\b))',
                       text, re.I)
    if premise:
        text = premise.group().rstrip(',;— ')
    else:
        text = re.split(r'(?<=[.!?。！？])\s+', text, maxsplit=1)[0]
    if len(text) > MAX_QUERY_CHARS:
        cut = text[:MAX_QUERY_CHARS]
        text = cut.rsplit(' ', 1)[0] if ' ' in cut else cut
    return text


def build(query, options, plan, limit=6):
    """Reserve one query per option before expansions; validate positional output.

    Empty planner entries identify generic advice unless the literal option
    has an explicit premise. Missing/invalid entries fall back to the option's
    own text, so parser failure cannot
    silently erase an option. Choice expansions must use supplied vocabulary;
    an unrelated profile cannot introduce ownership or habits into a query.
    """
    specs, seen = [], {}
    requirements = []

    def add(text, origin, option_index=None, target=None, required=False):
        if not isinstance(text, str):
            return
        text = ' '.join(text.split())
        key = text.casefold()
        if not text:
            return
        if target and target.startswith('sub:') and key in seen:
            return
        if target:
            requirements.append(dict(id=target, text=text, origin=origin))
        if key in seen:
            if target and target not in seen[key]['coverage_ids']:
                seen[key]['coverage_ids'].append(target)
            return
        if not required and len(specs) >= limit:
            return
        entry = dict(text=text, origin=origin, option_index=option_index,
                     coverage_ids=[target] if target else [])
        seen[key] = entry
        specs.append(entry)

    add(query, 'question', target='question', required=True)
    proposed = plan.get('option_queries')
    valid_array = isinstance(proposed, list) and len(proposed) == len(options)
    for index, option in enumerate(options):
        value = proposed[index] if valid_array else None
        if valid_array:
            value = _preserve_embedded_role(option, value)
        allowed = personal_evidence.terms(option + ' ' + query)
        words = personal_evidence.terms(value) if isinstance(value, str) else set()
        # Vocabulary overlap with the whole option can validate advice-only
        # rewrites. Keep a safe literal premise when that rewrite loses its
        # distinguishing terms; it remains a retrieval hypothesis, not a fact.
        literal = rerank_query._premise(option)
        premise_words = personal_evidence.terms(literal)
        if premise_words and len(words & premise_words) < len(premise_words) / 2:
            value, words = literal, premise_words
        if (isinstance(value, str) and len(value) <= MAX_QUERY_CHARS
                and (not value.strip() or (words and len(words & allowed) >= len(words) / 2))):
            add(value, 'option_premise', index, f'option:{index}', True)
        else:
            add(compact_option(option), 'option_fallback', index, f'option:{index}', True)
    allowed = personal_evidence.terms(query + ' ' + ' '.join(options))
    for field in ('sub_queries', 'expanded_queries'):
        values = plan.get(field)
        for index, value in enumerate(values if isinstance(values, list) else []):
            if not isinstance(value, str) or not value.strip():
                continue
            value = compact_option(value)
            if options and not personal_evidence.terms(value) <= allowed:
                continue
            # Decomposition is a coverage obligation, not an optional expansion.
            # A planner is asked for at most three steps; report excess explicitly.
            required = field == 'sub_queries' and index < 3
            add(value, field, target=f'sub:{index}' if required else None, required=required)
    plan['_coverage_requirements'] = requirements
    return specs
