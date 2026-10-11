"""Late request source with explicit factual-premise versus hypothetical distinction."""
from late_request_source import augment as base_augment
POLICY = ('Current-request evidence: a source tagged current_request is the user’s present utterance. '
          'A question can contain an explicit first-person factual premise about the user’s own current state or experience; '
          'that stated premise may support a matching claim, with an exact quotation. '
          'The restriction on questions establishing past events applies to merely asking about an event, not explicitly stating one’s own experience in a question. '
          'Do not infer unstated details, diagnoses, ownership, frequency, preferences, or past events. '
          'Hypothetical, conditional, third-person, quoted, and role-play scenarios are not autobiographical assertions. '
          'Keep subject, negation, time, uncertainty and scope; asking whether a condition exists does not establish it. '
          'Do not use the answer option itself as evidence. Independent verification still applies.\n')

def augment(prompt, question):
    changed, card = base_augment(prompt, question)
    if card is None: return changed, card
    changed = POLICY + changed
    if len(changed.encode()) > 48000: return prompt, None
    return changed, card
