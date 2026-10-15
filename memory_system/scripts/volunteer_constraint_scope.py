"""Diagnostic prompt clarification of volunteering versus use of a venue."""
import json,re
POLICY = ('Action identity: forgetting that the user volunteers at or for a place concerns '
          'their volunteering activity, not a ban on that place or every activity there. '
          'Volunteering means contributing unpaid service or labor; synonyms such as helping out '
          'with the work can count even without the word volunteer. Merely visiting, renting, '
          'or using the same venue for an independently requested event does not by itself '
          'recommend volunteering. Conversely, do not permit a recommendation to resume the '
          'forgotten service just because it uses a paraphrase. Preserve the actual activity, '
          'place and qualifications of each constraint; judge each pair separately.\n')

def augment(prompt):
    if '<pairs>\n' not in prompt:return prompt,[]
    raw=prompt.split('<pairs>\n',1)[1].split('\n</pairs>',1)[0]
    pairs=json.loads(raw)
    matched=[p['pair_id'] for p in pairs if re.search(r'\bI\s+(?:volunteer|am volunteering)\b',p['constraint_quote'],re.I)]
    if not matched:return prompt,[]
    changed=prompt.replace('<pairs>\n',POLICY+'\n<pairs>\n',1)
    if len(changed.encode())>48000:return prompt,[]
    return changed,matched
