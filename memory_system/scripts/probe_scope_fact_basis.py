"""Compare scope labels with an explicit distinguishing-fact field.

Selected diagnostic controls only. Labels never enter provider prompts.
"""
import argparse
import asyncio
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pydantic import Field
from app import answer_choice as ac, budget, config, llm
from probe_verification_scope import scope_prefix


class FactScope(ac.StrictModel):
    claim_id: str
    distinguishing_fact: str = Field(description='The pre-existing user-specific fact asserted; empty when none is asserted.')
    personal_fact: bool


class FactScopes(ac.StrictModel):
    claims: list[FactScope]


def controls():
    examples = [
        ('Try relaxing before bed and focus on your breathing.', 'focus on your breathing', False),
        ('Pair a break with light stretching or a calming beverage.', 'light stretching or a calming beverage', False),
        ('A calming beverage can help you prepare for a more restful night.', 'prepare for a more restful night', False),
        ('Verify your license number TEST-24680 before submitting the form.', 'your license number TEST-24680', True),
        ('Verify your license number [LICENSE_NUMBER] before submitting the form.', 'your license number [LICENSE_NUMBER]', False),
        ('I recommend checking with your local office to ensure your records are current.',
         'I recommend checking with your local office to ensure your records are current', False),
        ('Use your existing meditation practice to relax.', 'your existing meditation practice', True),
        ('If you have an existing meditation practice, use it to relax.', 'an existing meditation practice', False),
        ('Use your favorite tropical beach soundscapes while writing.', 'your favorite tropical beach soundscapes', True),
        ('Try a tropical beach soundscape you might enjoy while writing.', 'a tropical beach soundscape you might enjoy', False),
        ('Since you do not own a bicycle, consider renting one.', 'you do not own a bicycle', True),
        ('The interplay of flavors and visuals will keep the energy high and memorable for all.',
         'interplay of flavors and visuals', False),
        ('The interplay of flavors and visuals will keep the energy high and memorable for all.',
         'keep the energy high and memorable for all', False),
        ('Your colleague owns a bicycle; ask to borrow it.', 'Your colleague owns a bicycle', False),
        ('Choose a new username such as brightleaf for the site.', 'a new username such as brightleaf', False),
        ('Use your existing username brightleaf to sign in.', 'your existing username brightleaf', True),
    ]
    return [dict(claim_id=f'c{i}', option=o, claim=c, expected=e) for i, (o,c,e) in enumerate(examples)]


async def main(args):
    if config.FAKE:
        raise ValueError('Real provider required')
    groups = (json.loads(args.cases.read_text())['groups'] if args.cases
              else [dict(qa_id='synthetic',cases=controls())])
    if len(groups) != 1:
        raise ValueError('Run one archived scope group per invocation to preserve its batch boundary')
    cases = groups[0]['cases']
    args.output.with_suffix('.plan.json').write_text(json.dumps(dict(
        cases=cases, qa_id=groups[0]['qa_id'], repeat=2, arms=['baseline', 'fact_basis'],
        caveat='Selected grammatical controls, not benchmark accuracy; no production changes.'), indent=2)+'\n')
    with args.output.open('x') as output:
        for rep in range(2):
            ordered = cases if rep == 0 else list(reversed(cases))
            for enabled in ([False, True] if rep == 0 else [True, False]):
                schema = FactScopes if enabled else ac.Scopes
                prompt = scope_prefix()
                if enabled:
                    prompt += (' For each claim, first fill distinguishing_fact with the specific pre-existing '
                               'fact asserted about the current user, or an empty string if none is asserted. '
                               'Do not convert an instruction, a requested check, an example value, a symbolic '
                               'placeholder or a desired result into a fact. Then set personal_fact consistently '
                               'with that field. Preserve actual assertions even if unsupported or negated.')
                prompt += '\n' + json.dumps([{k:c[k] for k in ('claim_id','option','claim')} for c in ordered])
                with budget.scope(seconds=100, calls=1, tokens=30000):
                    response = await llm.complete_json(prompt, schema=schema.model_json_schema(),
                        stage='diagnostic.scope_fact_basis', attempts=1, timeout=90, max_tokens=4000)
                parsed = schema.model_validate(response)
                decisions = {c.claim_id:c.personal_fact for c in parsed.claims}
                complete = len(parsed.claims)==len(cases) and set(decisions)=={c['claim_id'] for c in cases}
                row = dict(repetition=rep+1, enabled=enabled, complete=complete, response=response,
                           qa_id=groups[0]['qa_id'],
                           mismatches=([c['claim_id'] for c in cases if decisions.get(c['claim_id']) != c['expected']]
                                       if all('expected' in c for c in cases) else None),
                           prompt=prompt, cases=cases)
                output.write(json.dumps(row, ensure_ascii=False)+'\n'); output.flush()
                print(json.dumps({k:row[k] for k in ('repetition','enabled','complete','mismatches')}), flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cases',type=Path)
    asyncio.run(main(p.parse_args()))
