"""Separate claim subject and modality without judging evidence support."""
from typing import Literal
from pydantic import BaseModel, ConfigDict


class RoleScope(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    claim_id: str
    subject: Literal['user_or_existing_user_relationship', 'generic_audience_or_other', 'mixed']
    modality: Literal['existing_assertion', 'proposed_or_conditional', 'mixed']

    @property
    def personal_fact(self):
        return self.subject != 'generic_audience_or_other' and self.modality != 'proposed_or_conditional'


class RoleScopes(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    claims: list[RoleScope]


def prefix(text, *, presupposition_guidance=False):
    text = text.replace('Return every claim_id once with personal_fact=true only if',
                        'A claim contains an existing personal assertion only if').replace(
        'Return false for', 'The following do not assert existing personal facts:').replace(
        'return true.', 'it contains an existing personal assertion.').replace(
        'True does not mean the claim is supported.', 'An assertion need not be supported.')
    text += (' Return only claim_id, subject and modality for each claim. '
                  'The subject category user_or_existing_user_relationship covers '
                  'a specific asserted fact about the user or an existing relationship '
                  'such as their regular customers. Generic intended visitors are '
                  'generic_audience_or_other. Modality existing_assertion means a '
                  'pre-existing fact is asserted; proposed_or_conditional means only '
                  'advice, hoped-for effects or hypothetical conditions. Use mixed '
                  'when an existing assertion is embedded in advice, or both types '
                  'occur. Do not judge evidence or answer quality.')
    if presupposition_guidance:
        text += (' Interpret the claim within the complete option, not in isolation. '
                 'Separate what the condition supposes from what the wording already '
                 'takes for granted. A condition such as "if you own a camera" does '
                 'not assert ownership. But "if you bring the camera you already own" '
                 'conditions bringing it while still asserting ownership; classify '
                 'that claim as mixed. Possessive wording is not by itself proof: '
                 '"if you develop an interest" and "your possible future interest" '
                 'do not assert an existing interest. Keep any existing personal '
                 'assertion embedded in the target claim, without importing facts '
                 'from unrelated clauses. Scope classification never establishes '
                 'that the assertion is true or supported by memory.')
    return text
