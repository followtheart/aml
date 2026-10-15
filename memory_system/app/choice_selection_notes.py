"""Bounded source-ID notes for comparing already admissible generic options."""
import json

from pydantic import BaseModel, ConfigDict, Field


class Note(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    letter: str
    evidence: list[str] = Field(max_length=2)
    contextual_fit: str
    unsupported_or_prohibited_details: str


class Selection(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    notes: list[Note]
    answer: str


def render(prompt, constraints):
    return prompt + '\nForget constraints:\n' + json.dumps(list(constraints.values()), ensure_ascii=False) + (
        '\nBefore choosing, produce one concise note for EVERY eligible option. '
        'evidence contains at most two source IDs from the supplied source or constraint cards, '
        'or [] if none is relevant. Return IDs only; the caller resolves original text. '
        'contextual_fit explains how the proposal fits this request and supported user preferences. '
        'unsupported_or_prohibited_details identifies missing, contradicted, outdated or forgotten '
        'personal assumptions. Proposed advice is not evidence of a prior habit. '
        'Preserve speaker attribution and updates. A topical question does not establish a diagnosis, '
        'ownership or preference. Do not invent a connection to unrelated personal facts. '
        'Compare the notes and select only from the eligible options. '
        'Return notes and answer in the requested schema.')


def validate(payload, eligible, catalog):
    result = Selection.model_validate(payload)
    if result.answer not in eligible:
        raise ValueError('Notes selection is outside verified eligible set')
    if len(result.notes) != len(eligible) or {n.letter for n in result.notes} != set(eligible):
        raise ValueError('Need exactly one note per eligible option')
    resolved = {}
    for note in result.notes:
        if any(sid not in catalog for sid in note.evidence):
            raise ValueError('Unknown notes evidence source ID')
        resolved[note.letter] = [catalog[sid] for sid in note.evidence]
    return dict(answer=result.answer, notes=[n.model_dump() for n in result.notes],
                resolved_evidence=resolved,
                caveat='Source identity validity is not semantic entailment verification.')
