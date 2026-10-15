"""Explicit existing premises survive advice normalization, without support."""
import sys
import asyncio
import copy
from pathlib import Path
from unittest.mock import patch, AsyncMock
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import choice_premises, config, answer_choice


def main():
    positives = [
        'You might integrate your trip preparations into your existing to-do list habit',
        'You could use your established meditation practice to relax',
        'You should review your existing bank account',
        'You might use your long-standing club membership',
    ]
    for claim in positives:
        option = 'A. ' + claim + '.'
        assert choice_premises.asserted_existing_personal_state(claim, option)
        with patch.object(config, 'CHOICE_EXISTING_STATE_GUARD', True):
            assert not choice_premises.is_suggestion(claim, option)
    for claim in ['You might start a new to-do list habit',
                  'You could adjust your existing routine',
                  'You might review existing accounts in general',
                  'You could use their existing collection']:
        assert not choice_premises.asserted_existing_personal_state(claim, 'A. '+claim)
    claim = positives[0]
    for prefix in ['If useful, ', 'Imagine that ', 'In a fictional scenario, ']:
        assert not choice_premises.asserted_existing_personal_state(claim, prefix+claim)
    assert not choice_premises.asserted_existing_personal_state(claim, claim+'; '+claim)
    with patch.object(config, 'CHOICE_EXISTING_STATE_GUARD', False):
        assert choice_premises.is_suggestion(claim, 'A. '+claim)
    payload = {'options': [{'letter':'A', 'kind':'personal', 'claims':[
        {'text':claim, 'status':'unsupported', 'premise_type':'unknown',
         'reason':'no_source', 'citations':[]}]}]}
    with patch.object(config, 'CHOICE_EXISTING_STATE_GUARD', True):
        entry = answer_choice.validate_assessments(payload, ['A. '+claim], {})[0]
        assert entry['claims'] and entry['claims'][0]['status']=='unsupported'
    # Even a mistaken downstream generic-scope vote cannot erase this premise.
    # Reuse the same input on both arms so only the guard changes the outcome.
    for enabled in (False, True):
        diagnostics = {}
        with patch.object(config, 'CHOICE_EXISTING_STATE_GUARD', enabled), \
             patch.object(answer_choice, '_judge', AsyncMock(return_value={'A:0': False})):
            updated = asyncio.run(answer_choice.reclassify_uncited([copy.deepcopy(entry)], diagnostics))
        assert bool(updated[0]['claims']) == enabled
        if enabled:
            assert updated[0]['claims'][0]['status'] == 'unsupported'
            assert diagnostics['answer_premise_scope']['protected_existing_claims'] == ['A:0']
        else:
            assert updated[0]['kind'] == 'generic'
    print('PASS: existing premise retention, hypothetical/generic controls, no fabricated support')


if __name__ == '__main__':
    main()
