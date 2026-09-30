"""Scope corrections preserve evidence and mandatory independent verification."""
import copy,unittest
from unittest.mock import AsyncMock,patch
from app import answer_choice as ac,choice_premises as helper

class CausalGuardTests(unittest.IsolatedAsyncioTestCase):
    async def test_scope_protection_never_grants_support_or_changes_sources(self):
        text='you have never owned a violin'
        option='A. Since '+text+', consider renting one.'
        claim=dict(text=text,status='unsupported',citations=[],dropped_citations=[],
            validation_errors=[],reason='no_source',premise_type='ownership')
        entry=dict(letter='A',option=option,claims=[claim],kind='personal',status='unsupported',
            validation_status='valid',validation_errors=[],primary_claim=0,warnings=[])
        original=copy.deepcopy(entry);diag={}
        with patch.object(ac,'_judge',AsyncMock(return_value={'A:0':False})):
            actual=await ac.reclassify_uncited([entry],diag)
        self.assertEqual(entry,original)
        self.assertEqual(actual,[original])
        self.assertEqual(diag['answer_premise_scope']['protected_causal_claims'],['A:0'])
        checks=ac.entailment_checks(actual,{},[option])
        payload={'checks':[dict(claim_id=c['claim_id'],entailed=False) for c in checks]}
        verified=ac.validate_entailments(payload,actual,checks)
        self.assertEqual(ac.eligible_choices(verified,set()),[])

    def test_hypotheses_and_advice_bodies_are_not_causal_assertions(self):
        cases=[('If you own a violin, try playing it.','you own a violin'),
               ('Since you might like music, try a concert.','you might like music'),
               ('Because rain is likely, take your umbrella.','your umbrella'),
               ('Given your interest in music, try a concert.','try a concert'),
               ('Since your fictional character owns a violin, write a scene.','your fictional character owns a violin')]
        for option,claim in cases:
            with self.subTest(option=option):
                self.assertFalse(helper.asserted_causal_prefix(dict(option=option,claim=claim)))

    def test_new_advice_boundaries(self):
        for case in [{'claim': 'locally sourced food', 'option': 'A. Incorporate locally sourced food into simple dishes.', 'expected': True, 'actual': True}, {'claim': 'local clubs or hobby groups that appeal to you', 'option': 'D. To feel connected, consider joining local clubs or hobby groups that appeal to you.', 'expected': True, 'actual': True}, {'claim': 'locally sourced food', 'option': 'A. I incorporate locally sourced food into simple dishes.', 'expected': False, 'actual': False}, {'claim': 'food', 'option': 'A. Incorporate food from your garden into dishes.', 'expected': False, 'actual': False}, {'claim': 'food', 'option': 'A. Incorporate food you grew last year into dishes.', 'expected': False, 'actual': False}, {'claim': 'food', 'option': 'A. Incorporate food that you already own into dishes.', 'expected': False, 'actual': False}, {'claim': 'your camera', 'option': 'A. Consider using your camera.', 'expected': False, 'actual': False}, {'claim': 'groups that you visited last year', 'option': 'A. Consider joining groups that you visited last year.', 'expected': False, 'actual': False}, {'claim': 'local clubs that appeal to you', 'option': 'A. Local clubs that appeal to you exist nearby.', 'expected': False, 'actual': False}, {'claim': 'new activities that you enjoy', 'option': 'A. Consider trying new activities that you enjoy.', 'expected': True, 'actual': True}, {'claim': 'the jazz that you enjoy', 'option': 'A. Consider exploring the jazz that you enjoy.', 'expected': False}, {'claim': 'chess that you like', 'option': 'A. Consider exploring chess that you like.', 'expected': False}, {'claim': 'the local clubs that appeal to you', 'option': 'A. Consider exploring the local clubs that appeal to you.', 'expected': False}, {'claim': 'those activities that you enjoy', 'option': 'A. Consider exploring those activities that you enjoy.', 'expected': False}]:
            with self.subTest(case=case):
                self.assertEqual(helper.is_suggestion(case['claim'],case['option']),case['expected'])

if __name__=='__main__':
    unittest.main()
