"""Runtime role scope keeps repair, budget and independent verification guards."""
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import answer_choice as gate, budget, config, llm, scope_roles
from probe_scope_roles import RoleScopes as ProbeRoleScopes
from probe_verification_scope import scope_prefix


def decision(cid, personal=False):
    return dict(claim_id=cid, subject='user_or_existing_user_relationship' if personal else 'generic_audience_or_other',
                modality='existing_assertion' if personal else 'proposed_or_conditional')


def entries(options, claims):
    return gate.validate_assessments(dict(options=[dict(letter=chr(65+i), kind='personal',
        claims=[dict(text=c, status='unsupported', reason='no_source', citations=[])])
        for i, c in enumerate(claims)]), options, {})


class ScopeRoles(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.flags = patch.multiple(config, CHOICE_SCOPE_ROLES=True, CHOICE_VERIFICATION_SCOPE=False)
        self.flags.start()
        self.addCleanup(self.flags.stop)

    async def test_audience_removal_remains_pending_independent_verification(self):
        data = entries(['A. Display books to draw visitors who enjoy debate.'], ['visitors who enjoy debate'])
        before = copy.deepcopy(data)
        with patch.object(llm, 'complete_json', AsyncMock(return_value=dict(claims=[decision('A:0')]))):
            out = await gate.reclassify_uncited(data, {})
        self.assertEqual(data, before)
        self.assertEqual(out[0]['claims'], [])
        self.assertEqual(out[0]['scope_reclassification'], 'awaiting_option_verification')

    async def test_mixed_existing_ownership_is_retained(self):
        data = entries(['A. Show visitors the violin you already own.'], ['the violin you already own'])
        result = decision('A:0', True) | dict(modality='mixed')
        with patch.object(llm, 'complete_json', AsyncMock(return_value=dict(claims=[result]))):
            out = await gate.reclassify_uncited(data, {})
        self.assertEqual(out[0]['claims'], data[0]['claims'])

    async def test_causal_assertion_guard_survives_wrong_scope(self):
        data = entries(['A. Since you own a camera, bring it.'], ['you own a camera'])
        diagnostics = {}
        with patch.object(llm, 'complete_json', AsyncMock(return_value=dict(claims=[decision('A:0')]))):
            out = await gate.reclassify_uncited(data, diagnostics)
        self.assertEqual(out[0]['claims'], data[0]['claims'])
        self.assertEqual(diagnostics['answer_premise_scope']['protected_causal_claims'], ['A:0'])

    async def test_partial_repair_freezes_accepted_decision(self):
        data = entries(['A. Invite visitors who enjoy debate.', 'B. Display the violin you already own.'],
                       ['visitors who enjoy debate', 'the violin you already own'])
        provider = AsyncMock(side_effect=[dict(claims=[decision('A:0')]), dict(claims=[decision('B:0', True)])])
        with patch.object(llm, 'complete_json', provider):
            out = await gate.reclassify_uncited(data, {})
        self.assertEqual(provider.await_count, 2)
        self.assertEqual(provider.call_args_list[1].kwargs['schema']['$defs']['RoleScope']['properties']['claim_id']['enum'], ['B:0'])
        self.assertEqual(out[0]['claims'], [])
        self.assertEqual(out[1]['claims'], data[1]['claims'])

    async def test_invalid_role_schema_keeps_original_claims(self):
        data = entries(['A. Invite visitors who enjoy debate.'], ['visitors who enjoy debate'])
        diagnostics = {}
        provider = AsyncMock(return_value=dict(claims=[dict(claim_id='A:0', personal_fact=False)]))
        with patch.object(llm, 'complete_json', provider):
            out = await gate.reclassify_uncited(data, diagnostics)
        self.assertEqual(out, data)
        self.assertEqual(diagnostics['answer_premise_scope']['status'], 'kept_original')

    async def test_small_budget_skips_without_erasing_claim(self):
        data = entries(['A. Invite visitors who enjoy debate.'], ['visitors who enjoy debate'])
        provider = AsyncMock()
        with patch.object(llm, 'complete_json', provider), budget.scope(seconds=240, calls=20, tokens=100):
            out = await gate.reclassify_uncited(data, {})
        provider.assert_not_called()
        self.assertEqual(out, data)

    def test_runtime_schema_matches_diagnostic_and_has_no_boolean_field(self):
        self.assertEqual(scope_roles.RoleScopes.model_json_schema(), ProbeRoleScopes.model_json_schema())
        self.assertNotIn('personal_fact', scope_roles.prefix(scope_prefix()))


if __name__ == '__main__':
    unittest.main()
