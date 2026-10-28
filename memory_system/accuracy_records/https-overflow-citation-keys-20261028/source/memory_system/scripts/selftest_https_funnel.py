"""HTTPS audit must distinguish missing telemetry from actual evidence loss."""
import copy
import unittest

from audit_https_funnel import analyze, classify, matches, reader_gate, reader_gate_detail, gold_forget_matches


class HTTPSFunnelTests(unittest.TestCase):
    def setUp(self):
        self.row = dict(error_stage=None, observed_add=1, observed_search=True,
                        snippet_turns=1, prepared_hits=1, raw_hits=1, memory_hits=1,
                        pool_hits=1, packet_hits=1, catalog_hits=['s0'],
                        gold_valid_citations=1, gold='A', eligible=['A'], score=1)

    def test_forget_detail_requires_recorded_block(self):
        self.row.update(score=0, eligible=[], gold_alignment={'kind': 'generic'},
                        gold_forget_constraint_blocked=True)
        self.assertEqual(reader_gate(self.row), 'S5:eligibility')
        self.assertEqual(reader_gate_detail(self.row), 'S5:forget_constraint')
        self.row['gold_forget_constraint_blocked'] = False
        self.assertEqual(reader_gate_detail(self.row), 'S5:eligibility')

    def test_forget_does_not_override_eligibility_or_entailment(self):
        self.row.update(score=0, gold_forget_constraint_blocked=True)
        self.assertEqual(reader_gate_detail(self.row), 'S6:selection')
        self.row.update(eligible=[], gold_alignment={'kind': 'personal',
                         'validation_errors': ['not_entailed']})
        self.assertEqual(reader_gate_detail(self.row), 'S5:independent_entailment')

    def test_only_valid_positive_gold_constraint_matches_count(self):
        good = dict(letter='A', valid=True, constraint_id='r0')
        result = dict(answer_constraints=[good,
                      dict(letter='A', valid=True, violates=False),
                      dict(letter='A', valid=False),
                      dict(letter='B', valid=True), dict(letter='A')])
        self.assertEqual(gold_forget_matches(result, 'A'), [good])

    def test_empty_source_cannot_match_every_turn(self):
        self.assertFalse(matches('', ['I enjoy painting.']))

    def test_unresolved_support_is_not_semantic_rejection(self):
        self.row.update(eligible=[], score=0, gold_valid_citations=0,
                        gold_alignment={'kind': 'personal', 'validation_errors': ['support_unresolved']})
        self.assertEqual(reader_gate(self.row), 'S4:support')
        self.assertEqual(reader_gate_detail(self.row), 'S4:support_unresolved')

    def test_other_option_failure_does_not_reclassify_gold(self):
        self.row.update(score=0, unresolved_support_options=['B'])
        self.assertEqual(reader_gate_detail(self.row), 'S6:selection')

    def test_runtime_error_keeps_precedence(self):
        self.row.update(error_stage='answer', unresolved_support_options=['A'])
        self.assertEqual(reader_gate_detail(self.row), 'runtime:answer')

    def test_missing_logs_are_not_a_retrieval_failure(self):
        self.row.update(observed_search=False, catalog_hits=[], pool_hits=0)
        self.assertEqual(classify(self.row), 'unobserved:server_logs')

    def test_dataset_omission_is_not_ingestion_loss(self):
        self.row.update(prepared_hits=0, raw_hits=0, eligible=[], score=0)
        self.assertEqual(classify(self.row), 'S0:snippet_absent_from_dataset')
        self.row.update(prepared_hits=1)
        self.assertEqual(classify(self.row), 'S0:ingestion')

    def test_catalog_survival_overrides_missing_atomic_fact(self):
        self.row.update(memory_hits=0, pool_hits=0, packet_hits=0)
        self.assertEqual(classify(self.row), 'OK')

    def test_transport_is_not_entailment(self):
        self.row.update(eligible=[])
        self.assertEqual(classify(self.row), 'S5:eligibility')

    def test_generic_advice_needs_no_personal_citation(self):
        self.row.update(gold_kind='generic', gold_valid_citations=0, score=0)
        self.assertEqual(classify(self.row), 'S6:selection')

    def test_eligible_gold_proves_selection_loss_even_without_csv_snippet(self):
        self.row.update(raw_hits=0, memory_hits=0, pool_hits=0, packet_hits=0,
                        catalog_hits=[], score=0)
        self.assertEqual(classify(self.row), 'S6:selection')

    def test_result_identity_and_user_isolation(self):
        result = dict(conversation_id='c', qa_id='0', user_id='https:run:1:c',
                      search_id='search', packet_hash='hash', score=0,
                      prediction='B', answer_source_catalog=[])
        qa = dict(id='0', question='Which activity?', answer='Paint',
                  options=['A. Paint', 'B. Walk'], gold_labels=['A'])
        source = dict(role='user', content='I enjoy painting.',
                      request_id='other:session:0', message_index=0)
        memory = [dict(user_id='other', source_messages=[source], memories=[])]
        searches = [dict(search_id='search', user_id=result['user_id'],
                         query=qa['question'], options=qa['options'], packet_hash='hash')]
        prepared = {'c': dict(qa=[qa], sessions=[[source]], source_history='history32')}
        gold = [dict(user_query='{"content":"Which activity?"}', correct_answer='Paint',
                     chat_history_32k_link='history32', chat_history_128k_link='history128',
                     related_conversation_snippet='[{"role":"user","content":"I enjoy painting."}]',
                     preference='Painting')]
        row = analyze([result], memory, searches, prepared, gold)[0]
        self.assertEqual(row['raw_hits'], 0)
        self.assertEqual(row['stage'], 'unobserved:server_logs')
        wrong = copy.deepcopy(searches)
        wrong[0]['user_id'] = 'other'
        with self.assertRaisesRegex(ValueError, 'identity/packet'):
            analyze([result], memory, wrong, prepared, gold)
        gold[0]['chat_history_32k_link'] = 'different persona'
        with self.assertRaisesRegex(ValueError, 'Gold identity'):
            analyze([result], memory, searches, prepared, gold)

    def test_reader_rejection_is_separate_from_missing_benchmark_snippet(self):
        self.row.update(eligible=[], score=0, catalog_hits=[], packet_hits=0,
                        gold_alignment={'validation_errors': ['not_entailed']})
        self.assertEqual(classify(self.row), 'S3:packet')
        self.assertEqual(reader_gate(self.row), 'S5:independent_entailment')

    def test_child_rejection_and_missing_alignment(self):
        self.row.update(eligible=[], score=0, gold_alignment={
            'claims': [{'validation_errors': ['not_entailed']}]})
        self.assertEqual(reader_gate(self.row), 'S5:independent_entailment')
        self.row['gold_alignment'] = {}
        self.assertEqual(reader_gate(self.row), 'unobserved:answer_alignment')

    def test_eligible_option_precedes_stale_child_error(self):
        self.row.update(score=0, gold_alignment={'validation_errors': ['not_entailed']})
        self.assertEqual(reader_gate(self.row), 'S6:selection')


if __name__ == '__main__':
    unittest.main()
