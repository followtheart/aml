import tempfile
from pathlib import Path
import unittest
from finalize_screened_source_https import reference_changes, provider_ledger, reader_statuses


class FinalizerTests(unittest.TestCase):
    def test_equal_coverage_totals_do_not_hide_lost_turn(self):
        before=[dict(turn_index=i,normalized_chars=10,full=bool(i)) for i in range(2)]
        after=[dict(turn_index=i,normalized_chars=10,full=not bool(i)) for i in range(2)]
        self.assertEqual(reference_changes(before,after),dict(gained=[0],lost=[1]))
        after[0]['normalized_chars']=11
        with self.assertRaises(ValueError):reference_changes(before,after)
        with self.assertRaises(ValueError):reference_changes(before,after[:1])

    def test_ledger_arm_separation_and_failed_usage_not_zero_billing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'provider.log'
            path.write_text('reader_arm=control provider_call kind=llm stage=eval.choice model=gpt-4o-mini status=ok attempt=1 total_tokens=10\n'
                'reader_arm=treatment provider_call kind=llm stage=eval.choice model=gpt-4o-mini status=cancelled attempt=1 total_tokens=0 usage_known=false\n'
                'reader_arm=treatment provider_call kind=llm stage=eval.choice model=gpt-4o-mini status=ok attempt=2 total_tokens=20\n')
            before=provider_ledger(path,arm='control');after=provider_ledger(path,arm='treatment')
        self.assertEqual((before['calls'],before['reported_tokens']),(1,10))
        self.assertEqual((after['calls'],after['reported_tokens'],after['retry_attempts']),(2,20,1))
        self.assertEqual(after['failed_without_trustworthy_usage'],1)

    def test_reader_partial_is_separate_from_invalid_and_provider_error(self):
        result=reader_statuses([dict(answer_calls=[dict(stage='witness',status='partial'),
            dict(stage='support',status='invalid',failure_phase='validator',error_type='ValueError'),
            dict(stage='support',status='error',failure_phase='provider',error_type='LLMError')])])
        self.assertEqual(result['statuses'],dict(partial=1,invalid=1,error=1))
        self.assertEqual(result['partial_by_stage'],dict(witness=1))
        self.assertEqual(result['failure_phases'],dict(validator=1,provider=1))


if __name__=='__main__':unittest.main()
