"""Offline regressions for explicit task-feedback experience memory."""
import os
import sys
import unittest
import copy
from unittest.mock import patch
from pydantic import ValidationError
from pathlib import Path

os.environ["AML_FAKE"] = "1"
os.environ["AML_MEMORY_DEBUG_LOG"] = ""
os.environ["AML_SEARCH_DEBUG_LOG"] = ""
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import experience, schemas, store


class ExperienceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.st = store.Store(":memory:")

    def tearDown(self):
        self.st.conn.close()

    def request(self, outcome="success", verified=False):
        return schemas.FeedbackRequest(
            user_id="u", session_id="s1", task="Fix parser test",
            outcome=outcome, trace="Changed parser and ran the failing test.",
            task_signature="python:test:parser", environment_verified=verified)

    async def test_feedback_creates_procedural_memory(self):
        ids = await experience.run_feedback(self.st, self.request("failure"))
        item = self.st.get_amus_by_ids(ids)[0]
        self.assertEqual(item["type"], "strategy")
        self.assertEqual(item["polarity"], "failure")
        self.assertEqual(item["harmful"], 0)
        self.assertEqual(item["epistemic_status"], "inferred")
        self.assertFalse(item["verified"])

    async def test_repeated_feedback_updates_counter(self):
        first = await experience.run_feedback(self.st, self.request())
        req = self.request()
        req.session_id = "s2"
        req.feedback_event_id = "event-2"
        req.used_memory_ids = first
        req.attribution_reason = "Applied this strategy in the traced attempt"
        second = await experience.run_feedback(self.st, req)
        self.assertEqual(first, second)
        item = self.st.get_amus_by_ids(first)[0]
        self.assertEqual(item["helpful"], 1)
        await experience.run_feedback(self.st, req)
        self.assertEqual(self.st.get_amus_by_ids(first)[0]["helpful"], 1)
        self.assertIn("event-2", item["support_sessions"])


    def structured(self, **updates):
        req = self.request(verified=True)
        req.feedback_event_id = 'verified-event'
        req.environment_signature = 'linux-python'
        req.code_version = 'v1'
        req.dependency_versions = {'python': '3.14'}
        req.verification_result = 'passed'
        req.trajectory = schemas.TaskTrajectory(task_type='debug', goal='Fix parser',
            constraints=['No API change'], acceptance_criteria=['Failing test passes'],
            steps=[schemas.TaskStep(step_id='test', action='Run test', output='passed', status='success')],
            metrics=[schemas.TaskMetric(name='passed', value=1, baseline=0, direction='higher')])
        for k,v in updates.items():setattr(req,k,v)
        return req

    def review_req(self, mid, version, status, event='verified-event', user='u'):
        return schemas.ExperienceReviewRequest(user_id=user, memory_id=mid, expected_version=version,
            status=status, reviewer='operator', rationale='Checked the execution and applicable scope.',
            evidence=[schemas.ExperienceEvidenceRef(event_id=event, quote=self.request().trace, relation='support')])

    async def test_structured_observation_is_stored_and_lesson_starts_as_hypothesis(self):
        req=self.structured(outcome='partial')
        ids=await experience.run_feedback(self.st,req)
        event=self.st.get_feedback('u','verified-event')
        self.assertEqual(event['payload']['trajectory']['metrics'][0]['baseline'],0)
        info=experience.inspect(self.st,'u',ids[0])
        self.assertEqual(info['experience']['status'],'hypothesis')
        self.assertEqual(info['independent_cases'],1)
        self.assertEqual(info['evidence'][0]['payload_hash'],event['payload_hash'])
        self.assertEqual(info['evidence'][0]['quote'],req.trace)
        self.assertFalse(self.st.get_amus_by_ids(ids)[0]['verified'])

    async def test_repeated_distillation_links_evidence_without_duplicate_case(self):
        req=self.structured()
        first=await experience.run_feedback(self.st,req)
        req.feedback_event_id='another-event';req.session_id='another-session'
        second=await experience.run_feedback(self.st,req)
        self.assertEqual(first,second)
        info=experience.inspect(self.st,'u',first[0])
        self.assertEqual({e['event_id'] for e in info['evidence']},{'verified-event','another-event'})
        self.assertEqual(info['independent_cases'],1)
        self.assertEqual(info['version'],2)
        await experience.run_feedback(self.st,req)
        self.assertEqual(experience.inspect(self.st,'u',first[0])['version'],2)

    async def test_review_transitions_are_versioned_and_do_not_auto_adopt(self):
        mid=(await experience.run_feedback(self.st,self.structured()))[0]
        result=experience.review(self.st,self.review_req(mid,1,'validated'))
        self.assertEqual(result['experience']['status'],'validated')
        with self.assertRaisesRegex(ValueError,'version'):
            experience.review(self.st,self.review_req(mid,1,'adopted'))
        result=experience.review(self.st,self.review_req(mid,2,'adopted'))
        self.assertEqual(result['version'],3)
        self.assertEqual(len(result['experience']['history']),2)
        result=experience.review(self.st,self.review_req(mid,3,'stale'))
        self.assertEqual(result['experience']['status'],'stale')

    async def test_cannot_skip_validation_or_reference_wrong_user_or_quote(self):
        mid=(await experience.run_feedback(self.st,self.structured()))[0]
        with self.assertRaisesRegex(ValueError,'transition'):
            experience.review(self.st,self.review_req(mid,1,'adopted'))
        with self.assertRaises(ValueError):
            experience.review(self.st,self.review_req(mid,1,'validated',user='v'))
        req=self.review_req(mid,1,'validated');req.evidence[0].quote='invented'
        with self.assertRaisesRegex(ValueError,'quote'):
            experience.review(self.st,req)
        self.assertEqual(experience.inspect(self.st,'u',mid)['version'],1)

    async def test_unverified_and_simulated_feedback_cannot_promote(self):
        for kind in ('observed','simulation'):
            req=self.structured(execution_kind=kind,feedback_event_id=kind)
            if kind=='observed':req.verification_result=None
            mid=(await experience.run_feedback(self.st,req))[0]
            with self.assertRaisesRegex(ValueError,'verified execution'):
                experience.review(self.st,self.review_req(mid,1,'validated',event=kind))

    async def test_environment_mismatch_does_not_validate(self):
        mid=(await experience.run_feedback(self.st,self.structured()))[0]
        await experience.run_feedback(self.st,self.structured(code_version='v2',feedback_event_id='v2'))
        with self.assertRaisesRegex(ValueError,'compatible'):
            experience.review(self.st,self.review_req(mid,1,'validated',event='v2'))

    async def test_partial_unknown_replay_and_simulation_do_not_count_as_outcome(self):
        ids=await experience.run_feedback(self.st,self.request())
        for i,(outcome,kind) in enumerate([('partial','observed'),('unknown','observed'),('success','replay'),('failure','simulation')]):
            req=self.request(outcome);req.execution_kind=kind;req.feedback_event_id=str(i)
            req.used_memory_ids=ids;req.attribution_reason='Used for this attempt'
            await experience.run_feedback(self.st,req)
        memory=self.st.get_amus_by_ids(ids)[0]
        self.assertEqual((memory['helpful'],memory['harmful']),(0,0))

    async def test_missing_or_cross_user_attribution_rejected_atomically(self):
        ids=await experience.run_feedback(self.st,self.request())
        req=self.structured(user_id='v',used_memory_ids=ids,attribution_reason='wrong user')
        with self.assertRaisesRegex(ValueError,'cross-user'):
            await experience.run_feedback(self.st,req)
        self.assertIsNone(self.st.get_feedback('v','verified-event'))
        self.assertEqual(self.st.get_amus_by_ids(ids)[0]['helpful'],0)

    async def test_purge_removes_observations_and_experience_history(self):
        mid=(await experience.run_feedback(self.st,self.structured()))[0]
        experience.review(self.st,self.review_req(mid,1,'validated'))
        self.st.purge_user('u')
        self.assertIsNone(self.st.get_feedback('u','verified-event'))
        self.assertEqual(self.st.conn.execute('SELECT count(*) FROM claim_versions WHERE user_id=?',('u',)).fetchone()[0],0)
        with self.assertRaises(ValueError):experience.inspect(self.st,'u',mid)

    def test_bad_metrics_and_duplicate_steps_are_rejected(self):
        with self.assertRaises(ValidationError):schemas.TaskMetric(name='x',value=float('nan'))
        step=schemas.TaskStep(step_id='1',action='test')
        with self.assertRaises(ValidationError):schemas.TaskTrajectory(task_type='test',goal='g',steps=[step,step])

    async def test_invalid_generated_quote_does_not_write(self):
        from app import llm
        async def invalid(*args,**kwargs):
            return {'items':[dict(type='strategy',title='x',description='x',content='x',evidence_quotes=['not in trace'])]}
        with patch.object(llm,'complete_json',invalid):
            with self.assertRaisesRegex(ValueError,'exact quotes'):
                await experience.run_feedback(self.st,self.structured())
        self.assertIsNone(self.st.get_feedback('u','verified-event'))

    def test_legacy_request_hash_unchanged(self):
        import hashlib,json
        req=self.request()
        old=req.model_dump(exclude={'feedback_event_id','trajectory','execution_kind','distill'})
        expected=hashlib.sha256(json.dumps(old,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
        self.assertEqual(experience.feedback_identity(req)[2],expected)

    async def test_observation_only_does_not_call_models(self):
        from app import llm
        req=self.structured(distill=False)
        with patch.object(llm,'complete_json',side_effect=AssertionError('No inference for observation')):
            self.assertEqual(await experience.run_feedback(self.st,req),[])
        self.assertIsNotNone(self.st.get_feedback('u','verified-event')['payload']['trajectory'])


if __name__ == "__main__":
    unittest.main(verbosity=2)
