"""Contract, lifecycle and tenant isolation checks for the replacement service."""
import tempfile
import unittest
from pathlib import Path
from fastapi.testclient import TestClient
from memory_connection.app.config import Settings
from memory_connection.app.https_app import create_app
from memory_connection.app.https_schemas import SearchResponse, AddResponse, PurgeResponse
from memory_connection.tests.test_pipeline import FixtureProvider


class HTTPSTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.settings = Settings(db_path=str(Path(self.temp.name) / 'memory.db'), api_key='test')
        self.provider = FixtureProvider()
        self.client = TestClient(create_app(self.settings, self.provider))
        self.client.__enter__()
        self.headers = {'Authorization': 'Bearer test'}
        self.add = dict(request_id=' req ', user_id=' u ', session_id=' s ', messages=[dict(
            role='user', content='甲源', source_event_id='source', trust_scope=['personal'], timestamp=1700000000000)])
        self.query = dict(user_id=' u ', query='乙需', top_k=10, options=None,
                          min_revision=None, include_history=None)
        self.feedback = dict(user_id=' u ', session_id=' s ', task='task', outcome='success', trace='exact trace',
            feedback_event_id='f1', environment_verified=True, environment_signature='env',
            code_version='v1', dependency_versions={'python': '3'}, verification_result='passed')

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.temp.cleanup()

    def post(self, path, data):
        return self.client.post(path, json=data, headers=self.headers)

    def test_auth_all_routes_and_validation(self):
        self.assertEqual(self.client.get('/health').json(), {'status': 'ok'})
        for method, path, data in [('post', '/add', self.add), ('post', '/search', self.query),
            ('post', '/feedback', self.feedback), ('post', '/experience/review', {}),
            ('get', '/feedback/u/f', None), ('get', '/experience/u/m', None),
            ('delete', '/memory/u', None), ('delete', '/users/u', None), ('post', '/answer', {})]:
            response = self.client.request(method, path, json=data)
            self.assertEqual(response.status_code, 401, path)
        self.assertEqual(self.post('/search', {'query': 'x', 'user_id': 'u'}).status_code, 422)
        self.assertEqual(self.post('/search', dict(self.query, top_k=101)).status_code, 422)
        self.assertEqual(self.post('/add', dict(self.add, messages=[{'role': 'tool', 'content': 'x'}])).status_code, 422)

    def test_add_search_provenance_idempotence_budget_and_purge(self):
        response = self.post('/add', self.add)
        self.assertEqual(response.status_code, 200, response.text)
        added = AddResponse.model_validate(response.json())
        self.assertEqual(added.request_id, ' req ')
        self.assertEqual(self.post('/add', self.add).json(), response.json())
        changed = dict(self.add, session_id='other')
        self.assertEqual(self.post('/add', changed).status_code, 422)
        found = self.post('/search', dict(self.query, min_revision=added.write_revision))
        self.assertEqual(found.status_code, 200, found.text)
        result = SearchResponse.model_validate(found.json())
        self.assertEqual(result.data[0].content, '甲源')
        self.assertEqual(result.data[0].sources[0]['source_event_id'], 'source')
        self.assertEqual(result.data[0].sources[0]['trust_scope'], ['personal'])
        self.assertTrue(result.data[0].is_constraint)
        self.assertEqual(found.headers['X-Memory-System'], 'memory_connection')
        self.assertEqual(len(found.headers['X-Request-ID']), 32)
        self.assertEqual(self.post('/search', dict(self.query, evidence_token_budget=256)).json()['data'], [])
        self.assertEqual(len(self.post('/search', self.query).json()['data']), 1)
        self.assertEqual(self.post('/search', dict(self.query, user_id='other')).json()['data'], [])
        purged = self.client.delete('/memory/%20u%20', headers=self.headers)
        receipt = PurgeResponse.model_validate(purged.json())
        self.assertEqual(receipt.row_counts['events'], 1)
        self.assertEqual(self.post('/search', self.query).json()['data'], [])
        self.assertGreater(self.post('/add', self.add).json()['scope_epoch'], added.scope_epoch)

    def test_empty_and_large_message_arrays_match_old_contract(self):
        self.assertEqual(self.post('/add', dict(self.add, messages=[])).status_code, 200)
        request = dict(self.add, request_id='large', messages=[dict(role='user', content='甲源') for _ in range(101)])
        self.assertEqual(self.post('/add', request).status_code, 200)

    def test_feedback_persistence_review_and_full_purge(self):
        response = self.post('/feedback', self.feedback)
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(self.post('/feedback', self.feedback).json(), result)
        self.assertEqual(self.post('/feedback', dict(self.feedback, trace='different')).status_code, 422)
        memory_id = result['memory_ids'][0]
        self.assertEqual(self.client.get('/feedback/other/f1', headers=self.headers).status_code, 404)
        self.assertEqual(self.client.get(f'/experience/other/{memory_id}', headers=self.headers).status_code, 404)
        review = dict(user_id=' u ', memory_id=memory_id, expected_version=1, status='validated',
            reviewer='operator', rationale='verified', evidence=[dict(event_id='f1', quote='exact trace', relation='support')])
        self.assertEqual(self.post('/experience/review', dict(review, evidence=[dict(event_id='f1', quote='invented')])).status_code, 422)
        promoted = self.post('/experience/review', review)
        self.assertEqual(promoted.status_code, 200, promoted.text)
        self.assertEqual(promoted.json()['experience']['status'], 'validated')
        self.assertEqual(self.post('/experience/review', review).status_code, 422)
        self.client.__exit__(None, None, None)
        self.client = TestClient(create_app(self.settings, FixtureProvider()))
        self.client.__enter__()
        record = self.client.get(f'/experience/%20u%20/{memory_id}', headers=self.headers).json()
        self.assertEqual(record['version'], 2)
        deleted = self.client.delete('/memory/%20u%20', headers=self.headers).json()
        self.assertEqual(deleted['row_counts']['feedback_events'], 1)
        self.assertEqual(deleted['row_counts']['experiences'], 1)
        self.assertEqual(self.client.get('/feedback/%20u%20/f1', headers=self.headers).status_code, 404)
        self.assertEqual(self.client.get(f'/experience/%20u%20/{memory_id}', headers=self.headers).status_code, 404)

    def test_feedback_observation_and_simulation_cannot_promote(self):
        response = self.post('/feedback', dict(self.feedback, distill=False))
        self.assertEqual(response.json()['memory_ids'], [])
        simulation = dict(self.feedback, feedback_event_id='sim', execution_kind='simulation')
        memory_id = self.post('/feedback', simulation).json()['memory_ids'][0]
        review = dict(user_id=' u ', memory_id=memory_id, expected_version=1, status='validated',
            reviewer='operator', rationale='check', evidence=[dict(event_id='f1', quote='exact trace', relation='support')])
        self.assertEqual(self.post('/experience/review', review).status_code, 422)
        self.assertEqual(self.post('/feedback', dict(self.feedback, feedback_event_id='bad', used_memory_ids=['foreign'], attribution_reason='test')).status_code, 422)

    def test_feedback_racing_purge_cannot_resurrect(self):
        self.provider.before_expand = lambda: self.client.app.state.service.store.purge(' u ')
        self.assertEqual(self.post('/feedback', self.feedback).status_code, 422)
        self.assertEqual(self.client.get('/feedback/%20u%20/f1', headers=self.headers).status_code, 404)

    def test_rejected_experience_not_recalled(self):
        memory_id = self.post('/feedback', self.feedback).json()['memory_ids'][0]
        review = dict(user_id=' u ', memory_id=memory_id, expected_version=1, status='rejected',
            reviewer='operator', rationale='reject', evidence=[dict(event_id='f1', quote='exact trace')])
        self.assertEqual(self.post('/experience/review', review).status_code, 200)
        self.assertEqual(self.post('/search', self.query).json()['data'], [])


if __name__ == '__main__':
    unittest.main()
