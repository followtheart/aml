"""Integration regression tests for source attribution before profile mutation."""
import asyncio,importlib.util,os,sys,unittest
from pathlib import Path
from unittest.mock import AsyncMock,patch
import numpy as np
os.environ['AML_FAKE']='1'
sys.path.insert(0,str(Path.cwd()))
from app import config,integrity,profile,schemas,store
class ProfileAttributionIntegration(unittest.IsolatedAsyncioTestCase):
    async def trial(self,text,quote,accepted):
        st=store.Store(':memory:')
        try:
            request=schemas.AddRequest(request_id='r',user_id='u',session_id='s',messages=[schemas.Message(role='user',content=text)])
            st.save_messages(request)
            mid=st.insert_amu(user_id='u',session_id='s',content='A statement about enjoying pottery.',embedding=np.array([1.,0.],dtype=np.float32))
            st.link_sources(mid,'r',[0])
            fact=dict(content='A statement about enjoying pottery.',type='fact',_sources=[0],evidence=[dict(message_index=0,quote=quote)])
            proposed={'items':[dict(content='The user enjoys pottery.',basis='stated',support_ids=[mid])]}
            replies=[proposed,{'valid':True,'reason':'explicit own statement'}] if accepted else [proposed]
            with patch.object(config,'FAKE',False),patch.object(profile.llm,'complete_json',AsyncMock(side_effect=replies)) as calls,patch.object(profile,'embed',AsyncMock(return_value=[np.array([1.,0.],dtype=np.float32)])) as embeddings:
                await profile.consolidate(st,request,[(mid,fact)])
            prefs=st.get_by_type('u',['preference'])
            self.assertEqual(len(prefs),int(accepted))
            self.assertEqual(calls.call_count,2 if accepted else 1)
            self.assertEqual(embeddings.call_count,int(accepted))
            self.assertIsNotNone(st.conn.execute('SELECT id FROM amu WHERE id=?',(mid,)).fetchone())
            if accepted:self.assertEqual([d['source_id'] for d in st.dependencies_for(prefs[0]['id'])],[mid])
        finally:st.conn.close()
    async def test_received_correspondence_cannot_create_a_profile(self):
        await self.trial('I received an email from Pat.\n---\nI enjoy pottery.\n---','I enjoy pottery.',False)
    async def test_own_draft_still_reaches_semantic_verification(self):
        await self.trial('Here is my draft:\n---\nI enjoy pottery.\n---','I enjoy pottery.',True)
    async def test_user_statement_outside_received_block_is_preserved(self):
        await self.trial('I got an email from Pat.\n---\nI enjoy music.\n---\nI enjoy pottery.','I enjoy pottery.',True)


class OriginalUserSupport(unittest.IsolatedAsyncioTestCase):
    async def trial(self, roles, indices, *, enabled=True, semantic=True, basis='stated'):
        st = store.Store(':memory:')
        try:
            req = schemas.AddRequest(request_id='r', user_id='u', session_id='s',
                messages=[schemas.Message(role=role, content='I enjoy pottery.') for role in roles])
            st.save_messages(req)
            mid = st.insert_amu(user_id='u', session_id='s', content='Pottery was discussed.',
                               embedding=np.array([1., 0.], dtype=np.float32))
            st.link_sources(mid, 'r', indices)
            fact = dict(content='Pottery was discussed.', type='fact', _sources=indices,
                        evidence=[dict(message_index=i, quote='I enjoy pottery.') for i in indices])
            proposal = {'items': [dict(content='The user enjoys pottery.', basis=basis, support_ids=[mid])]}
            user_supported = any(roles[i] == 'user' for i in indices)
            checked = not enabled or user_supported
            accepted = checked and semantic
            replies = [proposal] + ([{'valid': semantic}] if checked else [])
            with patch.object(config, 'FAKE', False), \
                    patch.object(config, 'PROFILE_REQUIRE_USER_SUPPORT', enabled), \
                    patch.object(config, 'PROFILE_MIN_SUPPORT', 1), \
                    patch.object(profile.llm, 'complete_json', AsyncMock(side_effect=replies)) as calls, \
                    patch.object(profile, 'embed', AsyncMock(return_value=[np.array([1., 0.], dtype=np.float32)])) as embed:
                await profile.consolidate(st, req, [(mid, fact)])
            self.assertEqual(len(st.get_by_type('u', ['preference'])), int(accepted))
            self.assertEqual(calls.call_count, 1 + int(checked))
            self.assertEqual(embed.call_count, int(accepted))
            self.assertEqual(st.sources_for_amu(mid)[0]['content'], 'I enjoy pottery.')
        finally:
            st.conn.close()

    async def test_assistant_only_stated_and_inferred_promotions_rejected(self):
        for basis in ('stated', 'inferred'):
            with self.subTest(basis=basis):
                await self.trial(['assistant'], [0], basis=basis)

    async def test_uncited_user_message_cannot_launder_assistant_support(self):
        await self.trial(['user', 'assistant'], [1])

    async def test_original_user_statement_reaches_verification(self):
        await self.trial(['user'], [0])

    async def test_mixed_sources_still_require_semantic_acceptance(self):
        await self.trial(['user', 'assistant'], [0, 1], semantic=False)

    async def test_disabled_flag_preserves_existing_path(self):
        await self.trial(['assistant'], [0], enabled=False)


class AttributionSpans(unittest.TestCase):
    def check(self, text, quote, **extra):
        messages = [schemas.Message(role='user', content=text)]
        return integrity.received_correspondence_only(
            {'evidence': [dict(message_index=0, quote=quote, **extra)]}, messages)

    def test_received_correspondence(self):
        self.assertTrue(self.check('I received an email from Pat.\n---\nI enjoy pottery.\n---', 'I enjoy pottery.'))

    def test_own_draft_and_reply(self):
        for prefix in ('Here is my draft:', 'I got an email from Pat. Here is my reply:'):
            self.assertFalse(self.check(prefix + '\n---\nI enjoy pottery.\n---', 'I enjoy pottery.'))

    def test_statement_outside_received_block(self):
        self.assertFalse(self.check('I got an email from Pat.\n---\nI enjoy music.\n---\nI enjoy pottery.', 'I enjoy pottery.'))

    def test_missing_or_ambiguous_quote(self):
        self.assertFalse(self.check('I enjoy pottery.', 'I enjoy swimming.'))
        self.assertFalse(self.check('I got an email from Pat.\n---\nI enjoy pottery.\n---\nI enjoy pottery.', 'I enjoy pottery.'))

    def test_unclosed_block_is_not_assumed_received(self):
        self.assertFalse(self.check('I received an email from Pat.\n---\nI enjoy pottery.', 'I enjoy pottery.'))

    def test_valid_offsets_resolve_repeated_quote(self):
        text = 'I got an email from Pat.\n---\nI enjoy pottery.\n---\nI enjoy pottery.'
        start = text.index('I enjoy pottery.')
        self.assertTrue(self.check(text, 'I enjoy pottery.', start=start, end=start+len('I enjoy pottery.')))
        start = text.rindex('I enjoy pottery.')
        self.assertFalse(self.check(text, 'I enjoy pottery.', start=start, end=start+len('I enjoy pottery.')))

    def test_invalid_structure_does_not_guess_attribution(self):
        messages = [schemas.Message(role='user', content='I enjoy pottery.')]
        for evidence in (None, [], ['invalid'], [{'message_index': True, 'quote': 'I enjoy pottery.'}],
                         [{'message_index': 2, 'quote': 'I enjoy pottery.'}]):
            self.assertFalse(integrity.received_correspondence_only({'evidence': evidence}, messages))

if __name__=='__main__':unittest.main()
