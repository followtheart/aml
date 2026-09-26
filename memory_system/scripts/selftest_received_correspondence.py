"""A received letter cannot be cited as the current user's own speech."""
import importlib.util,json,os,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import answer_choice as gate
if os.environ.get('ANSWER_CANDIDATE'):
    spec=importlib.util.spec_from_file_location('app.answer_candidate',os.environ['ANSWER_CANDIDATE'])
    gate=importlib.util.module_from_spec(spec);spec.loader.exec_module(gate)

class CorrespondenceTests(unittest.TestCase):
    def check(self,text,quote='I own a telescope.',basis='self_report',subject='current_user'):
        citation=gate.Citation(source_id='s0',quote=quote,basis=basis,subject=subject)
        return gate._check_citation(citation,'you own a telescope',{'s0':dict(role='user',text=text)},'ownership')

    def test_received_body_is_not_self_report(self):
        for marker in ['---','```']:
            r=self.check(f'I received this email from a teammate.\n\n{marker}\nI own a telescope.\n{marker}\n')
            self.assertFalse(r['valid']);self.assertIn('third_party_or_hypothetical',r['validation_errors'])

    def test_user_statement_outside_received_block_is_preserved(self):
        for text in ['I own a telescope. I got this email from a colleague:\n---\nCan I borrow it?\n---',
                     'I received an email from a colleague:\n---\nWhat do you own?\n---\n\nI own a telescope.']:
            self.assertTrue(self.check(text)['anchor'])

    def test_own_draft_is_preserved(self):
        for intro in ['Please polish my email to a teammate.',
                      'I received an email from a teammate. Here is my reply:']:
            self.assertTrue(self.check(intro+'\n---\nI own a telescope.\n---')['anchor'])

    def test_received_context_can_remain_non_anchor(self):
        r=self.check('I received an email from a teammate:\n---\nI own a telescope.\n---',basis='context',subject='third_party')
        self.assertTrue(r['valid']);self.assertFalse(r['anchor'])

    def test_plain_self_report_with_unrelated_received_mail_mention_is_preserved(self):
        self.assertTrue(self.check('I received some mail today. I own a telescope.')['anchor'])

if __name__=='__main__':unittest.main()
