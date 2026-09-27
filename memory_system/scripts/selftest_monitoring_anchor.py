"""Direct monitoring anchors preserve provenance and do not establish diagnoses."""
import copy
import unittest
import selftest_answer_choice
from app import answer_choice as gate

class MonitoringAnchorTests(unittest.TestCase):
    def fixture(self):
        claim_text='Since you are keeping an eye on your cholesterol'
        weak='When a lipid panel shows numbers over the guideline limits, how significant is that trend?'
        direct='Why would cholesterol numbers change between two routine checkups, even if my diet and exercise stayed the same?'
        sources={'weak':dict(id='weak',role='user',text=weak),'direct':dict(id='direct',role='user',text=direct)}
        prior=gate._check_citation(gate.Citation(source_id='weak',quote=weak,basis='topic_interest',subject='current_user'),claim_text,sources,'condition')
        self.assertTrue(prior['valid'])
        return [dict(letter='A',kind='personal',validation_status='valid',validation_errors=[],claims=[dict(text=claim_text,status='supported',reason='none',premise_type='condition',citations=[prior],validation_errors=[])])],sources
    def test_adds_direct_anchor_preserving_original_reference_and_status(self):
        entries,sources=self.fixture();prior=copy.deepcopy(entries[0]['claims'][0]['citations'][0]);d={}
        gate.recover_direct_monitoring_citations(entries,sources,d);claim=entries[0]['claims'][0]
        self.assertEqual(claim['citations'][0],prior)
        self.assertEqual(claim['citations'][1]['quote'],sources['direct']['text'])
        self.assertTrue(claim['citations'][1]['valid'])
        self.assertEqual(claim['status'],'supported')
        self.assertEqual(claim['citation_recovery'],'awaiting_entailment')
    def test_absent_reference_is_recovered_without_promoting_status(self):
        entries,sources=self.fixture();claim=entries[0]['claims'][0];claim.update(citations=[],status='unsupported',reason='no_source')
        gate.recover_direct_monitoring_citations(entries,sources,{})
        self.assertEqual(claim['citations'][0]['source_id'],'direct');self.assertEqual(claim['status'],'unsupported')
    def test_idempotent(self):
        entries,sources=self.fixture();gate.recover_direct_monitoring_citations(entries,sources,{});before=copy.deepcopy(entries)
        gate.recover_direct_monitoring_citations(entries,sources,{});self.assertEqual(entries,before)
    def test_no_personal_checkup_source_no_change(self):
        entries,sources=self.fixture();del sources['direct'];before=copy.deepcopy(entries)
        gate.recover_direct_monitoring_citations(entries,sources,{});self.assertEqual(entries,before)
    def test_assistant_and_persona_sources_cannot_supply_anchor(self):
        for change in [dict(role='assistant'),dict(declared='persona')]:
            entries,sources=self.fixture();sources['direct'].update(change);before=copy.deepcopy(entries)
            gate.recover_direct_monitoring_citations(entries,sources,{});self.assertEqual(entries,before)
    def test_attributed_hypothetical_and_other_person_do_not_supply_anchor(self):
        for text in [
            'Jordan wrote: Why would cholesterol numbers change between routine checkups, even if my diet stayed the same?',
            'Imagine cholesterol numbers changing between routine checkups, even if my diet stayed the same.',
            'Why would my sister’s cholesterol numbers change between routine checkups, even if my diet stayed the same?',
            'If I went for routine checkups, would cholesterol numbers change even if my diet stayed the same?',
            'Why do cholesterol readings change between routine checkups? My diet is varied.',
            'Why would my blood pressure readings change between routine checkups?']:
            entries,sources=self.fixture();sources['direct']['text']=text;before=copy.deepcopy(entries)
            gate.recover_direct_monitoring_citations(entries,sources,{});self.assertEqual(entries,before,text)
    def test_diagnosis_and_negated_monitoring_claims_not_enriched(self):
        for text in ['Since you have high cholesterol','Since you are treating high cholesterol','Since you are not monitoring your cholesterol']:
            entries,sources=self.fixture();entries[0]['claims'][0]['text']=text;before=copy.deepcopy(entries)
            gate.recover_direct_monitoring_citations(entries,sources,{});self.assertEqual(entries,before,text)
    def test_invalid_claim_and_existing_references_not_enriched(self):
        for change in ['claim_error','invalid_reference','direct_reference','unresolved']:
            entries,sources=self.fixture();claim=entries[0]['claims'][0]
            if change=='claim_error':claim['validation_errors']=['quote_not_in_source']
            elif change=='invalid_reference':claim['citations'][0]['valid']=False
            elif change=='direct_reference':claim['citations'][0]['basis']='self_report'
            else:entries[0]['validation_status']='unresolved'
            before=copy.deepcopy(entries);gate.recover_direct_monitoring_citations(entries,sources,{});self.assertEqual(entries,before)
    def test_relatives_metric_is_not_established_by_users_checkups(self):
        for text in ["Since you are keeping an eye on your father's cholesterol",
                     "Since you are monitoring your sister's cholesterol"]:
            entries,sources=self.fixture();entries[0]['claims'][0]['text']=text;before=copy.deepcopy(entries)
            gate.recover_direct_monitoring_citations(entries,sources,{})
            self.assertEqual(entries,before)
    def test_reference_limit_retained(self):
        entries,sources=self.fixture();entries[0]['claims'][0]['citations']*=3;before=copy.deepcopy(entries)
        gate.recover_direct_monitoring_citations(entries,sources,{});self.assertEqual(entries,before)

if __name__=='__main__':unittest.main()
