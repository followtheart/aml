"""Dispatch, failure and budget checks for the experimental reader."""
import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import budget, config, direct_catalog_reader as reader, eval_scoring
from audit_https_funnel import classify, reader_gate, reader_gate_detail

qa = dict(question='Which?', options=['A. One', 'B. Two'], scoring='choice',
          qa_type='single_choice', gold_labels=['A'])

async def main():
    count = 0
    for response in [{'answer':'A'}, {'answer':'Z'}, {'answer':['A']}, {'answer':'A','extra':1}]:
        with patch.object(config,'CHOICE_DIRECT_CATALOG_READER',True), patch.object(reader.llm,'complete_json',AsyncMock(return_value=response)) as call:
            pred, score, diag = await eval_scoring.evaluate(qa, [])
            assert call.await_count == 1
            assert diag['answer_reader_mode']=='direct_catalog'
            assert 'answer_eligible_options' not in diag
            assert diag['answer_validation']=='selection_only_no_entailment_check'
            assert (pred,score)==(('A',1.0) if response=={'answer':'A'} else ('',0.0))
            if not pred: assert diag['error_stage']=='answer'
            count += 1
    with patch.object(config,'CHOICE_DIRECT_CATALOG_READER',True), patch.object(reader.llm,'complete_json',AsyncMock(side_effect=TimeoutError('test'))):
        pred,score,diag=await eval_scoring.evaluate(qa,[])
        assert diag['answer_calls'][0]['status']=='error' and diag['error_type']=='TimeoutError'
        count+=1
    with patch.object(config,'CHOICE_DIRECT_CATALOG_READER',False), patch.object(eval_scoring.answer_choice,'answer',AsyncMock(return_value='A')) as old:
        assert (await eval_scoring.evaluate(qa,[]))[:2]==('A',1.0)
        assert old.await_count==1
        count+=1
    async def observe(*args,**kwargs):
        assert budget.current.get() is parent
        return {'answer':'A'}
    with budget.scope(calls=1) as parent, patch.object(reader.llm,'complete_json',observe):
        assert await reader.answer(qa,[],{})=='A'
        count+=1
    for fn in [classify,reader_gate,reader_gate_detail]:
        assert fn(dict(answer_reader_mode='direct_catalog',score=0,error_stage=None))=='S6:direct_selection'
        assert fn(dict(answer_reader_mode='direct_catalog',score=0,error_stage='answer'))=='runtime:answer'
        assert fn(dict(answer_reader_mode='direct_catalog',score=1,error_stage=None))=='OK'
        count+=1
    print(f'{count} direct catalog checks passed')

asyncio.run(main())
