"""Synthetic rejection-path checks; does not claim real-trace reproducibility."""
import asyncio
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import audit_vector_capacity as target


async def validate():
    checks = []
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        prepared, probe, path = root / 'prepared.jsonl', root / 'probe.json', root / 'reader.jsonl'
        prepared.write_text('{}\n')
        probe.write_text('{}\n')
        plan = dict(run=str(root), prepared=str(prepared), probe=str(probe),
                    prepared_sha256=target.replay.digest(prepared.read_text()),
                    probe_sha256=target.replay.digest(probe.read_text()),
                    script_sha256=target.replay.digest(Path(target.replay.__file__).read_text()),
                    qa_ids=['fixture'], capacities=[40, 80], caveat='Synthetic fixture only.')
        path.with_suffix('.plan.json').write_text(json.dumps(plan))
        rows = []
        for arm in ['control', 'candidate']:
            events = [dict(kind='embedding', texts=['fixture query'], stage='fixture.embed', response=[[1.0, 0.0]]),
                      dict(kind='ce', query='fixture query', document_hashes=[target.replay.digest('fixture document')],
                           stage='fixture.ce', response=[0.5]),
                      dict(kind='llm', prompt_sha256=target.replay.digest('fixture prompt'), stage='fixture.llm',
                           schema_sha256=target.replay.digest(json.dumps({}, sort_keys=True)), response={'answer': 'A'})]
            search = dict(packet_hash='same', status='retrieved', coverage_manifest={'search_degraded': False},
                          routes=[{'channel': 'vector', 'candidates': [{'id': 'one'}]}], fused=[{'id': 'one'}],
                          cascade={'coarse': {'selected_ids': ['one']}, 'fine': {'selected_ids': ['one']},
                                   'listwise': {'candidate_ids': ['one'], 'selected_ids': ['one']}})
            rows.append(dict(qa_id='fixture', arm=arm, prediction='A', score=1,
                             search=search, diagnostics={'answer_eligible_options': ['A']}, provider_events=events))

        async def replay(args):
            for row in rows:
                vectors = await target.embeddings.embed(['fixture query'], stage='fixture.embed')
                assert vectors.tolist() == [[1.0, 0.0]]
                scores = await target.cross_encoder.rerank('fixture query', ['fixture document'], stage='fixture.ce')
                assert scores == [0.5]
                result = await target.llm.complete_json('fixture prompt', schema={}, stage='fixture.llm')
                assert result == {'answer': 'A'}
            args.output.write_text(''.join(json.dumps(row) + '\n' for row in rows))

        mutations = {
            'embedding_input': lambda r: r[0]['provider_events'][0].update(texts=['changed']),
            'ce_document': lambda r: r[0]['provider_events'][1].update(document_hashes=['changed']),
            'llm_prompt': lambda r: r[0]['provider_events'][2].update(prompt_sha256='changed'),
            'llm_schema': lambda r: r[0]['provider_events'][2].update(schema_sha256='changed'),
            'missing_arm': lambda r: r.pop(),
            'route': lambda r: r[0]['search']['routes'][0].update(candidates=[]),
            'fine_ids': lambda r: r[0]['search']['cascade']['fine'].update(selected_ids=[]),
            'packet': lambda r: r[0]['search'].update(packet_hash='changed'),
            'score': lambda r: r[0].update(score=0),
        }
        with patch.object(target.replay, 'main', replay):
            path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            assert (await target.audit(path))['status'] == 'passed'
            checks.append('valid_fixture')
            for name, mutate in mutations.items():
                changed = copy.deepcopy(rows)
                mutate(changed)
                path.write_text(''.join(json.dumps(row) + '\n' for row in changed))
                try:
                    await target.audit(path)
                except (AssertionError, ValueError):
                    checks.append('reject_' + name)
                else:
                    raise AssertionError('Accepted mutation: ' + name)
            for row in rows:
                event = row['provider_events'][2]
                event.pop('response')
                event.update(error_type='LLMError', error_detail='fixture invalid structured response')
                row['diagnostics']['answer_calls'] = [dict(stage=event['stage'],
                    prompt_sha256=event['prompt_sha256'], error_type='LLMError', failure_phase='provider_parse')]

            async def replay_failure(args):
                for row in rows:
                    await target.embeddings.embed(['fixture query'], stage='fixture.embed')
                    await target.cross_encoder.rerank('fixture query', ['fixture document'], stage='fixture.ce')
                    try:
                        await target.llm.complete_json('fixture prompt', schema={}, stage='fixture.llm')
                    except target.llm.LLMError as error:
                        assert isinstance(error.__cause__, target.metrics.ResponseParseError)
                    else:
                        raise AssertionError('Recorded failure was swallowed')
                args.output.write_text(''.join(json.dumps(row) + '\n' for row in rows))

            path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            with patch.object(target.replay, 'main', replay_failure):
                audited = await target.audit(path)
            assert audited['status'] == 'passed' and len(audited['provider_errors']) == 2
            checks.append('preserve_provider_parse_exception_cause')
    return dict(status='passed', checks=checks,
                auditor_sha256=hashlib.sha256(Path(target.__file__).read_bytes()).hexdigest(),
                caveat='Synthetic provider/input/outcome rejection checks only. Complete real search and reader replay audit is still required.')


if __name__ == '__main__':
    print(json.dumps(asyncio.run(validate()), indent=2))
