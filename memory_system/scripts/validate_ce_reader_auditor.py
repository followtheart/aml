"""Synthetic integrity checks; no provider calls or performance claims."""
import asyncio
import copy
import hashlib
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import audit_ce_isolated_reader as target


async def validate():
    checks = []
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        probe = root / 'probe.jsonl'
        probe.write_text('{}\n')
        path = root / 'reader.jsonl'
        plan = dict(probe=str(probe), probe_sha256=hashlib.sha256(probe.read_bytes()).hexdigest(),
                    qa_ids=['fixture'], frozen_packets_exact=1, caveat='Synthetic fixture only.')
        path.with_suffix('.plan.json').write_text(json.dumps(plan))
        base = []
        for arm in ['control', 'candidate']:
            prompt = arm + ' full prompt'
            call = dict(stage='fixture.' + arm,
                        prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(), response={'ok': True})
            base.append(dict(qa_id='fixture', arm=arm, prediction='A', score=1,
                             ranking=dict(packet_hash='original', calls=[call], cascade={
                                 'fine': {'selected_ids': ['one']},
                                 'listwise': {'candidate_ids': ['one'], 'selected_ids': ['one']}}),
                             diagnostics={'answer_calls': [], 'answer_eligible_options': ['A']}))

        async def replay(args):
            for row in base:
                response = await target.llm.complete_json(row['arm'] + ' full prompt',
                                                          stage='fixture.' + row['arm'])
                assert response == {'ok': True}
            args.output.write_text(''.join(json.dumps(r) + '\n' for r in base))

        mutations = {
            'prompt_hash': lambda rows: rows[0]['ranking']['calls'][0].update(prompt_sha256='bad'),
            'stage': lambda rows: rows[0]['ranking']['calls'][0].update(stage='bad'),
            'missing_pair': lambda rows: rows.pop(),
            'arm_order': lambda rows: rows.reverse(),
            'packet_hash': lambda rows: rows[0]['ranking'].update(packet_hash='bad'),
            'admitted_ids': lambda rows: rows[0]['ranking']['cascade']['listwise'].update(candidate_ids=[]),
            'eligible': lambda rows: rows[0]['diagnostics'].update(answer_eligible_options=[]),
            'score': lambda rows: rows[0].update(score=0),
            'unused_call': lambda rows: rows[-1]['diagnostics']['answer_calls'].append(
                copy.deepcopy(rows[-1]['ranking']['calls'][0])),
        }
        with patch.object(target, 'replay', replay):
            path.write_text(''.join(json.dumps(r) + '\n' for r in base))
            assert (await target.audit(path))['status'] == 'passed'
            checks.append('valid_fixture')
            for name, mutate in mutations.items():
                rows = copy.deepcopy(base)
                mutate(rows)
                path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
                try:
                    await target.audit(path)
                except (AssertionError, ValueError):
                    checks.append('reject_' + name)
                else:
                    raise AssertionError('Auditor accepted mutation: ' + name)
    return dict(status='passed', checks=checks,
                auditor_sha256=hashlib.sha256(Path(target.__file__).read_bytes()).hexdigest(),
                caveat='Synthetic replay stub verifies integrity rejection paths only; full real trace audit remains required.')


if __name__ == '__main__':
    print(json.dumps(asyncio.run(validate()), indent=2))
