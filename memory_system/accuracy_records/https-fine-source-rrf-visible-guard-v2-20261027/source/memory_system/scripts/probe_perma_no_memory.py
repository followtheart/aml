"""Paired PERMA control using the unchanged local reader and empty evidence."""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import config, eval_data, eval_scoring, run_metadata
from app.perma import PROTOCOL


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = list(eval_data.read_records(args.prepared))
    if config.FAKE:
        raise ValueError('Quality comparison requires real models')
    for row in rows:
        eval_data.validate(row)
        if row.get('protocol') != PROTOCOL:
            raise ValueError('Expected PERMA MCQ data')
    digest = hashlib.sha256(args.prepared.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    errors = 0
    with args.output.open('x') as stream:
        for row in rows:
            for qa in row['qa']:
                pred, score, diag = await eval_scoring.evaluate(qa, [])
                errors += bool(diag.get('error_stage'))
                result = dict(conversation_id=row['conversation_id'], qa_id=qa['id'],
                              category=qa['category'], prediction=pred, score=score,
                              protocol=PROTOCOL, model=config.LLM_MODEL, fake=False,
                              condition='no_memory', data_sha256=digest,
                              versions=run_metadata.versions(), **diag)
                stream.write(json.dumps(result, ensure_ascii=False) + '\n')
                stream.flush()
                print(f"{row['conversation_id']}: prediction={pred!r} score={score}", flush=True)
    return int(errors > 0)


if __name__ == '__main__':
    sys.exit(asyncio.run(main()))
