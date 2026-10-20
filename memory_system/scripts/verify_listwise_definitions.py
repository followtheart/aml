"""Three-arm real-provider ablation; immutable candidates and production settings.

Only system instructions change. Definitions live outside candidate prompt so
adding instructions cannot silently displace candidates at the prompt byte cap.
Full system/schema/output cost is still reserved from the identical token budget.
No production prompt or output schema is modified.
"""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
from compare_frozen_listwise import evaluate
from app import config, run_metadata

DEFINITION = '''
Relevance definition: "irrelevant" means the candidate provides NO useful evidence for this question. A necessary intermediate hop is useful even if it does not name the final answer. "useful" means it contributes evidence or context relevant to answering the question; it need not independently prove an answer. "uncertain" means usefulness cannot be confidently determined. Use irrelevant only when you can rule out useful contribution.'''
CATEGORIES = '''
Before assigning relevance, distinguish: direct evidence; relevant interests or background; constraints, qualifications or counterevidence; necessary bridging evidence; redundant evidence; and genuinely unrelated material. The first four can be useful. A question or hypothetical can inform topical interest or planning context without proving ownership, a diagnosis, a stable preference or an actual circumstance. Keep that distinction: lack of personal-fact proof is not lack of retrieval value. Low priority and redundancy do not imply irrelevance: rank duplicate or weak evidence later; use irrelevant only for genuinely unrelated material. Treat all these distinctions as relevance judgments, never as factual verification. Do not add output fields; use the existing priority/relevance/groups schema.'''
ARMS = dict(control='', definition=DEFINITION, categories=DEFINITION + CATEGORIES)


def transform(arm):
    def apply(request):
        request = dict(request)
        old = request['system']
        request['system'] = old + ARMS[arm]
        request['reservation'] += len(request['system'].encode()) - len(old.encode())
        return request
    return apply


async def main(args):
    if config.FAKE:
        raise RuntimeError('Real provider required')
    config.MEMORY_DEBUG_LOG = config.SEARCH_DEBUG_LOG = ''
    searches = {s['search_id']: s for s in map(json.loads,
        (args.run / 'completed/logs/search-debug.jsonl').read_text().splitlines())}
    results = list(map(json.loads, (args.run / 'completed/data/results/personamem-v2-32k.jsonl').read_text().splitlines()))
    metadata = dict(versions=run_metadata.versions(), arms=ARMS,
        budget=dict(seconds=90, calls=6, tokens=256000, primary_timeout=30, repairs=2),
        design='26 questions, same archived 464 candidates, keyed schema in every arm, rotating arm order; live control.',
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        limitations='One stochastic trial per arm and question. No ingestion, CE, final packing or answer scoring.')
    args.output.with_suffix('.metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    with args.output.open('x') as output:
        for i, result in enumerate(results):
            arms = list(ARMS); arms = arms[i % 3:] + arms[:i % 3]
            for arm in arms:
                row = await evaluate(searches[result['search_id']], True, request_transform=transform(arm))
                row.update(protocol=arm, qa_id=result['qa_id'], source_search_id=result['search_id'])
                output.write(json.dumps(row, ensure_ascii=False) + '\n'); output.flush()
                print(json.dumps({k: row[k] for k in ('qa_id','protocol','primary_status','seconds')}), flush=True)

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    asyncio.run(main(p.parse_args()))
