"""Measure missing query embeddings, reusing a hash-verified fixed-model archive."""
import argparse
import asyncio
import hashlib
import json
import logging
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import budget, config, embeddings
from evaluation_model_guard import check_runtime, check_versions


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


async def main(args):
    check_runtime()
    if config.FAKE or config.EMBED_MODEL != 'openai/text-embedding-v4' or config.EMBED_DIM != 256:
        raise ValueError('Requires the fixed real embedding model and dimension')
    cache_plan = json.loads((args.cache / 'plan.json').read_text())
    cache_result = json.loads((args.cache / 'rankings.json').read_text())
    for name, key in (('plan.json', 'plan_sha256'), ('vectors.npz', 'vectors_sha256')):
        if sha(args.cache / name) != cache_result[key]:
            raise ValueError('Embedding archive hash differs')
    if (cache_plan['model'], cache_plan['dimension']) != (config.EMBED_MODEL, config.EMBED_DIM):
        raise ValueError('Embedding archive space differs')
    prior = np.load(args.cache / 'vectors.npz', allow_pickle=False)['vectors']
    if prior.shape != (len(cache_plan['texts']), 256) or not np.isfinite(prior).all():
        raise ValueError('Invalid archived matrix')
    lookup = {text: prior[i] for i, text in enumerate(cache_plan['texts'])}
    texts, input_hashes, counts = [], {}, []
    for root in args.run:
        if (root / 'exit-code').read_text().strip() != '0':
            raise ValueError('Requires terminal source runs')
        trace_file = root / 'server-search-debug.jsonl'
        input_hashes[str(trace_file)] = sha(trace_file)
        traces = [json.loads(line) for line in trace_file.read_text().splitlines()]
        for t in traces:
            check_versions(t['versions'], 'recorded Search')
            if t['versions']['embedding_model'] != config.EMBED_MODEL:
                raise ValueError('Recorded embedding model differs')
            texts.extend(s['text'] for s in t['query_specs'])
        counts.append(len(traces))
    if counts != [56, 61]:
        raise ValueError('Requires the complete regression and validation cohorts')
    texts = list(dict.fromkeys(texts))
    missing = [text for text in texts if text not in lookup]
    args.output.mkdir(parents=True, exist_ok=False)
    plan = dict(texts=texts, missing_texts=missing, counts=counts, input_hashes=input_hashes,
        cache=str(args.cache), cache_plan_sha256=sha(args.cache / 'plan.json'),
        cache_vectors_sha256=sha(args.cache / 'vectors.npz'), model=config.EMBED_MODEL,
        dimension=config.EMBED_DIM, script_sha256=sha(Path(__file__)),
        limitation='New query vectors are measured now; they are not historical provider outputs.')
    (args.output / 'plan.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2) + '\n')
    handler = logging.FileHandler(args.output / 'provider.log', mode='x')
    logging.getLogger('aml.metrics').addHandler(handler)
    logging.getLogger('aml.metrics').setLevel(logging.INFO)
    print(json.dumps(dict(unique_queries=len(texts), reused=len(texts)-len(missing), missing=len(missing))), flush=True)
    with budget.scope(seconds=300, calls=200, tokens=1000000) as usage:
        fresh = await embeddings.embed(missing, stage='diagnostic.lexical_porter_queries')
    lookup.update({text: fresh[i] for i, text in enumerate(missing)})
    vectors = np.asarray([lookup[text] for text in texts], dtype=np.float32)
    if vectors.shape != (len(texts), 256) or not np.isfinite(vectors).all():
        raise ValueError('Invalid measured matrix')
    if not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5):
        raise ValueError('Vectors are not normalized')
    for i, text in enumerate(texts):
        if text not in missing and not np.array_equal(vectors[i], prior[cache_plan['texts'].index(text)]):
            raise ValueError('Cached vectors changed')
    if input_hashes != {p: sha(Path(p)) for p in input_hashes}:
        raise ValueError('Trace changed during measurement')
    np.savez_compressed(args.output / 'vectors.npz', vectors=vectors)
    result = dict(queries=len(texts), measured=len(missing), reused=len(texts)-len(missing),
        calls=usage.calls, tokens=usage.tokens, plan_sha256=sha(args.output / 'plan.json'),
        vectors_sha256=sha(args.output / 'vectors.npz'))
    (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, action='append', required=True)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    asyncio.run(main(p.parse_args()))
