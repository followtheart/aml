"""Isolated, synthetic recall/latency check; never contacts a model or live DB."""
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

read = Path.read_text
with patch.object(Path, 'read_text', lambda path, *args, **kwargs:
                  '' if path.name == '.env' else read(path, *args, **kwargs)):
    from app import config, vector_index


def main():
    rows = []
    # Fixed policies make reruns comparable even with unrelated local env vars.
    with patch.object(config, 'VECTOR_APPROXIMATE', True), \
            patch.object(config, 'VECTOR_EXACT_LIMIT', 20000), \
            patch.object(config, 'VECTOR_CANDIDATE_LIMIT', 8192), \
            patch.object(config, 'VECTOR_CHUNK_SIZE', 2048):
        for count in (10000, 25000, 100000):
            rng = np.random.default_rng(20260918)
            matrix = rng.standard_normal((count, 64)).astype(np.float32)
            matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
            queries = rng.standard_normal((20, 64)).astype(np.float32)
            queries /= np.linalg.norm(queries, axis=1, keepdims=True)
            ids = [f'v{i:06}' for i in range(count)]
            start = time.perf_counter()
            index = vector_index.Index((mid, v.tobytes()) for mid, v in zip(ids, matrix))
            built = time.perf_counter() - start
            start = time.perf_counter()
            found, diagnostics = index.search(queries, ids, 10)
            elapsed = time.perf_counter() - start
            exact = np.argsort(-(queries @ matrix.T), axis=1)[:, :10]
            recall = [len({mid for mid, _ in result} & {ids[i] for i in truth}) / 10
                      for result, truth in zip(found, exact)]
            rows.append(dict(vectors=count, dimensions=64, queries=20, top_k=10,
                build_seconds=round(built, 6), total_search_seconds=round(elapsed, 6),
                mean_recall_at_10=round(float(np.mean(recall)), 4),
                min_recall_at_10=min(recall), approximate=diagnostics[0]['approximate'],
                max_scored=max(x['scored'] for x in diagnostics),
                max_bucket_visits=max(x.get('bucket_visits', 0) for x in diagnostics)))
    print(json.dumps(dict(seed=20260918, approximate_opt_in=True, exact_limit=20000, candidate_limit=8192,
        caveat='Synthetic normalized vectors; not answer accuracy or end-to-end latency.',
        results=rows), indent=2))


if __name__ == '__main__':
    main()
