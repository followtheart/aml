"""Revisioned vector-only cache and bounded approximate search for large pools.

Small eligible pools use exact blockwise top-k. Large pools use deterministic
multi-probe random-hyperplane buckets, followed by exact scores of a bounded
candidate set. Eligibility is always supplied by the current SQLite snapshot.
"""
from collections import OrderedDict
import threading
import numpy as np
from . import budget, config

_cache, _lock = OrderedDict(), threading.RLock()
_minimum_epochs = {}


def evict(identity, user_id, epoch=None):
    with _lock:
        if epoch is not None:
            _minimum_epochs[(identity, user_id)] = max(epoch, _minimum_epochs.get((identity, user_id), 0))
        for key in list(_cache):
            if key[:2] == (identity, user_id):
                del _cache[key]


def get(key, load):
    with _lock:
        found = _cache.get(key)
        if found is not None:
            _cache.move_to_end(key)
            return found, True
    found = Index(load())
    with _lock:
        while _cache and sum(x.nbytes for x in _cache.values()) + found.nbytes > config.VECTOR_CACHE_BYTES:
            _cache.popitem(last=False)
        if found.nbytes <= config.VECTOR_CACHE_BYTES and key[3] >= _minimum_epochs.get(key[:2], 0):
            _cache[key] = found
    return found, False


class Index:
    def __init__(self, rows):
        self.ids, vectors = [], []
        for mid, blob in rows:
            budget.check()
            self.ids.append(mid)
            vectors.append(np.frombuffer(blob, dtype=np.float32))
        self.matrix = np.stack(vectors) if vectors else np.empty((0, config.EMBED_DIM), dtype=np.float32)
        self.positions = {mid: i for i, mid in enumerate(self.ids)}
        self.ids = np.array(self.ids)
        self.buckets = None
        # Include an allowance for Python lookup/bucket objects, not only arrays.
        self.nbytes = self.matrix.nbytes + self.ids.nbytes + len(self.ids) * 160
        if config.VECTOR_APPROXIMATE and len(self.ids) > config.VECTOR_EXACT_LIMIT:
            rng = np.random.default_rng(1701)
            self.planes = rng.standard_normal((3, self.matrix.shape[1], 10)).astype(np.float32)
            self.buckets = [dict() for _ in self.planes]
            for start in range(0, len(self.ids), config.VECTOR_CHUNK_SIZE):
                budget.check()
                block = self.matrix[start:start + config.VECTOR_CHUNK_SIZE]
                for table, planes in zip(self.buckets, self.planes):
                    codes = ((block @ planes >= 0) * (1 << np.arange(10))).sum(axis=1)
                    for offset, code in enumerate(codes):
                        table.setdefault(int(code), []).append(start + offset)
            self.nbytes += self.planes.nbytes + len(self.ids) * 3 * 40

    def _candidates(self, query, allowed, k):
        if self.buckets is None or len(allowed) <= config.VECTOR_EXACT_LIMIT:
            return sorted(allowed), False, len(allowed), False
        cap = max(k, config.VECTOR_CANDIDATE_LIMIT)
        found = set()
        visited, visit_limit = 0, max(k, config.VECTOR_CANDIDATE_LIMIT) * 16
        tables = []
        for table, planes in zip(self.buckets, self.planes):
            projection = query @ planes
            code = int(((projection >= 0) * (1 << np.arange(10))).sum())
            codes = np.array(sorted(table), dtype=np.int32)
            flips = ((codes[:, None] ^ code) & (1 << np.arange(10))) != 0
            # Probe uncertain hyperplanes first instead of arbitrary numeric
            # tie ordering among all buckets at the same Hamming distance.
            costs = flips @ (np.abs(projection) / np.maximum(np.linalg.norm(planes, axis=0), 1e-9))
            tables.append(iter(codes[np.argsort(costs, kind='stable')].tolist()))
        for _ in range(1024):
            budget.check()
            added = False
            for table, iterator in zip(self.buckets, tables):
                code = next(iterator, None)
                if code is None:
                    continue
                added = True
                for position in table[code]:
                    if visited % config.VECTOR_CHUNK_SIZE == 0:
                        budget.check()
                    visited += 1
                    if position in allowed:
                        found.add(position)
                    if len(found) >= cap:
                        return sorted(found), True, visited, False
                    if visited >= visit_limit:
                        return sorted(found), True, visited, True
            if not added:
                break
        return sorted(found), len(found) < len(allowed), visited, False

    def search(self, queries, allowed_ids, k):
        queries = np.atleast_2d(np.asarray(queries, dtype=np.float32))
        if len(self.ids) and self.matrix.shape[1] != queries.shape[1]:
            raise ValueError('Stored and query embedding dimensions differ')
        allowed = {self.positions[mid] for mid in allowed_ids if mid in self.positions}
        results, diagnostics = [], []
        for query in queries:
            budget.check()
            if k <= 0 or not allowed:
                results.append([])
                diagnostics.append(dict(approximate=False, eligible=len(allowed), scored=0))
                continue
            positions, approximate, visited, exhausted = self._candidates(query, allowed, k)
            best = []
            for start in range(0, len(positions), config.VECTOR_CHUNK_SIZE):
                budget.check()
                indices = np.array(positions[start:start + config.VECTOR_CHUNK_SIZE], dtype=np.int64)
                values = self.matrix[indices] @ query
                # Partition by score, including boundary ties before deterministic ID sorting.
                if len(values) > k:
                    threshold = np.partition(values, -k)[-k]
                    keep = np.flatnonzero(values >= threshold)
                    indices, values = indices[keep], values[keep]
                best.extend((float(v), str(self.ids[i])) for v, i in zip(values, indices))
                best = sorted(best, key=lambda x: (-x[0], x[1]))[:k]
            results.append([(mid, score) for score, mid in best])
            diagnostics.append(dict(approximate=approximate, eligible=len(allowed), scored=len(positions),
                                    bucket_visits=visited, candidate_budget_exhausted=exhausted))
        return results, diagnostics
