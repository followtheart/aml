"""Semantic boundary segmentation (ULM §3.2; SeCom / EverMemOS).

Turn-level memories are too fragmented and session-level ones too noisy, so a
request is cut into topic segments where the embedding similarity between the
running segment and the next message drops sharply. Deterministic: no LLM.
"""
from typing import List

import numpy as np

from . import config


def segment_indices(vecs: np.ndarray, max_size: int,
                    min_size: int = None, drop: float = None) -> List[List[int]]:
    """Return message index groups; each group is contiguous and <= max_size."""
    min_size = config.SEGMENT_MIN_MESSAGES if min_size is None else min_size
    drop = config.SEGMENT_SIM_DROP if drop is None else drop
    n = len(vecs)
    if n == 0:
        return []
    segments, current = [], [0]
    centroid = np.array(vecs[0], dtype=np.float32)
    for i in range(1, n):
        sim = float(np.dot(_unit(centroid), _unit(vecs[i])))
        boundary = len(current) >= min_size and sim < drop
        if boundary or len(current) >= max_size:
            segments.append(current)
            current, centroid = [i], np.array(vecs[i], dtype=np.float32)
        else:
            current.append(i)
            centroid = centroid + vecs[i]
    segments.append(current)
    return segments


def _unit(v: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    return v / norm if norm else v
