"""Translate controlled relevance labels into valid listwise test responses.

Only test fixtures use these labels; production listwise ranking has no scores.
"""
import math


def from_scores(values):
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1 for v in values):
        return dict(ranking=[], irrelevant=[], groups=[])
    cutoff = .15 if max(values, default=0) >= .5 else 0
    return dict(ranking=sorted(range(len(values)), key=lambda i: (-values[i], i)),
                irrelevant=[i for i, value in enumerate(values) if value <= 0 or value < cutoff], groups=[])
