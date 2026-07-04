"""Planted-repeat probe: memorization-free in-context retrieval measurement.

Model-free construction; scoring is done by the caller's own forward pass.
"""

import numpy as np


def build_probe(token_pool, window_len=2048, span_len=32,
                gaps=(64, 256, 1024, 1792), per_gap=6, seed=1):
    """Random-token windows (sampled from token_pool's empirical distribution)
    with one span planted twice, `gap` tokens apart.

    Returns (ids (W, L) int64, spans list of (gap, first_start, second_start)).
    """
    rng = np.random.RandomState(seed)
    pool = np.asarray(token_pool).ravel()
    windows, spans = [], []
    for gap in gaps:
        for _ in range(per_gap):
            w = pool[rng.randint(0, len(pool), size=window_len)].copy()
            span = pool[rng.randint(0, len(pool), size=span_len)]
            p2 = window_len - span_len - 8
            p1 = p2 - gap
            if p1 < 8:
                raise ValueError(f"gap {gap} too large for window {window_len}")
            w[p1:p1 + span_len] = span
            w[p2:p2 + span_len] = span
            windows.append(w)
            spans.append((gap, p1, p2))
    return np.stack(windows).astype(np.int64), spans


def summarize(nll, spans, span_len=32):
    """nll: (W, L-1) from the caller's model on the probe ids.
    Returns {gap: {"first", "second", "gain"}}; gain = retrieval, in nats.
    The span's first token is skipped (unpredictable in both occurrences)."""
    out = {}
    for w, (gap, p1, p2) in enumerate(spans):
        d = out.setdefault(gap, {"first": [], "second": []})
        d["first"].append(nll[w, p1:p1 + span_len - 1].mean())
        d["second"].append(nll[w, p2:p2 + span_len - 1].mean())
    return {
        gap: {
            "first": float(np.mean(v["first"])),
            "second": float(np.mean(v["second"])),
            "gain": float(np.mean(v["first"]) - np.mean(v["second"])),
        } for gap, v in out.items()
    }
