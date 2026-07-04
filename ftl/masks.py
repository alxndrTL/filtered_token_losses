"""Token-level masks for filtered losses. Pure numpy, model-free.

All functions operate on one window of token ids (list[int] or 1-D array) and
return arrays indexed by window position 0..L-1. Position 0 is never scored;
metrics slice [1:] and align with NLL arrays of shape (L-1,) where nll[k]
scores the token at window position k+1.
"""

import numpy as np


def copy_n_mask(ids, n):
    """True at i if the n-gram ending at i already occurred in ids[:i]."""
    ids = list(ids)
    L = len(ids)
    m = np.zeros(L, dtype=bool)
    seen = set()
    for i in range(n - 1, L):
        g = tuple(ids[i - n + 1:i + 1])
        m[i] = g in seen
        seen.add(g)
    return m


def nocopy_mask(ids):
    """True at i if the token type at i has NOT occurred in ids[:i]
    (equivalently: not completing a repeated n-gram for any n)."""
    ids = list(ids)
    m = np.zeros(len(ids), dtype=bool)
    seen = set()
    for i, t in enumerate(ids):
        m[i] = t not in seen
        seen.add(t)
    return m


def first_occurrence_index(ids, n=5):
    """out[i] = position (gram end) of the FIRST occurrence of the n-gram
    ending at i, or -1 if i is itself the first occurrence / i < n-1."""
    ids = list(ids)
    L = len(ids)
    out = np.full(L, -1, dtype=np.int32)
    first = {}
    for i in range(n - 1, L):
        g = tuple(ids[i - n + 1:i + 1])
        if g in first:
            out[i] = first[g]
        else:
            first[g] = i
    return out


def copy_distance(ids, n=5):
    """dist[i] = tokens back to the end of the MOST RECENT earlier occurrence
    of the n-gram ending at i; 0 if none."""
    ids = list(ids)
    L = len(ids)
    dist = np.zeros(L, dtype=np.int32)
    last = {}
    for i in range(n - 1, L):
        g = tuple(ids[i - n + 1:i + 1])
        if g in last:
            dist[i] = i - last[g]
        last[g] = i
    return dist
