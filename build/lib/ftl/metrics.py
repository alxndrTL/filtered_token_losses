"""Metrics over per-token NLL arrays. Model-free: callers supply NLLs.

Conventions: pack masks have shape (W, L); NLL arrays have shape (W, L-1)
where nll[w, k] scores the token at window position k+1 given the prefix.
Masks are sliced [:, 1:] internally to align.
"""

import numpy as np

from .masks import copy_distance, first_occurrence_index

DIST_BINS = ((1, 32), (32, 128), (128, 512), (512, 2048))


def _align(mask):
    return mask[:, 1:]


def filtered_mean(nll, mask):
    """Mean NLL over masked positions."""
    m = _align(mask)
    return float((nll * m).sum() / m.sum())


def bootstrap_ci(nll, mask, n_boot=2000, seed=0):
    """95% cluster-bootstrap CI (over windows) of the filtered mean."""
    m = _align(mask)
    counts = m.sum(1).astype(float)
    vals = np.where(counts > 0, (nll * m).sum(1) / np.maximum(counts, 1), 0.0)
    rng = np.random.RandomState(seed)
    W = len(vals)
    idx = rng.randint(0, W, size=(n_boot, W))
    means = (vals[idx] * counts[idx]).sum(1) / np.maximum(counts[idx].sum(1), 1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_gap(nll_cand, nll_ref, mask, n_boot=2000, seed=0):
    """Paired Δ = cand − ref on masked positions, with bootstrap CI.
    Positive = candidate worse. Use for candidate-vs-baseline AND
    neighbor-vs-neighbor (pass the neighbor as ref)."""
    g = nll_cand - nll_ref
    lo, hi = bootstrap_ci(g, mask, n_boot=n_boot, seed=seed)
    return filtered_mean(g, mask), lo, hi


def recall_cliff(nll_cand, nll_ref, ids, n=5, bins=DIST_BINS):
    """Paired copy-gap stratified by distance to previous occurrence."""
    d = np.stack([copy_distance(r, n) for r in ids])[:, 1:]
    g = nll_cand - nll_ref
    out = {}
    for lo, hi in bins:
        sel = (d >= lo) & (d < hi)
        out[f"{lo}-{hi}"] = float((g * sel).sum() / max(sel.sum(), 1))
    return out


def hard_copy_mask(ids, selector_nll, tau=2.0, n=5):
    """Copy events whose FIRST occurrence cost the selector > tau nats.
    selector_nll must come from a model EXTERNAL to the compared pair.
    Returns (W, L) mask."""
    W, L = ids.shape
    hard = np.zeros((W, L), dtype=bool)
    for w in range(W):
        fo = first_occurrence_index(ids[w], n)
        has = np.where(fo >= 1)[0]
        j = fo[has]
        hard[w, has[selector_nll[w, j - 1] > tau]] = True
    return hard


def recall_efficiency(nll, ids, tau=2.0, n=5, bins=DIST_BINS):
    """Capability-invariant recall metric: 1 - sum NLL(repeat)/sum NLL(first)
    over copy events whose first occurrence cost THIS model > tau nats.
    Returns {"overall": x, "<lo>-<hi>": x, ...}. Higher = better recall."""
    fs, rs, ds = [], [], []
    for w in range(ids.shape[0]):
        fo = first_occurrence_index(ids[w], n)
        has = np.where(fo >= 1)[0]
        has = has[has >= 1]
        j = fo[has]
        fs.append(nll[w, j - 1])
        rs.append(nll[w, has - 1])
        ds.append(has - j)
    f, r, d = map(np.concatenate, (fs, rs, ds))
    sel = f > tau
    out = {"overall": float(1 - r[sel].sum() / f[sel].sum())}
    for lo, hi in bins:
        s = sel & (d >= lo) & (d < hi)
        if s.sum() >= 20:
            out[f"{lo}-{hi}"] = float(1 - r[s].sum() / f[s].sum())
    return out


def failure_rate(nll, mask, threshold=1.0):
    """Fraction of masked events with NLL > threshold (tail-robust companion
    to the heavy-tailed copy-5 mean)."""
    m = _align(mask)
    return float((nll[m] > threshold).mean())


def report(nll, pack, ref_nll=None, selector_nll=None, tau=2.0, prefix=""):
    """One-call flat dict of all metrics, ready for a metrics logger.

    nll:           candidate per-token NLLs (W, L-1)
    pack:          dict from ftl.pack (ids/copy5/nocopy/content)
    ref_nll:       optional matched-baseline NLLs -> paired gaps + cliff
    selector_nll:  optional external-selector NLLs -> hard-copy5 metrics
    """
    ids, copy5, nocopy = pack["ids"], pack["copy5"], pack["nocopy"]
    state = pack["content"] & nocopy if pack.get("content") is not None else None
    out = {
        f"{prefix}nll/all": filtered_mean(nll, np.ones_like(copy5)),
        f"{prefix}nll/copy5": filtered_mean(nll, copy5),
        f"{prefix}copy5/failure_rate": failure_rate(nll, copy5),
    }
    if state is not None and state.any():
        out[f"{prefix}nll/state"] = filtered_mean(nll, state)
    for k, v in recall_efficiency(nll, ids, tau=tau).items():
        out[f"{prefix}recall_eff/{k}"] = v
    if ref_nll is not None:
        for name, mask in [("all", np.ones_like(copy5)), ("copy5", copy5)]:
            gap, lo, hi = paired_gap(nll, ref_nll, mask)
            out[f"{prefix}gap/{name}"] = gap
            out[f"{prefix}gap/{name}_ci_lo"] = lo
            out[f"{prefix}gap/{name}_ci_hi"] = hi
        for k, v in recall_cliff(nll, ref_nll, ids).items():
            out[f"{prefix}cliff/{k}"] = v
    if selector_nll is not None:
        hard = hard_copy_mask(ids, selector_nll, tau=tau)
        out[f"{prefix}nll/hard_copy5"] = filtered_mean(nll, hard)
        if ref_nll is not None:
            gap, lo, hi = paired_gap(nll, ref_nll, hard)
            out[f"{prefix}gap/hard_copy5"] = gap
    return out
