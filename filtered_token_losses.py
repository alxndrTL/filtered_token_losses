"""filtered_token_losses.py — measurement-only recall/state diagnostics for
LLM architecture research. Single portable file, numpy-only, no model code.
Based on arXiv:2606.20936 (Li & Merrill) + extensions; see the repo's RECAP.md
for the empirical validation of every metric here.

Copy this file into your codebase. The integration boundary: your code
produces per-token NLL arrays on a frozen eval pack; this file turns them
into capability-resolved metrics.

    import filtered_token_losses as ftl

    # once per campaign (packs are TOKENIZER-SPECIFIC — freeze and version):
    pack = ftl.build_pack(doc_iter, tokenizer, n_windows=300, window_len=2048)
    ftl.save_pack("evalpack_v1.npz", pack)

    # at every eval checkpoint of every candidate:
    nll = per_token_nll(model, pack["ids"])       # your forward pass, (W, L-1)
    np.save(f"nll/{run}/{step}.npy", nll.astype(np.float16))   # archive!
    logger.log(ftl.report(nll, pack, ref_nll=baseline_nll))

    # occasionally, the pure-mechanism recall check:
    ids, spans = ftl.build_probe(pack["ids"])
    logger.log(ftl.summarize_probe(per_token_nll(model, ids), spans))

Conventions: pack masks have shape (W, L) over window positions; NLL arrays
have shape (W, L-1) where nll[w, k] scores the token at window position k+1
given the prefix. Position 0 is never scored.

Metric cheat sheet (all from the same forward pass as validation loss):
  nll/all              standard validation loss
  nll/state            content words ∩ first occurrence -> state-conditioned
                       readout (favors recurrence)
  nll/copy5            repeated-5-gram positions -> visible-prefix retrieval
                       (favors attention); heavy-tailed, mean driven by rare
                       hard retrievals
  copy5/failure_rate   fraction of copy events with NLL > 1 nat (tail-robust)
  recall_eff/*         1 - NLL(repeat)/NLL(first occurrence), own-selected
                       hard events: ~invariant to LM capability, sensitive to
                       the retrieval mechanism. Watch the 512-2048 bin.
  gap/*                paired Δ vs a matched baseline (+ = candidate worse)
  cliff/*              paired copy-gap by distance -> localizes recall range
  hard_copy5           copy events whose first occurrence surprised an
                       EXTERNAL selector model (> tau nats): removes
                       generic-language repeats
"""

import numpy as np

__all__ = [
    "copy_n_mask", "nocopy_mask", "first_occurrence_index", "copy_distance",
    "build_pack", "save_pack", "load_pack",
    "filtered_mean", "bootstrap_ci", "paired_gap", "recall_cliff",
    "hard_copy_mask", "recall_efficiency", "failure_rate", "report",
    "build_probe", "summarize_probe",
]

DIST_BINS = ((1, 32), (32, 128), (128, 512), (512, 2048))


# --------------------------------------------------------------------------
# masks (per window of token ids; indexed by window position 0..L-1)
# --------------------------------------------------------------------------

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
    """True at i if the token type at i has NOT occurred in ids[:i]."""
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


# --------------------------------------------------------------------------
# eval pack: freeze once per (tokenizer, window length, corpus)
# --------------------------------------------------------------------------

def build_pack(doc_iter, tokenizer, n_windows, window_len=2048,
               content_fn=None, eos_id=None):
    """Pack documents (iterable of str) into full windows with EOS separators.

    tokenizer: HF-style, called as tokenizer(text, add_special_tokens=False,
    return_offsets_mapping=bool(content_fn)).
    content_fn: optional text -> bool char mask (True on content-word chars);
    enables the STATE filter. Returns dict: ids/content/copy5/nocopy, (W, L).
    """
    if eos_id is None:
        eos_id = tokenizer.eos_token_id
    L = window_len
    stream_ids, stream_content = [], []
    ids_list, content_list = [], []
    for text in doc_iter:
        enc = tokenizer(text, add_special_tokens=False,
                        return_offsets_mapping=content_fn is not None)
        stream_ids.extend(enc["input_ids"])
        if content_fn is not None:
            cmask = content_fn(text)
            stream_content.extend(
                cmask[s:e].any() if e > s else False
                for s, e in enc["offset_mapping"])
        else:
            stream_content.extend([False] * len(enc["input_ids"]))
        stream_ids.append(eos_id)
        stream_content.append(False)
        while len(stream_ids) >= L:
            ids_list.append(np.array(stream_ids[:L], dtype=np.int32))
            content_list.append(np.array(stream_content[:L], dtype=bool))
            del stream_ids[:L], stream_content[:L]
        if len(ids_list) >= n_windows:
            break
    if len(ids_list) < n_windows:
        raise ValueError(f"corpus exhausted at {len(ids_list)}/{n_windows} windows")
    ids = np.stack(ids_list[:n_windows])
    return {
        "ids": ids,
        "content": np.stack(content_list[:n_windows]),
        "copy5": np.stack([copy_n_mask(r, 5) for r in ids]),
        "nocopy": np.stack([nocopy_mask(r) for r in ids]),
    }


def save_pack(path, pack):
    np.savez_compressed(path, **pack)


def load_pack(path):
    return dict(np.load(path))


# --------------------------------------------------------------------------
# metrics over per-token NLL arrays (model-free; callers supply NLLs)
# --------------------------------------------------------------------------

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
    Returns {"overall": x, "<lo>-<hi>": x, ...}. Higher = better recall.
    (Validated: Pythia 410m->1.4b shifts it +0.005; raw copy-5 shifts ~13%.)"""
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
    pack:          dict with ids/copy5/nocopy and optional content, (W, L)
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


# --------------------------------------------------------------------------
# planted-repeat probe: memorization-free in-context retrieval
# --------------------------------------------------------------------------

def build_probe(token_pool, window_len=2048, span_len=32,
                gaps=(64, 256, 1024, 1792), per_gap=6, seed=1):
    """Random-token windows (sampled from token_pool's empirical distribution)
    with one span planted twice, `gap` tokens apart. The second occurrence is
    predictable ONLY via in-context retrieval — no memorization confound.

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


def summarize_probe(nll, spans, span_len=32):
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
