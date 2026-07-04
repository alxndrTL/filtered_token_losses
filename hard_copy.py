"""HARD-COPY5: copy-5 events restricted to context-specific repeats, on real data.

A copy-5 event at position i has a first occurrence of the same 5-gram ending at
some earlier position j. If a reference transformer's NLL at j (first sight, no
copy available) exceeds tau, the continuation is NOT predictable from generic
language statistics — so low NLL at i requires actual retrieval. Selection uses
a transformer EXTERNAL to each comparison tier (cross-selection) to avoid
selection bias: 370m tier selected by Pythia-1.4b, 1.4b tier by Pythia-410m.

Computed entirely from archived per-token NLLs — no model forwards.
"""

import numpy as np

import analyze_large as al
from analyze import filtered_stats

N = 5
DOMAINS = ["pg19", "wiki", "python", "climbmix"]


def first_occurrence_index(ids):
    """For each position i, index j (end of FIRST occurrence of the 5-gram
    ending at i), or -1 if i is itself the first occurrence / no 5-gram."""
    L = len(ids)
    out = np.full(L, -1, dtype=np.int32)
    first = {}
    for i in range(N - 1, L):
        g = tuple(ids[i - N + 1:i + 1])
        if g in first:
            out[i] = first[g]
        else:
            first[g] = i
    return out


def build(selector_nll):
    """Masks over target positions 1..L-1: (copy5, hard) given selector NLLs."""
    copy_m, hard_m = {}, {}
    for dom in DOMAINS:
        ids = al.masks_for(dom)[1]
        W, L = ids.shape
        c = np.zeros((W, L), bool)
        h = np.zeros((W, L), bool)
        for w in range(W):
            fo = first_occurrence_index(ids[w].tolist())
            has = fo >= 0
            c[w] = has
            # selector NLL at first occurrence j -> array index j-1 (targets
            # start at position 1); j == 0 can't be scored, drop those
            j = fo[has]
            ok = j >= 1
            idx = np.where(has)[0][ok]
            h[w, idx] = selector_nll[dom][w, j[ok] - 1] > TAU
        copy_m[dom] = c[:, 1:]
        hard_m[dom] = h[:, 1:]
    return copy_m, hard_m


def gap(nll, cand, anchor, mask):
    g = np.concatenate([nll[cand][d] - nll[anchor][d] for d in DOMAINS])
    m = np.concatenate([mask[d] for d in DOMAINS])
    return filtered_stats(g, m)


TAU = 2.0

def main():
    global TAU
    models = ["pythia", "pythia14", "mamba", "mamba14", "mamba2", "rwkv"]
    nll = {m: al.load_nll(m) for m in models}

    for tau in [1.0, 2.0, 3.0]:
        TAU = tau
        copy_small, hard_small = build(nll["pythia14"])   # selects for 370m tier
        _, hard_large = build(nll["pythia"])              # selects for 1.4b tier
        n_all = sum(copy_small[d].sum() for d in DOMAINS)
        n_hard = sum(hard_small[d].sum() for d in DOMAINS)
        print(f"\n=== tau={tau:.0f} nats: {n_hard}/{n_all} copy-5 events survive "
              f"({100*n_hard/n_all:.0f}%) ===")
        rows = [("Mamba-370m", "mamba", "pythia", hard_small, copy_small),
                ("Mamba2-370m", "mamba2", "pythia", hard_small, copy_small),
                ("RWKV-4-430m", "rwkv", "pythia", hard_small, copy_small),
                ("Mamba-1.4b", "mamba14", "pythia14", hard_large, copy_small)]
        for name, cand, anchor, hard, copy_m in rows:
            m_all, lo_a, hi_a, _ = gap(nll, cand, anchor, copy_m)
            m_h, lo_h, hi_h, _ = gap(nll, cand, anchor, hard)
            print(f"{name:12s} vs {anchor:9s}: COPY5 {m_all:+.3f} [{lo_a:+.3f},{hi_a:+.3f}]"
                  f"  ->  HARD-COPY5 {m_h:+.3f} [{lo_h:+.3f},{hi_h:+.3f}]")


if __name__ == "__main__":
    main()
