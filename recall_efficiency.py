"""Recall efficiency: a real-data recall metric ~invariant to LM capability.

For each copy-5 event, compare the model's OWN NLL at the first occurrence of
the 5-gram (no copy available: pure language-knowledge cost) with its NLL at
the repeat. efficiency = 1 - sum(NLL_repeat) / sum(NLL_first), over events
whose first occurrence cost the model itself > TAU nats.

Within-model construction cancels LM capability to first order (validated:
Pythia 410m->1.4b moves it +0.005 while raw COPY5 moves ~13%). Distance
stratification gives the capability-invariant recall-range profile.
Computed entirely from archived per-token NLLs.
"""

import numpy as np

import analyze_large as al
from hard_copy import first_occurrence_index

DOMAINS = ["pg19", "wiki", "python", "climbmix"]
MODELS = ["pythia", "pythia14", "mamba", "mamba14", "mamba2", "rwkv"]
NAMES = {"pythia": "Pythia-410m", "pythia14": "Pythia-1.4b",
         "mamba": "Mamba-370m", "mamba14": "Mamba-1.4b",
         "mamba2": "Mamba2-370m", "rwkv": "RWKV-4-430m"}
TAU = 2.0
DBINS = [(1, 128), (128, 512), (512, 2048)]


def events(fo_maps, nll):
    """Yield (first_nll, repeat_nll, distance) arrays pooled over all windows."""
    fs, rs, ds = [], [], []
    for dom in DOMAINS:
        arr = nll[dom]
        for w, fo in enumerate(fo_maps[dom]):
            has = np.where(fo >= 1)[0]
            has = has[has >= 1]
            j = fo[has]
            fs.append(arr[w, j - 1])
            rs.append(arr[w, has - 1])
            ds.append(has - j)
    return np.concatenate(fs), np.concatenate(rs), np.concatenate(ds)


def main():
    fo_maps = {}
    for dom in DOMAINS:
        ids = al.masks_for(dom)[1]
        fo_maps[dom] = [first_occurrence_index(r.tolist()) for r in ids]

    print(f"Recall efficiency (own-selection first > {TAU} nats), "
          f"overall + by distance to first occurrence")
    print(f"{'model':12s} {'overall':>8s} " +
          " ".join(f"{lo}-{hi:>4d}" for lo, hi in DBINS))
    for m in MODELS:
        f, r, d = events(fo_maps, al.load_nll(m))
        sel = f > TAU
        cells = [f"{1 - r[sel].sum()/f[sel].sum():8.3f}"]
        for lo, hi in DBINS:
            s = sel & (d >= lo) & (d < hi)
            cells.append(f"{1 - r[s].sum()/f[s].sum():8.3f}")
        print(f"{NAMES[m]:12s} " + " ".join(cells))


if __name__ == "__main__":
    main()
