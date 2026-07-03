"""Scale and Mamba2 comparison on the filtered-loss diagnostics.

Tiers (matched Pile + NeoX tokenizer):
  370m tier: pythia-410m vs mamba-370m vs mamba2-370m  (architecture A/B)
  1.4b tier: pythia-1.4b vs mamba-1.4b                 (does scale close the gap?)
"""

import json

import numpy as np

from analyze import filtered_stats
from distance_profile import BINS, boot_mean, copy_distance

TIERS = {
    "370m": {"anchor": "pythia", "cands": ["mamba", "mamba2"]},
    "1.4b": {"anchor": "pythia14", "cands": ["mamba14"]},
}
NAMES = {"pythia": "Pythia-410m", "mamba": "Mamba-370m", "mamba2": "Mamba2-370m",
         "pythia14": "Pythia-1.4b", "mamba14": "Mamba-1.4b"}
GROUPS = {"prose": ["pg19", "wiki"], "python": ["python"], "climbmix": ["climbmix"]}


def load_nll(key):
    out = dict(np.load(f"nll_{key}.npz"))
    out["climbmix"] = dict(np.load(f"nll_{key}_climbmix.npz"))["climbmix"]
    return out


def masks_for(dom):
    src = np.load("eval_data_climbmix.npz" if dom == "climbmix" else "eval_data.npz")
    content = src[f"{dom}_content"][:, 1:]
    copy5 = src[f"{dom}_copy5"][:, 1:]
    nocopy = src[f"{dom}_nocopy"][:, 1:]
    ids = src[f"{dom}_ids"]
    return {"ALL": np.ones_like(copy5, bool), "STATE": content & nocopy,
            "COPY5": copy5}, ids


def main():
    models = sorted({m for t in TIERS.values() for m in [t["anchor"]] + t["cands"]})
    nll = {m: load_nll(m) for m in models}

    print("=== Filtered losses (mean NLL, nats) ===")
    hdr = f"{'group':9s} {'filter':6s} " + " ".join(f"{NAMES[m]:>12s}" for m in models)
    print(hdr)
    for gname, doms in GROUPS.items():
        fm = {f: np.concatenate([masks_for(d)[0][f] for d in doms]) for f in
              ["ALL", "STATE", "COPY5"]}
        for f in ["ALL", "STATE", "COPY5"]:
            vals = []
            for m in models:
                cat = np.concatenate([nll[m][d] for d in doms])
                mean, *_ = filtered_stats(cat, fm[f])
                vals.append(mean)
            print(f"{gname:9s} {f:6s} " + " ".join(f"{v:12.3f}" for v in vals))

    print("\n=== Paired COPY5 gap vs tier anchor (nats, +=candidate worse; all domains pooled) ===")
    all_doms = ["pg19", "wiki", "python", "climbmix"]
    fmask = np.concatenate([masks_for(d)[0]["COPY5"] for d in all_doms])
    dists = np.concatenate([
        np.stack([copy_distance(r.tolist()) for r in masks_for(d)[1]])[:, 1:]
        for d in all_doms])
    for tier, cfg in TIERS.items():
        anchor = np.concatenate([nll[cfg["anchor"]][d] for d in all_doms])
        for c in cfg["cands"]:
            cand = np.concatenate([nll[c][d] for d in all_doms])
            g = cand - anchor
            mean, lo, hi, _ = filtered_stats(g, fmask)
            print(f"{NAMES[c]:12s} vs {NAMES[cfg['anchor']]:12s}: "
                  f"COPY5 Δ={mean:+.3f} [{lo:+.3f},{hi:+.3f}]", end="  cliff: ")
            parts = []
            for lo_b, hi_b in BINS:
                sel = (dists >= lo_b) & (dists < hi_b)
                counts = sel.sum(1).astype(float)
                vals = np.where(counts > 0,
                                (g * sel).sum(1) / np.maximum(counts, 1), 0)
                parts.append(f"{lo_b}-{hi_b}:{(g*sel).sum()/sel.sum():+.3f}")
            print(" ".join(parts))

    print("\n=== Planted-repeat retrieval gain (nats) ===")
    import glob
    probes = {}
    for f in glob.glob("planted_probe*.json"):
        probes.update(json.load(open(f)))
    gaps = ["64", "256", "1024", "1792"]
    print(f"{'model':12s} " + " ".join(f"gap{g:>5s}" for g in gaps))
    for m in models:
        if m in probes:
            print(f"{NAMES[m]:12s} " + " ".join(f"{probes[m][g]['gain']:8.2f}" for g in gaps))


if __name__ == "__main__":
    main()
