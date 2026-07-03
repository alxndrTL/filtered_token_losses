"""Distance-stratified recall profile (post-hoc query over archived per-token NLLs).

For every position completing a repeated 5-gram, compute the distance (in tokens)
back to the end of the most recent earlier occurrence, bin it, and report the
paired gap Δ = NLL(candidate) − NLL(transformer) per bin. A recall-healthy
architecture is flat in distance; a recall cliff shows as Δ growing with distance.
"""

import json

import numpy as np

MODELS = ["pythia", "mamba", "rwkv"]
DOMAINS = ["pg19", "wiki", "python"]
BINS = [(1, 32), (32, 128), (128, 512), (512, 2048)]
N = 5
RNG = np.random.RandomState(0)


def copy_distance(ids):
    """dist[i] = tokens back to end of most recent earlier occurrence of the
    5-gram ending at i; 0 if none."""
    L = len(ids)
    dist = np.zeros(L, dtype=np.int32)
    last = {}
    for i in range(N - 1, L):
        g = tuple(ids[i - N + 1:i + 1])
        if g in last:
            dist[i] = i - last[g]
        last[g] = i
    return dist


def boot_mean(vals, counts, n_boot=2000):
    W = len(vals)
    idx = RNG.randint(0, W, size=(n_boot, W))
    m = (vals[idx] * counts[idx]).sum(1) / np.maximum(counts[idx].sum(1), 1)
    return np.percentile(m, [2.5, 97.5])


def main():
    data = np.load("eval_data.npz")
    nll = {m: dict(np.load(f"nll_{m}.npz")) for m in MODELS}

    out = {}
    # pool all domains; also keep prose-only
    for scope, doms in [("all", DOMAINS), ("prose", ["pg19", "wiki"])]:
        out[scope] = {m: [] for m in MODELS[1:]}
        dists, gaps = [], {m: [] for m in MODELS[1:]}
        for dom in doms:
            ids = data[f"{dom}_ids"]
            d = np.stack([copy_distance(row.tolist()) for row in ids])[:, 1:]
            dists.append(d)
            for m in MODELS[1:]:
                gaps[m].append(nll[m][dom] - nll["pythia"][dom])
        d = np.concatenate(dists)
        for m in MODELS[1:]:
            g = np.concatenate(gaps[m])
            for lo, hi in BINS:
                sel = (d >= lo) & (d < hi)
                counts = sel.sum(1).astype(float)
                vals = np.where(counts > 0, (g * sel).sum(1) / np.maximum(counts, 1), 0)
                mean = (g * sel).sum() / sel.sum()
                ci = boot_mean(vals, counts)
                out[scope][m].append({"bin": f"{lo}-{hi}", "mean": float(mean),
                                      "lo": float(ci[0]), "hi": float(ci[1]),
                                      "count": int(sel.sum())})
    with open("distance_profile.json", "w") as f:
        json.dump(out, f, indent=1)
    for m in MODELS[1:]:
        print(m, " | ".join(f"{p['bin']}: {p['mean']:+.3f} [{p['lo']:+.3f},{p['hi']:+.3f}] n={p['count']}"
                            for p in out["all"][m]))


if __name__ == "__main__":
    main()
