"""Filtered-token-loss analysis (arXiv:2606.20936 Fig. 7 replication on open models).

Filters (measure-only, same per-token NLL as standard validation):
  ALL          every scored position
  STATE        content words ∩ no-copy (state-conditioned readout proxy)
  COPY5        completes a repeated 5-gram (visible-prefix retrieval proxy)

Outputs: results.json, results.md, fig7_repro.png
"""

import json

import numpy as np

MODELS = ["pythia", "mamba", "rwkv"]
LABELS = {"pythia": "Pythia-410m (Transformer)",
          "mamba": "Mamba-370m (SSM)",
          "rwkv": "RWKV-4-430m (RNN)"}
DOMAINS = ["pg19", "wiki", "python"]
DOM_LABEL = {"pg19": "PG-19 (books)", "wiki": "Wikipedia", "python": "Python"}
RNG = np.random.RandomState(0)
N_BOOT = 2000


def copy_n_mask(ids, n):
    """True where the n-gram ending at i appeared earlier in the window."""
    L = len(ids)
    m = np.zeros(L, dtype=bool)
    seen = set()
    for i in range(n - 1, L):
        g = tuple(ids[i - n + 1:i + 1])
        m[i] = g in seen
        seen.add(g)
    return m


def boot_ci(per_window_vals, per_window_counts):
    """Cluster bootstrap over windows of a count-weighted mean."""
    W = len(per_window_vals)
    idx = RNG.randint(0, W, size=(N_BOOT, W))
    v = per_window_vals[idx]
    c = per_window_counts[idx]
    means = (v * c).sum(1) / np.maximum(c.sum(1), 1)
    return np.percentile(means, [2.5, 97.5])


def filtered_stats(nll, mask):
    """nll: (W, L-1); mask: (W, L-1) bool. Weighted mean + bootstrap CI."""
    counts = mask.sum(1)
    sums = (nll * mask).sum(1)
    ok = counts > 0
    mean = sums[ok].sum() / counts[ok].sum()
    vals = np.where(counts > 0, sums / np.maximum(counts, 1), 0.0)
    lo, hi = boot_ci(vals, counts.astype(float))
    return float(mean), float(lo), float(hi), int(counts.sum())


def main():
    data = np.load("eval_data.npz")
    nll = {m: dict(np.load(f"nll_{m}.npz")) for m in MODELS}

    results = {}
    for dom in DOMAINS:
        ids = data[f"{dom}_ids"]
        # masks are defined over the full window; scored targets are positions 1..L-1
        content = data[f"{dom}_content"][:, 1:]
        copy5 = data[f"{dom}_copy5"][:, 1:]
        nocopy = data[f"{dom}_nocopy"][:, 1:]
        allm = np.ones_like(copy5, dtype=bool)
        filters = {"ALL": allm, "STATE (content∩no-copy)": content & nocopy,
                   "COPY5": copy5}
        results[dom] = {}
        for fname, fmask in filters.items():
            results[dom][fname] = {}
            for m in MODELS:
                mean, lo, hi, n = filtered_stats(nll[m][dom], fmask)
                results[dom][fname][m] = {"mean": mean, "lo": lo, "hi": hi, "n": n}

        # paired per-position gaps vs transformer, first-half robustness
        half = np.zeros_like(allm)
        half[:, :1023] = True
        results[dom]["_first1024"] = {}
        for fname, fmask in filters.items():
            results[dom]["_first1024"][fname] = {}
            for m in MODELS:
                mean, lo, hi, n = filtered_stats(nll[m][dom], fmask & half)
                results[dom]["_first1024"][fname][m] = {"mean": mean, "lo": lo,
                                                        "hi": hi, "n": n}

    # copy-n gap curves, prose pooled (pg19+wiki): Δ = recurrent − transformer
    gap_curves = {m: [] for m in MODELS[1:]}
    ns = list(range(1, 9))
    for n in ns:
        masks, gaps = {}, {m: [] for m in MODELS[1:]}
        for dom in ["pg19", "wiki"]:
            ids = data[f"{dom}_ids"]
            cm = np.stack([copy_n_mask(row.tolist(), n) for row in ids])[:, 1:]
            for m in MODELS[1:]:
                d = (nll[m][dom] - nll["pythia"][dom])
                gaps[m].append((d, cm))
        for m in MODELS[1:]:
            dall = np.concatenate([g[0] for g in gaps[m]])
            call = np.concatenate([g[1] for g in gaps[m]])
            mean, lo, hi, cnt = filtered_stats(dall, call)
            gap_curves[m].append({"n": n, "mean": mean, "lo": lo, "hi": hi,
                                  "count": cnt})

    with open("results.json", "w") as f:
        json.dump({"filtered": results, "copy_n_gap_prose": gap_curves}, f, indent=1)

    # markdown table
    lines = ["| Domain | Filter | " + " | ".join(LABELS[m] for m in MODELS) + " |",
             "|---|---|---|---|---|"]
    for dom in DOMAINS:
        for fname in ["ALL", "STATE (content∩no-copy)", "COPY5"]:
            row = [DOM_LABEL[dom], fname]
            for m in MODELS:
                r = results[dom][fname][m]
                row.append(f"{r['mean']:.3f} [{r['lo']:.3f},{r['hi']:.3f}]")
            lines.append("| " + " | ".join(row) + " |")
    with open("results.md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    make_figure(results, gap_curves, ns)


def make_figure(results, gap_curves, ns):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    SURFACE = "#fcfcfb"
    INK = "#0b0b0b"
    MUTED = "#898781"
    GRID = "#e1e0d9"
    BASE = "#c3c2b7"
    COLORS = {"pythia": "#2a78d6", "mamba": "#1baf7a", "rwkv": "#eda100"}

    plt.rcParams.update({
        "font.family": "sans-serif", "text.color": INK,
        "axes.edgecolor": BASE, "axes.labelcolor": "#52514e",
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.facecolor": SURFACE, "figure.facecolor": SURFACE,
        "font.size": 10,
    })
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.2))
    fig.subplots_adjust(left=0.05, right=0.99, top=0.76, bottom=0.13, wspace=0.28)

    filter_names = ["ALL", "STATE (content∩no-copy)", "COPY5"]
    titles = ["All tokens", "Content ∩ No-Copy (state)", "Copy-5 only (recall)"]
    x = np.arange(len(DOMAINS))
    off = {"pythia": -0.18, "mamba": 0.0, "rwkv": 0.18}
    for ax, fname, title in zip(axes[:3], filter_names, titles):
        for m in MODELS:
            means = [results[d][fname][m]["mean"] for d in DOMAINS]
            los = [results[d][fname][m]["lo"] for d in DOMAINS]
            his = [results[d][fname][m]["hi"] for d in DOMAINS]
            ax.errorbar(x + off[m], means,
                        yerr=[np.subtract(means, los), np.subtract(his, means)],
                        fmt="o", color=COLORS[m], markersize=6.5, capsize=0,
                        elinewidth=1.6, label=LABELS[m])
        ax.set_title(title, fontsize=11, color=INK, pad=8)
        ax.set_xticks(x, [DOM_LABEL[d].split(" ")[0] for d in DOMAINS])
        ax.set_ylabel("loss (nats / token)" if fname == "ALL" else "")
        ax.grid(axis="y", color=GRID, linewidth=0.7)
        ax.set_axisbelow(True)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)

    ax = axes[3]
    ax.axhline(0, color=BASE, linewidth=1)
    for m in MODELS[1:]:
        c = gap_curves[m]
        means = [p["mean"] for p in c]
        los = [p["lo"] for p in c]
        his = [p["hi"] for p in c]
        ax.plot(ns, means, "-o", color=COLORS[m], linewidth=2, markersize=5.5,
                label=LABELS[m])
        ax.fill_between(ns, los, his, color=COLORS[m], alpha=0.15, linewidth=0)
        ax.annotate({"mamba": "Mamba", "rwkv": "RWKV"}[m], (ns[-1], means[-1]),
                    xytext=(6, 0), textcoords="offset points",
                    color=COLORS[m], fontsize=9, va="center")
    ax.set_title("Gap vs transformer on copy-n tokens\n(prose, + = recurrent worse)",
                 fontsize=10.5, color=INK, pad=4)
    ax.set_xlabel("repeated n-gram length")
    ax.set_ylabel("Δ NLL (nats)")
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.set_xlim(0.5, 10.2)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.92), fontsize=10)
    fig.suptitle("Filtered token losses on open Pile models "
                 "(replication of arXiv:2606.20936 §6)",
                 y=0.985, fontsize=12, color=INK)
    fig.savefig("fig7_repro.png", dpi=160)
    print("saved fig7_repro.png")


if __name__ == "__main__":
    main()
