"""Updated Fig-7-style replication figure: six models, four domains.

Color encodes architecture family; marker shape encodes size tier
(circle = 130-430m tier, diamond = 1.4b tier).
"""

import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import analyze_large as al
from analyze import filtered_stats
from build_data import copy_masks  # noqa: F401
from analyze import copy_n_mask

SURFACE, INK, MUTED = "#fcfcfb", "#0b0b0b", "#898781"
GRID, BASE = "#e1e0d9", "#c3c2b7"
FAM = {"transformer": "#2a78d6", "mamba1": "#1baf7a", "rwkv": "#eda100",
       "mamba2": "#4a3aa7"}

MODELS = [
    ("pythia",   "Pythia-410m",  FAM["transformer"], "o"),
    ("pythia14", "Pythia-1.4b",  FAM["transformer"], "D"),
    ("mamba",    "Mamba-370m",   FAM["mamba1"], "o"),
    ("mamba14",  "Mamba-1.4b",   FAM["mamba1"], "D"),
    ("mamba2",   "Mamba2-370m",  FAM["mamba2"], "o"),
    ("rwkv",     "RWKV-4-430m",  FAM["rwkv"], "o"),
]
DOMAINS = [("pg19", "PG-19"), ("wiki", "Wikipedia"), ("python", "Python"),
           ("climbmix", "ClimbMix")]
FILTERS = [("ALL", "All tokens"), ("STATE", "Content ∩ No-Copy (state)"),
           ("COPY5", "Copy-5 only (recall)")]

plt.rcParams.update({
    "font.family": "sans-serif", "text.color": INK,
    "axes.edgecolor": BASE, "axes.labelcolor": "#52514e",
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.facecolor": SURFACE, "figure.facecolor": SURFACE, "font.size": 10,
})

nll = {k: al.load_nll(k) for k, *_ in MODELS}
masks = {d: al.masks_for(d)[0] for d, _ in DOMAINS}

fig, axes = plt.subplots(1, 4, figsize=(15.5, 4.6))
fig.subplots_adjust(left=0.045, right=0.995, top=0.72, bottom=0.13, wspace=0.26)

x = np.arange(len(DOMAINS))
offs = np.linspace(-0.3, 0.3, len(MODELS))
for ax, (fkey, ftitle) in zip(axes[:3], FILTERS):
    for (key, label, color, marker), off in zip(MODELS, offs):
        means, los, his = [], [], []
        for d, _ in DOMAINS:
            mean, lo, hi, _ = filtered_stats(nll[key][d], masks[d][fkey])
            means.append(mean); los.append(lo); his.append(hi)
        ax.errorbar(x + off, means,
                    yerr=[np.subtract(means, los), np.subtract(his, means)],
                    fmt=marker, color=color, markersize=5.5, capsize=0,
                    elinewidth=1.4, label=label)
    ax.set_title(ftitle, fontsize=11, color=INK, pad=8)
    ax.set_xticks(x, [n for _, n in DOMAINS])
    ax.set_ylabel("loss (nats / token)" if fkey == "ALL" else "")
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)

# panel 4: copy-n paired gap vs matched transformer, prose pooled
ax = axes[3]
data = np.load("eval_data.npz")
ns = list(range(1, 9))
cmasks = {}
for dom in ["pg19", "wiki"]:
    ids = data[f"{dom}_ids"]
    cmasks[dom] = {n: np.stack([copy_n_mask(r.tolist(), n) for r in ids])[:, 1:]
                   for n in ns}
gap_series = [("mamba", "pythia", FAM["mamba1"], "-", "Mamba-370m"),
              ("mamba14", "pythia14", FAM["mamba1"], "--", "Mamba-1.4b"),
              ("mamba2", "pythia", FAM["mamba2"], "-", "Mamba2-370m"),
              ("rwkv", "pythia", FAM["rwkv"], "-", "RWKV-4-430m")]
ax.axhline(0, color=BASE, linewidth=1)
for cand, anchor, color, ls, label in gap_series:
    ys = []
    for n in ns:
        num = den = 0.0
        for dom in ["pg19", "wiki"]:
            g = nll[cand][dom] - nll[anchor][dom]
            sel = cmasks[dom][n]
            num += (g * sel).sum(); den += sel.sum()
        ys.append(num / den)
    ax.plot(ns, ys, ls, marker="o", color=color, linewidth=2, markersize=5)
    ax.annotate(label.split("-")[0] + ("-1.4b" if "1.4b" in label else ""),
                (ns[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
                color=color, fontsize=8.5, va="center")
ax.set_title("Gap vs matched transformer on copy-n\n(prose, + = worse recall)",
             fontsize=10.5, color=INK, pad=4)
ax.set_xlabel("repeated n-gram length")
ax.set_ylabel("Δ NLL (nats)")
ax.set_xlim(0.5, 10.8)
ax.grid(axis="y", color=GRID, linewidth=0.7)
ax.set_axisbelow(True)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=6, frameon=False,
           bbox_to_anchor=(0.5, 0.90), fontsize=9.5)
fig.suptitle("Filtered token losses on open Pile models — six models, four domains "
             "(arXiv:2606.20936 §6 replication)", y=0.985, fontsize=12, color=INK)
fig.savefig("fig7_repro.png", dpi=160)
print("saved fig7_repro.png")
