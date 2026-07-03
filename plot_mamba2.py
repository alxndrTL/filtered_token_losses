"""Figure: scale vs architecture on the recall diagnostics (Mamba2 & 1.4b tier).

Color encodes architecture family (fixed assignment), linestyle encodes size.
"""

import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analyze_large import BINS, copy_distance  # noqa: F401  (BINS reuse)

SURFACE, INK, MUTED = "#fcfcfb", "#0b0b0b", "#898781"
GRID, BASE = "#e1e0d9", "#c3c2b7"
COL = {"transformer": "#2a78d6", "mamba1": "#1baf7a", "mamba2": "#4a3aa7"}

plt.rcParams.update({
    "font.family": "sans-serif", "text.color": INK,
    "axes.edgecolor": BASE, "axes.labelcolor": "#52514e",
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.facecolor": SURFACE, "figure.facecolor": SURFACE, "font.size": 10,
})

# --- left panel data: paired copy-5 gap by distance vs matched anchor ---
import analyze_large as al

nll = {m: al.load_nll(m) for m in ["pythia", "pythia14", "mamba", "mamba14", "mamba2"]}
all_doms = ["pg19", "wiki", "python", "climbmix"]
fmasks, ids_all = {}, []
dists = np.concatenate([
    np.stack([copy_distance(r.tolist()) for r in al.masks_for(d)[1]])[:, 1:]
    for d in all_doms])

def cliff(cand, anchor):
    g = np.concatenate([nll[cand][d] for d in all_doms]) - \
        np.concatenate([nll[anchor][d] for d in all_doms])
    out = []
    for lo, hi in BINS:
        sel = (dists >= lo) & (dists < hi)
        out.append((g * sel).sum() / sel.sum())
    return out

series = [
    ("Mamba-370m",  cliff("mamba", "pythia"),    COL["mamba1"], "-"),
    ("Mamba-1.4b",  cliff("mamba14", "pythia14"), COL["mamba1"], "--"),
    ("Mamba2-370m", cliff("mamba2", "pythia"),   COL["mamba2"], "-"),
]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))
fig.subplots_adjust(left=0.07, right=0.98, top=0.70, bottom=0.13, wspace=0.30)

x = np.arange(len(BINS))
ax1.axhline(0, color=BASE, linewidth=1)
for name, ys, c, ls in series:
    ax1.plot(x, ys, ls, marker="o", color=c, linewidth=2, markersize=6, label=name)
    ax1.annotate(name, (x[-1], ys[-1]), xytext=(8, 0), textcoords="offset points",
                 color=c, fontsize=9, va="center")
ax1.set_xticks(x, [f"{lo}-{hi}" for lo, hi in BINS])
ax1.set_xlim(-0.3, 4.6)
ax1.set_xlabel("distance to previous occurrence (tokens)")
ax1.set_ylabel("Δ NLL vs matched transformer (nats)")
ax1.set_title("Recall cliff vs matched Pythia\n(all domains, + = worse recall)",
              fontsize=11, color=INK)
ax1.grid(axis="y", color=GRID, linewidth=0.7)
ax1.set_axisbelow(True)
for s in ["top", "right"]:
    ax1.spines[s].set_visible(False)

# --- right panel: planted-repeat retrieval gain ---
import glob
probes = {}
for f in glob.glob("planted_probe*.json"):
    probes.update(json.load(open(f)))
gaps = [64, 256, 1024, 1792]
pseries = [
    ("Pythia-410m",  "pythia",   COL["transformer"], "-"),
    ("Pythia-1.4b",  "pythia14", COL["transformer"], "--"),
    ("Mamba-370m",   "mamba",    COL["mamba1"], "-"),
    ("Mamba-1.4b",   "mamba14",  COL["mamba1"], "--"),
    ("Mamba2-370m",  "mamba2",   COL["mamba2"], "-"),
]
for name, key, c, ls in pseries:
    ys = [probes[key][str(g)]["gain"] for g in gaps]
    ax2.plot(gaps, ys, ls, marker="o", color=c, linewidth=2, markersize=6, label=name)
ax2.axhline(0, color=BASE, linewidth=1)
ax2.set_xscale("log", base=2)
ax2.set_xticks(gaps, [str(g) for g in gaps])
ax2.set_xlim(48, 2600)
ax2.set_xlabel("gap between planted spans (tokens)")
ax2.set_ylabel("retrieval gain (nats)")
ax2.set_title("Planted-repeat retrieval\n(memorization-free)", fontsize=11, color=INK)
ax2.grid(axis="y", color=GRID, linewidth=0.7)
ax2.set_axisbelow(True)
for s in ["top", "right"]:
    ax2.spines[s].set_visible(False)

handles, labels = ax2.get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False,
           bbox_to_anchor=(0.5, 0.91), fontsize=9)
fig.suptitle("Architecture (d_state 16→128) buys recall; parameter scale doesn't",
             y=0.985, fontsize=12, color=INK)
fig.savefig("fig_mamba2.png", dpi=160)
print("saved fig_mamba2.png")
