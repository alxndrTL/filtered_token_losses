"""Figure for the two enhanced diagnostics: recall-vs-distance and planted-repeat probe."""

import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, MUTED = "#fcfcfb", "#0b0b0b", "#898781"
GRID, BASE = "#e1e0d9", "#c3c2b7"
COLORS = {"pythia": "#2a78d6", "mamba": "#1baf7a", "rwkv": "#eda100"}
LABELS = {"pythia": "Pythia-410m (Transformer)", "mamba": "Mamba-370m (SSM)",
          "rwkv": "RWKV-4-430m (RNN)"}
SHORT = {"pythia": "Pythia", "mamba": "Mamba", "rwkv": "RWKV"}

plt.rcParams.update({
    "font.family": "sans-serif", "text.color": INK,
    "axes.edgecolor": BASE, "axes.labelcolor": "#52514e",
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.facecolor": SURFACE, "figure.facecolor": SURFACE, "font.size": 10,
})

dist = json.load(open("distance_profile.json"))["all"]
probe = json.load(open("planted_probe.json"))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
fig.subplots_adjust(left=0.07, right=0.98, top=0.70, bottom=0.13, wspace=0.30)

# Panel 1: recall cliff — paired Δ on copy-5 tokens vs distance to previous occurrence
bin_labels = [p["bin"] for p in dist["mamba"]]
x = np.arange(len(bin_labels))
ax1.axhline(0, color=BASE, linewidth=1)
for m in ["mamba", "rwkv"]:
    means = [p["mean"] for p in dist[m]]
    los = [p["lo"] for p in dist[m]]
    his = [p["hi"] for p in dist[m]]
    ax1.plot(x, means, "-o", color=COLORS[m], linewidth=2, markersize=6)
    ax1.fill_between(x, los, his, color=COLORS[m], alpha=0.15, linewidth=0)
    ax1.annotate(SHORT[m], (x[-1], means[-1]), xytext=(8, 0),
                 textcoords="offset points", color=COLORS[m], va="center")
ax1.set_xticks(x, bin_labels)
ax1.set_xlim(-0.3, 3.75)
ax1.set_xlabel("distance to previous occurrence (tokens)")
ax1.set_ylabel("Δ NLL vs transformer (nats)")
ax1.set_title("Recall cliff: copy-5 gap by distance\n(all domains, + = recurrent worse)",
              fontsize=11, color=INK)
ax1.grid(axis="y", color=GRID, linewidth=0.7)
ax1.set_axisbelow(True)
for s in ["top", "right"]:
    ax1.spines[s].set_visible(False)

# Panel 2: planted-repeat probe — memorization-free retrieval gain vs gap
gaps = [64, 256, 1024, 1792]
for m in ["pythia", "mamba", "rwkv"]:
    gains = [probe[m][str(g)]["gain"] for g in gaps]
    ax2.plot(gaps, gains, "-o", color=COLORS[m], linewidth=2, markersize=6,
             label=LABELS[m])
    ax2.annotate(SHORT[m], (gaps[-1], gains[-1]), xytext=(8, 0),
                 textcoords="offset points", color=COLORS[m], va="center")
ax2.axhline(0, color=BASE, linewidth=1)
ax2.set_xscale("log", base=2)
ax2.set_xticks(gaps, [str(g) for g in gaps])
ax2.set_xlim(48, 3400)
ax2.set_xlabel("gap between planted spans (tokens)")
ax2.set_ylabel("retrieval gain (nats)")
ax2.set_title("Planted-repeat probe: in-context retrieval\n(random-token windows, no memorization)",
              fontsize=11, color=INK)
ax2.grid(axis="y", color=GRID, linewidth=0.7)
ax2.set_axisbelow(True)
for s in ["top", "right"]:
    ax2.spines[s].set_visible(False)

handles, labels = ax2.get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
           bbox_to_anchor=(0.5, 0.91), fontsize=9.5)
fig.suptitle("Enhanced recall diagnostics for architecture search", y=0.985,
             fontsize=12, color=INK)
fig.savefig("fig_enhancements.png", dpi=160)
print("saved fig_enhancements.png")
