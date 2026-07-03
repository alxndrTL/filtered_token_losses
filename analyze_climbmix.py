"""Filtered losses + recall cliff on ClimbMix, printed alongside Pile-domain numbers."""

import json

import numpy as np

from analyze import LABELS, MODELS, filtered_stats
from distance_profile import BINS, boot_mean, copy_distance


def main():
    data = np.load("eval_data_climbmix.npz")
    nll = {m: dict(np.load(f"nll_{m}_climbmix.npz"))["climbmix"] for m in MODELS}

    content = data["climbmix_content"][:, 1:]
    copy5 = data["climbmix_copy5"][:, 1:]
    nocopy = data["climbmix_nocopy"][:, 1:]
    filters = {"ALL": np.ones_like(copy5, dtype=bool),
               "STATE (content∩no-copy)": content & nocopy,
               "COPY5": copy5}

    results = {}
    lines = ["| Filter | " + " | ".join(LABELS[m] for m in MODELS) + " |",
             "|---|---|---|---|"]
    for fname, fmask in filters.items():
        results[fname] = {}
        row = [fname]
        for m in MODELS:
            mean, lo, hi, n = filtered_stats(nll[m], fmask)
            results[fname][m] = {"mean": mean, "lo": lo, "hi": hi, "n": n}
            row.append(f"{mean:.3f} [{lo:.3f},{hi:.3f}]")
        lines.append("| " + " | ".join(row) + " |")
    print("\n".join(lines))

    # recall cliff
    ids = data["climbmix_ids"]
    d = np.stack([copy_distance(row.tolist()) for row in ids])[:, 1:]
    results["cliff"] = {}
    for m in MODELS[1:]:
        g = nll[m] - nll["pythia"]
        results["cliff"][m] = []
        parts = []
        for lo_, hi_ in BINS:
            sel = (d >= lo_) & (d < hi_)
            counts = sel.sum(1).astype(float)
            vals = np.where(counts > 0, (g * sel).sum(1) / np.maximum(counts, 1), 0)
            mean = float((g * sel).sum() / sel.sum())
            ci = boot_mean(vals, counts)
            results["cliff"][m].append({"bin": f"{lo_}-{hi_}", "mean": mean,
                                        "lo": float(ci[0]), "hi": float(ci[1]),
                                        "count": int(sel.sum())})
            parts.append(f"{lo_}-{hi_}: {mean:+.3f}")
        print(f"cliff {m}: " + " | ".join(parts))

    with open("climbmix_results.json", "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
