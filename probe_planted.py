"""Planted-repeat probe: memorization-free in-context retrieval at controlled distance.

Windows are iid tokens drawn from the PG-19 empirical unigram distribution (so
vocab/frequencies look natural but there is no syntax and no memorizable text).
One 32-token span is planted twice, `gap` tokens apart. NLL on the second
occurrence is achievable only via in-context retrieval (induction); NLL on the
first occurrence gives the no-information baseline. Retrieval gain = first − second.
"""

import json
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
from score import MODELS, load_model, nll_batch  # noqa: E402

L = 2048
SPAN = 32
GAPS = [64, 256, 1024, 1792]
PER_GAP = 6
RNG = np.random.RandomState(1)


def build_windows():
    data = np.load("eval_data.npz")
    pool = data["pg19_ids"].ravel()
    windows, spans = [], []          # spans: (first_start, second_start)
    for gap in GAPS:
        for _ in range(PER_GAP):
            w = pool[RNG.randint(0, len(pool), size=L)].copy()
            span = pool[RNG.randint(0, len(pool), size=SPAN)]
            p2 = L - SPAN - 8
            p1 = p2 - gap
            assert p1 >= 8
            w[p1:p1 + SPAN] = span
            w[p2:p2 + SPAN] = span
            windows.append(w)
            spans.append((gap, p1, p2))
    return np.stack(windows).astype(np.int64), spans


def main():
    ids, spans = build_windows()
    results = {}
    for key in MODELS:
        model = load_model(key)
        nlls = []
        for s in range(0, ids.shape[0], 2):
            batch = torch.from_numpy(ids[s:s + 2]).cuda()
            nlls.append(nll_batch(model, batch))
        nll = torch.cat(nlls).numpy()          # (W, L-1), target position i -> nll[i-1]
        del model
        torch.cuda.empty_cache()
        per_gap = {g: {"first": [], "second": []} for g in GAPS}
        for w, (gap, p1, p2) in enumerate(spans):
            # skip the span's first token: it is unpredictable in both occurrences
            per_gap[gap]["first"].append(nll[w, p1:p1 + SPAN - 1].mean())
            per_gap[gap]["second"].append(nll[w, p2:p2 + SPAN - 1].mean())
        results[key] = {
            str(g): {
                "first": float(np.mean(v["first"])),
                "second": float(np.mean(v["second"])),
                "second_std": float(np.std(v["second"])),
                "gain": float(np.mean(v["first"]) - np.mean(v["second"])),
            } for g, v in per_gap.items()
        }
        print(key, json.dumps(results[key], indent=None), flush=True)
    with open("planted_probe.json", "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
