"""How to wire ftl into a training/eval codebase — validated on archived data.

In your campaign the flow is:

    # once per campaign (freeze!):
    pack = ftl.build_pack(doc_iter, your_tokenizer, n_windows=300)
    ftl.save_pack("evalpack_v1.npz", pack)

    # at every eval checkpoint of every candidate:
    nll = per_token_nll(candidate_model, pack["ids"])   # your forward pass
    np.save(f"nll/{run_id}/{step}.npy", nll.astype(np.float16))  # archive!
    logger.log(ftl.report(nll, pack, ref_nll=baseline_nll))

    # occasionally (mechanism check):
    ids, spans = ftl.build_probe(pack["ids"])
    logger.log(ftl.summarize_probe(per_token_nll(candidate_model, ids), spans))

This script demonstrates report() on this repo's archived NLLs and checks the
numbers match the session's findings.
"""

import numpy as np

import ftl

# assemble a pack-shaped dict from the archived eval data (pg19 domain)
data = np.load("eval_data.npz")
pack = {"ids": data["pg19_ids"], "copy5": data["pg19_copy5"],
        "nocopy": data["pg19_nocopy"], "content": data["pg19_content"]}

nll_mamba = dict(np.load("nll_mamba.npz"))["pg19"]
nll_pythia = dict(np.load("nll_pythia.npz"))["pg19"]
nll_selector = dict(np.load("nll_pythia14.npz"))["pg19"]

metrics = ftl.report(nll_mamba, pack, ref_nll=nll_pythia,
                     selector_nll=nll_selector, prefix="mamba370/pg19/")
for k, v in sorted(metrics.items()):
    print(f"{k:42s} {v:8.4f}")

# consistency with the session's analyze.py numbers
assert abs(metrics["mamba370/pg19/nll/all"] - 2.795) < 0.01
assert abs(metrics["mamba370/pg19/nll/copy5"] - 0.251) < 0.01
print("\nconsistent with archived analysis ✓")
