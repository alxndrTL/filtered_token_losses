"""Build packed eval windows from ClimbMix (OptimalScale text mirror).

ClimbMix docs are often short, so documents are packed contiguously with EOS
separators into 2048-token windows (as in the paper), instead of one-doc-per-
window. Content masks are computed per document and carried through packing.
Output: eval_data_climbmix.npz with the same key layout as eval_data.npz.
"""

import json

import numpy as np

from build_data import L, copy_masks, prose_content_charmask

N_WINDOWS = 48


def main():
    import datasets
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-410m")
    eos = tokenizer.eos_token_id

    # parts are sorted by cluster: interleave docs from spread-out parts so the
    # sample covers the mixture, not one topic cluster
    parts = [f"part_{i}.jsonl" for i in range(0, 100, 8)]
    def doc_iter():
        iters = [iter(datasets.load_dataset("OptimalScale/ClimbMix",
                                            data_files=p, split="train",
                                            streaming=True)) for p in parts]
        while iters:
            for it in list(iters):
                try:
                    yield next(it)
                except StopIteration:
                    iters.remove(it)

    stream_ids, stream_content = [], []
    ids_list, content_list, copy5_list, nocopy_list = [], [], [], []
    n_docs = 0
    clusters = {}
    for ex in doc_iter():
        text = ex["text"]
        if not text or len(text) < 200:
            continue
        n_docs += 1
        clusters[ex.get("cluster_id", -1)] = clusters.get(ex.get("cluster_id", -1), 0) + 1
        enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        cmask_char = prose_content_charmask(text)
        cm = [cmask_char[s:e].any() if e > s else False
              for s, e in enc["offset_mapping"]]
        stream_ids.extend(enc["input_ids"])
        stream_content.extend(cm)
        stream_ids.append(eos)
        stream_content.append(False)
        while len(stream_ids) >= L:
            w_ids = np.array(stream_ids[:L], dtype=np.int32)
            w_cm = np.array(stream_content[:L], dtype=bool)
            del stream_ids[:L], stream_content[:L]
            c5, nc = copy_masks(w_ids.tolist())
            ids_list.append(w_ids)
            content_list.append(w_cm)
            copy5_list.append(c5)
            nocopy_list.append(nc)
        if len(ids_list) >= N_WINDOWS:
            break
    ids = np.stack(ids_list[:N_WINDOWS])
    content = np.stack(content_list[:N_WINDOWS])
    copy5 = np.stack(copy5_list[:N_WINDOWS])
    nocopy = np.stack(nocopy_list[:N_WINDOWS])
    np.savez_compressed("eval_data_climbmix.npz",
                        climbmix_ids=ids, climbmix_content=content,
                        climbmix_copy5=copy5, climbmix_nocopy=nocopy)
    print(json.dumps({
        "windows": int(ids.shape[0]), "docs": n_docs,
        "clusters_seen": len(clusters),
        "copy5_frac": float(copy5[:, 1:].mean()),
        "nocopy_frac": float(nocopy[:, 1:].mean()),
        "content_nocopy_frac": float((content & nocopy)[:, 1:].mean()),
    }))


if __name__ == "__main__":
    main()
