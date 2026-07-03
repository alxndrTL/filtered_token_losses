"""Score per-token NLL for each model on the packed windows in eval_data.npz.

All three models share the GPT-NeoX tokenizer and were pretrained on the Pile,
so per-position NLLs are directly comparable (paired), as in arXiv:2606.20936.

Usage: score.py <model_key> [--bench] [--data FILE --domains a,b --out FILE]
"""

import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

MODELS = {
    "pythia": "EleutherAI/pythia-410m",
    "mamba": "state-spaces/mamba-370m-hf",
    "rwkv": "RWKV/rwkv-4-430m-pile",
}
BATCH = {"pythia": 4, "mamba": 2, "rwkv": 2}
DOMAINS = ["pg19", "wiki", "python"]


def load_model(key):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(MODELS[key], torch_dtype=torch.float32)
    model.eval().cuda()
    return model


@torch.no_grad()
def nll_batch(model, ids):
    """ids: (B, L) int64 cuda. Returns (B, L-1) float32 cpu NLL of targets 1..L-1."""
    logits = model(input_ids=ids).logits.float()          # (B, L, V)
    logits = logits[:, :-1]
    targets = ids[:, 1:]
    B, Lm1, V = logits.shape
    out = torch.empty(B, Lm1, device=ids.device)
    step = 256                                            # chunk over time to cap memory
    for s in range(0, Lm1, step):
        lg = logits[:, s:s + step].reshape(-1, V)
        tg = targets[:, s:s + step].reshape(-1)
        out[:, s:s + step] = F.cross_entropy(lg, tg, reduction="none").view(B, -1)
    return out.cpu()


def argval(flag, default):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


def main():
    key = sys.argv[1]
    bench = "--bench" in sys.argv
    data_file = argval("--data", "eval_data.npz")
    domains = argval("--domains", ",".join(DOMAINS)).split(",")
    out_file = argval("--out", f"nll_{key}.npz")
    data = np.load(data_file)
    model = load_model(key)

    out = {}
    for dom in domains:
        ids_all = torch.from_numpy(data[f"{dom}_ids"]).long()
        if bench:
            ids_all = ids_all[:BATCH[key]]
        nlls = []
        t0 = time.time()
        for s in range(0, ids_all.shape[0], BATCH[key]):
            batch = ids_all[s:s + BATCH[key]].cuda()
            nlls.append(nll_batch(model, batch))
            done = s + batch.shape[0]
            rate = (time.time() - t0) / done
            print(f"[{key}/{dom}] {done}/{ids_all.shape[0]} windows, "
                  f"{rate:.1f}s/window", flush=True)
        out[dom] = torch.cat(nlls).numpy().astype(np.float32)
        print(f"[{key}/{dom}] mean NLL {out[dom].mean():.4f}", flush=True)
        if bench:
            return
    np.savez_compressed(out_file, **out)
    print(f"saved {out_file}")


if __name__ == "__main__":
    main()
