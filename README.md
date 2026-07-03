# Filtered token losses — replication of arXiv:2606.20936 §6 on open models

Li & Merrill ("Comparing Transformers and Hybrid Models at the Token Level",
arXiv:2606.20936) propose *filtered token losses*: measurement-only sub-losses
computed from the same per-token NLL as standard validation, sliced by token
category, to surface capability differences that aggregate loss averages away.
Their Figure 7 shows this on 1B dev runs (Transformer / Hybrid / Pure RNN).

This repo replicates the diagnostic on independent open models that share the
GPT-NeoX tokenizer and Pile pretraining data (losses are therefore paired
per position, same prefix / same target):

- **Pythia-410m** — transformer
- **Mamba-370m** — pure recurrent (SSM)
- **RWKV-4-430m-pile** — pure RNN

Data: PG-19 test books (48 windows), Wikipedia (16), Python source (32),
packed into 2048-token windows.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
.venv/bin/pip install -r requirements.txt
.venv/bin/python -c "import nltk; nltk.download('averaged_perceptron_tagger_eng')"
```

Reproduce: `build_data.py` / `build_climbmix.py` → `score.py <model> [--data
--domains --out]` per model → `analyze.py` / `analyze_climbmix.py` /
`distance_profile.py` / `probe_planted.py`. The committed `*.npz` archives
(eval packs + per-token NLLs, ~4 MB total) let every analysis re-run without
any model forward passes. See `RECAP.md` for the campaign field guide.

## Filters (all computed from one forward pass)

| Filter | Definition | Probes |
|---|---|---|
| ALL | every scored position | standard validation loss |
| STATE | content words ∩ no-copy (first in-window occurrence of the token type) | state-conditioned readout |
| COPY5 | position completes a repeated 5-gram from the visible prefix | visible-prefix retrieval / recall |

## Result (mean NLL in nats, 95% cluster bootstrap CI over windows)

See `results.md` / `results.json` / `fig7_repro.png`. Headline:

- **Aggregate loss hides opposing regimes.** On Python, Pythia and Mamba are
  identical on ALL (1.170 vs 1.169), but Mamba is 0.13 nats *better* on STATE
  and 0.09 nats *worse* on COPY5.
- **Copy-5 exposes recurrent recall weakness**, exactly as in the paper's
  Fig. 7 (they report Pure RNN 0.10–0.20 nats worse): Mamba +0.09 to +0.12
  nats vs Pythia on Wikipedia/Python; RWKV-4 +0.12 (PG-19), +0.50 (Python),
  +1.06 (Wikipedia). Holds when restricted to positions <1024 (RWKV's
  training context), so it is not a length artifact.
- **STATE amplifies the recurrent advantage** (Mamba −0.13 to −0.22 nats vs
  Pythia across domains, ~2x its aggregate advantage).
- **Copy-n gap curve** (prose): Mamba's paired gap vs the transformer moves
  monotonically from −0.05 nats at n=1 (recurrent favored on ordinary reuse)
  to +0.09 at n=8 (worse the more the target is pure retrieval); RWKV climbs
  +0.10 → +0.60.

## Files

- `build_data.py` — packs eval windows, computes copy masks + POS/code tags
- `score.py <model>` — per-token NLL for one model (fp32, works on a 1080 Ti;
  recurrent models use the slow sequential path, ~4-5 s/window)
- `analyze.py` — filtered means, bootstrap CIs, copy-n gap curves, figure
- `eval_data.npz`, `nll_*.npz`, `results.{json,md}`, `fig7_repro.png`

## ClimbMix (held-out 2025 data, `analyze_climbmix.py`)

48 windows packed from NVIDIA ClimbMix (OptimalScale text mirror, docs
interleaved across part files to cover 8 mixture clusters; parts are sorted by
cluster so naive streaming samples a single topic). None of the three models
saw this data in training. Results (`climbmix_results.json`):

| Filter | Pythia | Mamba | RWKV |
|---|---|---|---|
| ALL | 2.805 | 2.720 | 2.855 |
| STATE | 4.437 | 4.281 | 4.410 |
| COPY5 | 0.335 | 0.446 | 0.902 |

Same split as the Pile domains but sharper: Mamba's COPY5 penalty vs Pythia is
+0.111 nats (vs +0.01-0.12 on Pile-adjacent text) while still winning ALL
(−0.085) and STATE (−0.156); RWKV's COPY5 penalty is +0.567. Recall cliff at
distance 512-2048: Mamba +0.338, RWKV +1.222 — both larger than on
Pile-adjacent domains, consistent with parametric memory partially masking
retrieval deficits on eval text that resembles training data. Held-out-mixture
data is therefore the *better* substrate for this diagnostic.

## Enhanced diagnostics (`fig_enhancements.png`)

- **Recall cliff** (`distance_profile.py`): copy-5 paired gap stratified by
  distance to the previous occurrence, computed *retroactively* from the
  archived per-token NLLs (no model re-runs). Mamba: +0.01 → +0.21 nats from
  bin 1-32 to bin 512-2048; RWKV: +0.08 → +1.13. Localizes *where* recall
  fails, not just whether.
- **Planted-repeat probe** (`probe_planted.py`): a 32-token span planted twice
  at controlled gap inside random-token windows (PG-19 unigram frequencies) —
  the second occurrence is predictable only by in-context retrieval, so this is
  memorization-free. Retrieval gain (first-occurrence NLL − second-occurrence
  NLL): Pythia ~5.3-6.0 nats flat across gaps 64-1792; Mamba 2.1 → ~0.3 by gap
  256; RWKV ≤0.9 everywhere. Natural-text COPY5 understates mechanistic recall
  gaps by ~10x because natural repeats are also predictable from language
  statistics; run both.

## Using this in an architecture-discovery pipeline

Report `ALL`, `STATE`, `COPY5` for every candidate (negligible cost — same
forward pass as validation). Gate on the paired COPY5 gap vs a matched
transformer baseline: a candidate whose COPY5 loss exceeds the baseline's by
more than a tolerance (e.g. 0.05 nats) has a recall defect that ALL will not
show. A candidate that closes STATE but lags COPY5 has good state-tracking
but broken retrieval; the reverse means attention is doing all the work.
