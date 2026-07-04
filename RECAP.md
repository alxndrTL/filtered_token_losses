# Recall diagnostics for the hybrid-architecture campaign — recap

Everything learned from replicating and extending arXiv:2606.20936 (Li &
Merrill, "Comparing Transformers and Hybrid Models at the Token Level"),
July 3, 2026. Code and data: this directory. Companion figures:
`fig7_repro.png`, `fig_enhancements.png`.

## The core idea

The paper introduces **filtered token losses**: measurement-only sub-losses
sliced from the *same per-token NLL as standard validation* (zero extra
forwards, never in the training objective). Aggregate loss is low-signal for
architecture search because it averages over computation regimes that respond
*oppositely* to the sequence mixer. Three filters separate them:

| Filter | Definition | What it probes | Favors |
|---|---|---|---|
| ALL | every position | standard validation | — |
| STATE | content words ∩ first in-window occurrence of token type | state-conditioned readout | recurrence |
| COPY5 | completes a repeated 5-gram from the prefix | visible-prefix retrieval | attention |

## What we verified on open models (Pythia-410m / Mamba-370m / RWKV-4-430m, all Pile + NeoX tokenizer)

1. **Aggregate loss hides opposing regimes.** Python: Pythia and Mamba tie on
   ALL (1.170 vs 1.169 nats) while Mamba is 0.13 better on STATE and 0.09
   *worse* on COPY5. A single validation number can be a wash between two
   models with different capability profiles.
2. **COPY5 exposes recurrent recall weakness** at the paper's predicted
   magnitude (Mamba +0.09…+0.12 nats vs transformer; RWKV up to +1.06), and
   STATE roughly doubles the recurrent advantage vs the aggregate view.
3. **Recall failure is a *range* failure** (recall cliff, `distance_profile.py`):
   the paired copy-gap grows monotonically with distance to the previous
   occurrence — Mamba +0.01 → +0.21 nats (ClimbMix: → +0.34), RWKV → +1.13
   (ClimbMix: → +1.22) from bin 1-32 to bin 512-2048. The cliff's shape points
   at the design knob: cliff at your attention layers' effective range →
   placement/ratio problem; uniform offset → state capacity.
4. **Natural-text copy events understate mechanistic gaps ~10×** because they
   are partly predictable from language statistics and parametric memory.
   The planted-repeat probe (`probe_planted.py`: span planted twice in
   random-token windows → second occurrence predictable only via in-context
   retrieval) shows: transformer retrieval gain ~6 nats, flat to gap 1792;
   Mamba 2.1 → ~0.3 nats past gap 256; RWKV ≤0.9 everywhere.
5. **Held-out mixture data gives sharper signal than train-adjacent data**
   (parametric memory masks retrieval deficits on familiar text): Mamba's
   COPY5 gap is +0.01 on PG-19 but +0.11 on ClimbMix. Prefer ClimbMix-style
   held-out mixtures as the eval substrate.
6. **Architecture buys recall; parameter scale doesn't** (Pythia/Mamba 1.4b +
   Mamba2-370m runs): 4x params shrinks Mamba's paired COPY5 gap (+0.088 →
   +0.053) but the planted-repeat retrieval gain is *unchanged* (~0.3 nats past
   gap 256 at both sizes) — scale papers over the natural-text symptom via
   parametric knowledge without fixing the mechanism. Mamba2 (d_state 16→128,
   same size/data) cuts the gap to +0.027 and retains 2-4 nats of real
   retrieval to gap 1792. For the campaign: state size / mixer design is the
   recall lever; don't expect scale to rescue a recall-broken candidate.

7. **HARD-COPY5 removes the "usual English" contamination** (`hard_copy.py`):
   keep only copy-5 events whose *first occurrence* cost an external selector
   transformer > τ nats — if the token was surprising at first sight, its later
   predictability must come from retrieval, not language statistics. Real-data
   analogue of the planted probe's first-vs-second logic; computed retroactively
   from the NLL archives. At τ=2 (20% of events survive), paired gaps amplify
   ~2.5-3x: Mamba-370m +0.088→+0.226, RWKV +0.51→+1.40, Mamba2 +0.027→+0.070.
   Use a selector external to both compared models (cross-selection) to avoid
   selection bias. Residual caveat: a larger candidate can still beat the
   selector via parametric knowledge, so hard-copy5 reduces but cannot fully
   remove the LM-capability leak — the planted probe remains the pure-mechanism
   test.

8. **RECALL EFFICIENCY is the capability-invariant recall metric**
   (`recall_efficiency.py`): within-model ratio 1 − NLL(repeat)/NLL(first
   occurrence) over copy events whose first occurrence cost the model itself
   > 2 nats. LM capability cancels (it lowers both terms): Pythia 410m→1.4b
   moves it only 0.914→0.919 while raw COPY5 moves ~13%. Mechanism ordering:
   Pythia ~0.92 > Mamba2 0.90 > Mamba1 0.86-0.88 >> RWKV 0.63. Distance-
   stratified: all models ~0.87 at range 1-128; at 512-2048 transformers hit
   0.94, Mamba2 0.91, Mamba1 0.86-0.89, RWKV collapses to 0.53. Real data,
   no external selector, zero extra forwards. Campaign primary recall metric;
   planted probe stays as the pure-mechanism control.

## Campaign workflow

- **Freeze one versioned eval pack** (packed windows + masks) for the entire
  campaign; all numbers must come from it or they are not comparable.
  Match window length to your training context (we used 2048; paper used 8192).
- **Archive per-token NLLs (fp16) for every candidate.** ~2 bytes/token. Every
  future filter is then a retroactive query over all past candidates — the
  recall cliff cost zero model re-runs for exactly this reason.
- **Use 1: candidate vs baseline transformer.** Matched data/tokenizer/budget
  baseline; gate on paired COPY5 gap (+ threshold ⇒ recall defect ALL won't show).
- **Use 2: neighbor vs neighbor.** Pair the two candidates *directly* per
  position (Δ_AB = ℓ_A − ℓ_B) and bootstrap that (cluster bootstrap over
  windows), not the difference of gaps-to-baseline CIs.
- **Diagnosis matrix**: closes STATE but lags COPY5 → state-strong,
  retrieval-broken (add/move attention). Closes COPY5 but lags STATE →
  borrowing attention's strength without state-tracking gains.
- **Log filters at every eval checkpoint** (Fig. 7 style): filtered curves
  separate architectures much earlier than aggregate loss → prune doomed runs
  at ~10% budget. NLL is smooth in training; accuracy metrics are steppy.
- **Escalation ladder for ambiguous neighbor deltas**: flat COPY5 →
  distance-stratified cliff (long-range bin is ~3× the flat signal) →
  planted-repeat probe (effect sizes in whole nats; answers "did the retrieval
  mechanism change at all"). More seeds only after that.

## Statistical resolution (measured, ClimbMix, paired Δ Mamba−Pythia)

| Eval tokens | COPY5 CI half-width |
|---|---|
| 16k | ±0.160 |
| 98k | ±0.051 |
| ~600k (extrapolated, 1/√N) | ±0.02 |

ALL-tokens paired gap: ±0.007 at 98k tokens. Versus SWDE/FDA/NIAH: binary
accuracy on ≤ a few k prompted examples ⇒ ±1.5-3pt binomial noise plus
template/needle-position sensitivity, and zero signal while accuracy sits at
floor early in training. Filtered NLL replaces them for *search*; keep one
NIAH-style end-to-end check per finalist (it tests prompted retrieval at full
context length, which window-length COPY5 doesn't).

## Caveats & gotchas

- **COPY5 is heavy-tailed**: most copy events cost ~0 for every model; the mean
  is driven by rare hard retrievals. Don't skimp on eval tokens for this filter.
  (en gros, la plupart des 5-gram qu'on releve sont des expressions anglaises communes,
  seuls 10% des 5-gram sont des "vrais" 5-gram qui sont detectables via du rappel)
  
- cool, normalement bcp moins bruité que FDA/SWDE/NIAH, juste, comme d'hab, faire attention à l'inter-seed variance

- Planted-repeat inputs are OOD (random tokens) — that's the price of removing
  the memorization confound; it stayed cleanly discriminative.
- STATE here is a Penn-POS approximation of the paper's Brown-tag Top-10 filter;
  copy tags are proxies, not proof of a copying mechanism (paper's own caveat).

## Pointers

- Paper: arXiv:2606.20936. Their models: Olmo 3 7B vs Olmo Hybrid 7B
  (arXiv:2604.03444); 1B dev runs = Transformer / GDN-attention 3:1 Hybrid /
  pure GDN. Fig. 7 = the filtered-loss proof of concept.
- Related synthetic recall: MQAR (Arora et al., "Zoology", ICLR 2024) —
  complementary to the planted-repeat probe.
- Repo scripts: `build_data.py`, `build_climbmix.py`, `score.py` (`--data
  --domains --out`), `analyze.py`, `analyze_climbmix.py`,
  `distance_profile.py`, `probe_planted.py`, `plot_enhancements.py`.
