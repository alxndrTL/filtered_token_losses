"""ftl — filtered token losses: measurement-only recall/state diagnostics
for LLM architecture research (arXiv:2606.20936 + extensions).

Integration boundary: your codebase produces per-token NLL arrays (W, L-1)
on a frozen eval pack; ftl turns them into capability-resolved metrics.
Nothing in this package loads or runs a model.

Typical use:
    pack = ftl.load_pack("evalpack_v1.npz")        # frozen per campaign
    nll = your_eval_fn(pack["ids"])                # (W, L-1) float
    metrics = ftl.report(nll, pack, ref_nll=baseline_nll)
    logger.log(metrics)                            # flat dict of floats
"""

from .masks import (copy_distance, copy_n_mask, first_occurrence_index,
                    nocopy_mask)
from .metrics import (bootstrap_ci, failure_rate, filtered_mean,
                      hard_copy_mask, paired_gap, recall_cliff,
                      recall_efficiency, report)
from .pack import build_pack, load_pack, save_pack
from .probe import build_probe, summarize as summarize_probe

__all__ = [
    "copy_n_mask", "nocopy_mask", "first_occurrence_index", "copy_distance",
    "filtered_mean", "bootstrap_ci", "paired_gap", "recall_cliff",
    "hard_copy_mask", "recall_efficiency", "failure_rate", "report",
    "build_pack", "save_pack", "load_pack",
    "build_probe", "summarize_probe",
]
