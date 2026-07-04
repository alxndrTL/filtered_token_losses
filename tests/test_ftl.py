"""Correctness tests for the ftl package. Run: python tests/test_ftl.py"""

import sys
import pathlib

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import ftl


def test_copy_masks():
    ids = [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 9]
    c5 = ftl.copy_n_mask(ids, 5)
    assert list(np.where(c5)[0]) == [9], c5          # 5-gram completes at idx 9
    c2 = ftl.copy_n_mask(ids, 2)
    assert c2[6] and c2[9] and not c2[10]
    nc = ftl.nocopy_mask(ids)
    assert list(nc) == [True]*5 + [False]*5 + [True]

    fo = ftl.first_occurrence_index(ids, n=5)
    assert fo[9] == 4 and fo[4] == -1               # first occ of the 5-gram ends at 4
    d = ftl.copy_distance(ids, n=5)
    assert d[9] == 5


def test_metrics_alignment():
    # window: positions 0..9; nll[k] scores position k+1
    ids = np.array([[1, 2, 3, 4, 5, 1, 2, 3, 4, 5]])
    copy5 = np.stack([ftl.copy_n_mask(ids[0], 5)])
    nll = np.zeros((1, 9))
    nll[0, 8] = 3.0                                  # position 9 = the copy event
    assert ftl.filtered_mean(nll, copy5) == 3.0
    assert ftl.failure_rate(nll, copy5, threshold=1.0) == 1.0


def test_recall_efficiency_perfect_and_absent():
    # first occurrence expensive, repeat free -> efficiency 1
    ids = np.array([[1, 2, 3, 4, 5, 9, 1, 2, 3, 4, 5]])
    nll = np.ones((1, 10)) * 0.1
    nll[0, 3] = 5.0                                  # position 4: first occ gram end
    nll[0, 9] = 0.0                                  # position 10: repeat
    eff = ftl.recall_efficiency(nll, ids, tau=2.0)
    assert eff["overall"] == 1.0
    nll[0, 9] = 5.0                                  # no recovery -> efficiency 0
    eff = ftl.recall_efficiency(nll, ids, tau=2.0)
    assert eff["overall"] == 0.0


def test_paired_gap_sign():
    ids = np.array([[1, 2, 3, 4, 5, 1, 2, 3, 4, 5]])
    copy5 = np.stack([ftl.copy_n_mask(ids[0], 5)])
    a = np.zeros((1, 9)); b = np.zeros((1, 9))
    a[0, 8] = 2.0                                    # candidate worse on the copy event
    gap, lo, hi = ftl.paired_gap(a, b, copy5)
    assert gap == 2.0


def test_probe_roundtrip():
    pool = np.arange(100, 1100)
    ids, spans = ftl.build_probe(pool, window_len=256, span_len=8,
                                 gaps=(32, 64), per_gap=2, seed=0)
    assert ids.shape == (4, 256)
    for gap, p1, p2 in spans:
        assert p2 - p1 == gap
        assert (ids[0].dtype == np.int64)
    # spans really are planted twice
    for w, (gap, p1, p2) in enumerate(spans):
        assert (ids[w, p1:p1+8] == ids[w, p2:p2+8]).all()
    fake_nll = np.ones((4, 255))
    s = ftl.summarize_probe(fake_nll, spans, span_len=8)
    assert set(s) == {32, 64} and abs(s[32]["gain"]) < 1e-9


def test_hard_copy_mask():
    ids = np.array([[1, 2, 3, 4, 5, 9, 1, 2, 3, 4, 5]])
    sel = np.zeros((1, 10))
    sel[0, 3] = 5.0                                  # selector surprised at first occ
    hard = ftl.hard_copy_mask(ids, sel, tau=2.0)
    assert hard[0, 10] and hard.sum() == 1
    sel[0, 3] = 0.5                                  # easy first occ -> excluded
    hard = ftl.hard_copy_mask(ids, sel, tau=2.0)
    assert hard.sum() == 0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        f()
        print(f"ok  {f.__name__}")
    print(f"{len(fns)} tests passed")
