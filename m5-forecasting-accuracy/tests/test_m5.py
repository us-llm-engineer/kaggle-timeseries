"""Leak and correctness tests. Run: python3 -m pytest -q tests/test_m5.py (needs prep/m5_prep.npz)."""
import os, sys
import numpy as np, pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from m5lib import *

@pytest.fixture(scope="module")
def d(): return load_prep()

def test_features_do_not_see_the_future(d):
    rows = np.arange(0, 500); X = d.D[rows].copy(); t = np.full(len(rows), 1500); hz = np.full(len(rows), 10)
    f1, _ = features(Hist(X), d, np.arange(len(rows)), t, hz, glob=rows)
    X2 = X.copy(); X2[:, 1500:] = 777                       # mutant: corrupt every day at or after the origin
    f2, _ = features(Hist(X2), d, np.arange(len(rows)), t, hz, glob=rows)
    assert np.allclose(f1.to_numpy(), f2.to_numpy())

def test_train_cells_stop_at_fit_end():
    r, t, h = draw_rows(np.arange(5), np.arange(1129, 1885, 7), 3, FIT_END, np.random.default_rng(0))
    assert (t + h <= FIT_END).all()

def test_align_matches_group_sums():
    pred = np.random.default_rng(0).random((12, 28)) + 0.5; grp = np.repeat(np.arange(3), 4); ref = np.full((3, 28), 10.0)
    out = align(pred, grp, ref, 1.0, clip=(0.0, 100.0))
    assert np.allclose([out[grp == g].sum(0) for g in range(3)], 10.0)

def test_scorer_perfect_is_zero_and_matches_reference(d):
    s = Scorer(d, "choose"); assert abs(s.score(s.y)) < 1e-9
    W = WRMSSE(d.meta, d.D[:, :1885].astype(np.float64), dollars(d.P, d.D, 1858, 1885))
    pred = d.D[:, 1857:1885].astype(np.float64)
    assert abs(W.score(s.y, pred) - s.score(pred)) < 1e-9
