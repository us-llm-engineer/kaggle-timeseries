import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src")); sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, pandas as pd, pytest
from opt14 import features as F, common as C
from synth import make_stock


def test_build_shapes_and_finiteness():
    book, trade, _ = make_stock(1)
    feat, seq = F.build_stock(book, trade, stock_id=3)
    assert len(feat) == 24 and feat.time_id.is_monotonic_increasing
    assert seq.shape == (24, C.N_BINS, F.N_CH) and seq.dtype == np.float16
    assert np.isfinite(seq.astype(np.float32)).all()
    core = ["rv_all", "rv_sub5", "pavg_iv2", "spread_m", "n_trade", "tcnt_w3"]
    assert np.isfinite(feat[core].to_numpy()).all()
    assert (feat.rv_w1 <= feat.rv_w2 + 1e-9).all() and (feat.rv_w5 <= feat.rv_w6 + 1e-9).all()   # cumulative windows monotone


def test_preavg_beats_raw_rv_under_noise():
    """Heavy iid noise (noise/signal variance 0.25): plain RV is biased far up; the corrected estimators are close to truth."""
    rng = np.random.default_rng(0)
    G, N, sig, noise = 400, 599, 1e-3, 5e-4
    x = np.cumsum(rng.normal(0, sig / np.sqrt(N), (G, N + 1)), axis=1)
    Y = x + rng.normal(0, noise, (G, N + 1))
    res = F.preavg(Y)
    rv = np.sqrt((np.diff(Y, axis=1) ** 2).sum(1))
    assert np.median(rv) / sig > 2.0                                    # raw RV biased up by more than 2x
    for k in ("iv1", "iv2"):
        est = np.sqrt(np.maximum(res[k], 0))
        assert abs(np.median(est) / sig - 1) < 0.25, k                  # corrected within 25% (median)
        assert abs(np.median(est) / sig - 1) < abs(np.median(rv) / sig - 1) / 3
    assert np.median(res["pav_raw"]) < np.median(rv)                    # pre-averaging alone already reduces the noise bias
    assert abs(np.median(res["sigu"]) / noise - 1) < 0.1                # noise level recovered


def test_preavg_unbiased_without_noise():
    rng = np.random.default_rng(1)
    Y = np.cumsum(rng.normal(0, 1e-3 / np.sqrt(599), (300, 600)), axis=1)
    iv1 = np.sqrt(np.maximum(F.preavg(Y)["iv1"], 0))
    assert abs(np.median(iv1) / 1e-3 - 1) < 0.1


def test_gaps_do_not_break_and_estimators_track_truth():
    for p in (0.9, 0.6, 0.35):
        book, trade, truth = make_stock(7, n_windows=60, p_obs=p, noise=5e-5)
        feat, _ = F.build_stock(book, trade, 1)
        t = np.array([truth[i] for i in feat.time_id])
        assert np.isfinite(feat[["rv_all", "pavt_iv2", "pavg_iv2", "rv_sub5"]].to_numpy()).mean() > 0.97
        assert np.corrcoef(np.log(feat.rv_sub5), np.log(t))[0, 1] > 0.7      # features carry the true volatility
        assert np.corrcoef(np.log(feat.pavt_raw), np.log(t))[0, 1] > 0.6


def test_cross_features_use_no_target():
    book, trade, _ = make_stock(2)
    feats = []
    for s in range(1, 5):
        f, _ = F.build_stock(*make_stock(10 + s)[:2], stock_id=s)
        feats.append(f)
    df = pd.concat(feats, ignore_index=True)
    a = df.copy(); a["target"] = np.random.default_rng(0).random(len(a))
    b = df.copy(); b["target"] = np.random.default_rng(1).random(len(b))
    ca, cb = F.add_cross(a), F.add_cross(b)
    cols = [c for c in ca.columns if c != "target"]
    pd.testing.assert_frame_equal(ca[cols], cb[cols])        # identical features whatever the target


def test_split_roles():
    t = np.arange(32768)
    r = C.split_role(t)
    assert (C.split_role(t) == r).all()
    frac = [(r == k).mean() for k in range(3)]
    assert abs(frac[0] - 0.70) < 0.01 and abs(frac[1] - 0.15) < 0.01 and abs(frac[2] - 0.15) < 0.01


def test_rmspe_and_scalar():
    y = np.array([1.0, 2.0, 4.0]); p = 1.2 * y
    assert C.rmspe(y, y) == 0 and abs(C.rmspe(y, p) - 0.2) < 1e-12
    c = C.best_scalar(y, p)
    assert abs(c - 1 / 1.2) < 1e-12 and C.rmspe(y, c * p) < 1e-12
