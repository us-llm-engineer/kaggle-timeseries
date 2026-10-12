"""No-leak tests with mutants (synthetic data, seconds). Run: python3 tests/test_smoke.py"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from enefit13 import data as D, nn as N


def toy(days=20, S=3, seed=0):
    rng = np.random.default_rng(seed)
    T = days * 24
    t0 = pd.Timestamp("2022-01-01")
    rows = []
    for u in range(S):
        for c in (0, 1):
            for i in range(T):
                rows.append((u, c, t0 + pd.Timedelta(hours=i), float(rng.random() * 100), 10.0 + u))
    p = pd.DataFrame(rows, columns=["prediction_unit_id", "is_consumption", "datetime", "y", "scale"])
    p["y_norm"] = p.y / p.scale
    return p


def test_lags_use_only_48h_or_older():
    p = toy()
    an, ar, s, ti, sid, t0 = D.series_grid(p)
    base = D.add_lag_features(p.copy(), an, s, ti)
    mut = an.copy()
    day0 = 10 * 24                         # target day D starts here; everything from D-1 00:00 on is unknown at forecast time
    cut = day0 + 5
    mut[:, day0 - 24:] = 1e6
    q = D.add_lag_features(p.copy(), mut, s, ti)
    row = (ti >= day0) & (ti < day0 + 24)
    for c in D.LAGCOLS:
        assert np.allclose(base.loc[row, c].values, q.loc[row, c].values, equal_nan=True), c
    assert np.allclose(base.loc[row, "lag48"].values, an[s[row], ti[row] - 48], equal_nan=True)


def test_mutant_detects_leak():
    p = toy()
    an, ar, s, ti, sid, t0 = D.series_grid(p)
    base = D.add_lag_features(p.copy(), an, s, ti)
    mut = an.copy(); day0 = 10 * 24; mut[:, day0 - 24:] = 1e6
    q = D.add_lag_features(p.copy(), mut, s, ti)
    row = (ti >= day0 + 48) & (ti < day0 + 72)   # two days later the corrupted day IS visible (lag48 of those rows)
    assert not np.allclose(base.loc[row, "lag48"].values, q.loc[row, "lag48"].values)


def test_dl_lookback_ends_at_d_minus_2():
    p = toy(days=30)
    an, ar, s, ti, sid, t0 = D.series_grid(p)
    ar2 = ar.copy()
    td = dict(an=an, ar=ar2, S=an.shape[0], T=an.shape[1], sc=np.ones_like(an), wx=np.zeros(an.shape + (2,), dtype="float32"))
    d = 12
    lb, tg, sc, wx = N.batch_arrays(td, np.array([0]), np.array([d]), "raw")
    ds = d * 24
    assert lb.shape[1] == N.L
    assert np.allclose(lb[0], ar[0, ds - 192: ds - 24])
    assert np.allclose(tg[0], ar[0, ds: ds + 24])
    # nothing from day D-1 or D leaks into the look-back (mutate them, look-back unchanged)
    ar3 = ar.copy(); ar3[:, ds - 24:] = 9e9
    td["ar"] = ar3
    lb2, _, _, _ = N.batch_arrays(td, np.array([0]), np.array([d]), "raw")
    assert np.allclose(lb2, lb)


def test_roles_disjoint():
    p = toy(days=5)
    p["datetime"] = pd.Timestamp("2023-01-01") + (p.datetime - p.datetime.min())
    m = D.roles(p)
    assert not (m["fit"] & m["choose"]).any() and not (m["choose"] & m["report"]).any()


def test_tide_output_depends_on_covariates_and_dropout():
    import torch
    m = N.TiDE(15, 8, "rel", drop=0.5)
    b = 4
    lb, wx, cal = torch.randn(b, N.L), torch.randn(b, 24, 15), torch.randn(b, 24, 6)
    sid, st, ex = torch.zeros(b, dtype=torch.long), torch.zeros(b, 4, dtype=torch.long), torch.zeros(b, 1)
    m.eval()
    a = m(lb, wx, cal, sid, st, ex)
    wx2 = wx.clone(); wx2[:, :, 8] += 5.0                      # perturb one weather channel: output must change
    assert (m(lb, wx2, cal, sid, st, ex) - a).abs().max() > 1e-4
    m.train()
    assert (m(lb, wx, cal, sid, st, ex) - m(lb, wx, cal, sid, st, ex)).abs().max() > 1e-6   # dropout active


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f(); print("ok", k)
