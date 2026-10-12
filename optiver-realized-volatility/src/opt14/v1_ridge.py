"""V1 - noise-robust realised-volatility estimators -> log-space ridge + stock effects + RMSPE-optimal scalar.
Two feature sets so the estimators' value is measured, not assumed: 'plain' (plain RV family) vs 'est' (plain + pre-averaging estimators)."""
import numpy as np, pandas as pd
from .common import rmspe, best_scalar
from .obs import History, log

PLAIN = ["rv_w1", "rv_w2", "rv_w3", "rv_w4", "rv_w5", "rv_w6", "rv_late300", "rv_late450", "rv_wap2", "bpv", "rq", "trade_rv", "rv_grid1"]
SUBS = ["rv_sub5", "rv_sub30"]
EST = ["pavg_raw", "pavg_iv1", "pavg_iv2", "pavg_iv3", "pavt_raw", "pavt_iv1", "pavt_iv2", "pavt_iv3", "pavt_sigu", "pavg_sigu"]
EXTRA = ["n_book", "n_trade", "spread_m", "depth1_m", "nz_ret_frac"]


def _design(df, cols, stocks):
    X = [np.log(np.clip(df[c].to_numpy(dtype=np.float64), 1e-6, None)) for c in cols if c not in ("spread_m", "nz_ret_frac")]
    X += [df[c].to_numpy(dtype=np.float64) * (1e3 if c == "spread_m" else 1.0) for c in cols if c in ("spread_m", "nz_ret_frac")]
    X = np.stack(X, axis=1)
    D = (df["stock_id"].to_numpy()[:, None] == stocks[None, :]).astype(np.float64)
    return np.concatenate([X, D, np.ones((len(df), 1))], axis=1)


def _ridge(X, y, alpha):
    p = X.shape[1]
    A = X.T @ X + alpha * np.eye(p); A[-1, -1] -= alpha          # intercept unpenalised
    return np.linalg.solve(A, X.T @ y)


def run(df, out_dir, name, cols, logf=None):
    fit, ch, rep = (df[df.role == r] for r in (0, 1, 2))
    stocks = np.sort(df.stock_id.unique())
    Xf, Xc, Xr = (_design(d, cols, stocks) for d in (fit, ch, rep))
    # standardise columns by fit statistics so one alpha is meaningful
    mu, sd = Xf.mean(0), Xf.std(0); mu[-1], sd[-1] = 0, 1; sd[sd < 1e-9] = 1
    Xf, Xc, Xr = ((X - mu) / sd for X in (Xf, Xc, Xr))
    yl = np.log(fit.target.to_numpy())
    hist = History(f"{out_dir}/{name}_history.json", {"variant": name, "n_fit": len(fit), "n_choose": len(ch), "n_report": len(rep), "features": len(cols)})
    best = None
    for alpha in (1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0):
        w = _ridge(Xf, yl, alpha)
        pf, pc = np.exp(Xf @ w), np.exp(Xc @ w)
        c = best_scalar(ch.target.to_numpy(), pc)                 # scalar fitted on the choose role
        row = {"alpha": alpha, "fit_rmspe": rmspe(fit.target, c * pf), "choose_rmspe": rmspe(ch.target, c * pc), "scalar": c,
               "report_rmspe_monitor_only": rmspe(rep.target, c * np.exp(Xr @ w))}
        hist.add(row); log(logf, f"{name} alpha={alpha:g} choose RMSPE {row['choose_rmspe']:.4f} fit {row['fit_rmspe']:.4f}")
        if best is None or row["choose_rmspe"] < best[0]: best = (row["choose_rmspe"], alpha, w, c)
    _, alpha, w, c = best
    out = {}
    for role, d, X in ((1, ch, Xc), (2, rep, Xr)):
        p = c * np.exp(X @ w)
        out[role] = pd.DataFrame({"stock_id": d.stock_id.to_numpy(), "time_id": d.time_id.to_numpy(), "target": d.target.to_numpy(), "pred": p, "role": role})
    pd.concat(out.values()).to_parquet(f"{out_dir}/{name}_predictions.parquet")
    res = {"variant": name, "alpha": alpha, "scalar": c, "choose_rmspe": rmspe(ch.target, out[1].pred), "report_rmspe": rmspe(rep.target, out[2].pred)}
    hist.meta.update(res); hist.flush()
    log(logf, f"{name} FINAL alpha={alpha:g} choose {res['choose_rmspe']:.4f} report {res['report_rmspe']:.4f}")
    return res


def run_all(df, out_dir, logf=None):
    base = ["rv_all"] if "rv_all" in df else []
    sets = {"v1_plain": ["rv_all"] + PLAIN + SUBS + EXTRA, "v1_est": ["rv_all"] + PLAIN + SUBS + EST + EXTRA}
    return [run(df, out_dir, n, cols, logf) for n, cols in sets.items()]
