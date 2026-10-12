"""CPU variants V1-V4 on a multi-core host (a 4-core cloud CPU sandbox in the reported run). Fitted on targets <= d_1885; choose role = origin 1885; report = origin 1913; submit = origin 1941.
V1 bottom-up LightGBM, squared loss.  V2 same features, Tweedie loss.  V3 V2 reconciled to a separate scaled aggregate-level model (store x department groups).
V4 lag-free day-level model per store, forecasts aligned to the independent total forecast of the aggregate model.
Run: python3 src/cpu_variants.py --cores 16 --budget-min 22 [--smoke]"""
import argparse, os, sys, time, json
import numpy as np, lightgbm as lgb
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m5lib import *

OUT = os.environ.get("M5_OUT", "results"); os.makedirs(f"{OUT}/preds", exist_ok=True); os.makedirs("logs", exist_ok=True)
CAT_B = ["wday", "month", "ev", "item", "dept", "cat", "state"]
PARAMS = dict(learning_rate=0.05, num_leaves=63, min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, max_bin=127, verbose=-1)
STATE = {}


def log(msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}"; print(line, flush=True); open("logs/cpu_variants.log", "a").write(line + "\n")


def init(prep):
    os.environ["M5_PREP"] = prep; globals()["_d"] = load_prep()


def store_job(args):
    """V1+V2 for one store: shared training cells, two objectives."""
    store, threads, origins, per, rounds, seed = args; d = _d
    rows = np.where(d.codes[3] == store)[0]; h = Hist(d.D[rows]); rng = np.random.default_rng(seed + store)
    r, t, hz = draw_rows(np.arange(len(rows)), origins, per, FIT_END, rng)
    Xtr, _ = features(h, d, r, t, hz, glob=rows); ytr = h.X[r, t + hz - 1]
    rv, tv, hv = full_rows(np.arange(len(rows)), ORIGIN["choose"]); Xv, _ = features(h, d, rv, tv, hv, glob=rows); yv = h.X[rv, tv + hv - 1]
    out = {}
    for name, obj in (("V1", dict(objective="regression")), ("V2", dict(objective="tweedie", tweedie_variance_power=1.1))):
        t0 = time.time()
        m = lgb.train({**PARAMS, **obj, "metric": "rmse", "num_threads": threads, "seed": seed}, lgb.Dataset(Xtr, ytr, categorical_feature=CAT_B),
                      rounds, valid_sets=[lgb.Dataset(Xv, yv, categorical_feature=CAT_B)], callbacks=[lgb.early_stopping(50, verbose=False)])
        preds = {}
        for role in ORIGIN:
            rr, tt, hh = full_rows(np.arange(len(rows)), ORIGIN[role]); X, _ = features(h, d, rr, tt, hh, glob=rows)
            preds[role] = np.clip(m.predict(X, num_iteration=m.best_iteration), 0, None).reshape(len(rows), HZ)
        out[name] = dict(rows=rows, preds=preds, best_iter=m.best_iteration, secs=time.time() - t0, train_rows=len(ytr), valid_rmse=float(m.best_score["valid_0"]["rmse"]))
    return store, out


def lagfree_job(args):
    """V4 base model for one store: no lag features; calendar, price, trend and ids only. Target = sales on day g."""
    store, threads, rounds, seed = args; d = _d
    rows = np.where(d.codes[3] == store)[0]; rng = np.random.default_rng(seed + store)
    days = np.arange(FIT_END - 728, FIT_END)
    def mk(dd):
        r = np.repeat(rows, len(dd)); g = np.tile(dd, len(rows)); f = {}
        f["wday"] = d.wday[g]; f["month"] = d.month[g]; f["dom"] = d.dom[g]; f["ev"] = np.where(d.ev_t[:, g].sum(0) > 0, d.ev_t[:, g].argmax(0) + 1, 0)
        f["snap"] = d.snap[d.snap_state[r], g]; f["trend"] = g / 1969.0; f["woy"] = d.doy[g] // 7
        f["price"] = d.Pf[r, g] / d.Pmax[r]; f["pr_ratio"] = d.Pf[r, g] / (d.Pf[r, FIT_END - 1] + 1e-3)
        for nm, ci in (("item", 0), ("dept", 1), ("cat", 2)): f[nm] = d.codes[ci][r]
        import pandas as pd; return pd.DataFrame(f).astype(np.float32), r, g
    Xtr, r, g = mk(days); keep = rng.random(len(r)) < 0.5; Xtr, ytr = Xtr[keep], d.D[r[keep], g[keep]].astype(np.float32)
    cats = ["wday", "month", "ev", "item", "dept", "cat"]
    Xv, rv, gv = mk(np.arange(FIT_END, FIT_END + HZ)); yv = d.D[rv, gv].astype(np.float32)
    t0 = time.time()
    m = lgb.train({**PARAMS, "objective": "regression", "metric": "rmse", "num_threads": threads, "seed": seed}, lgb.Dataset(Xtr, ytr, categorical_feature=cats), rounds,
                  valid_sets=[lgb.Dataset(Xv, yv, categorical_feature=cats)], callbacks=[lgb.early_stopping(50, verbose=False)])
    preds = {}
    for role in ORIGIN:
        X, _, _ = mk(np.arange(ORIGIN[role], ORIGIN[role] + HZ)); preds[role] = np.clip(m.predict(X, num_iteration=m.best_iteration), 0, None).reshape(len(rows), HZ)
    return store, dict(rows=rows, preds=preds, best_iter=m.best_iteration, secs=time.time() - t0, train_rows=len(ytr))


def agg_model(d, threads, origins, per, rounds, seed):
    """Separate scaled model for the 12,350 aggregate series (levels 1-11): target and lags divided by 1 + trailing mean, level id as a feature."""
    A = np.asarray((d.S[d.lvl > 0] @ d.D.astype(np.float64)))
    lv = d.lvl[d.lvl > 0].astype(np.float32); h = Hist(A); rng = np.random.default_rng(seed)
    r, t, hz = draw_rows(np.arange(len(A)), origins, per, FIT_END, rng)
    X, s = features(h, d, r, t, hz, scale=True, lvl=lv); y = h.X[r, t + hz - 1] / s
    rv, tv, hv = full_rows(np.arange(len(A)), ORIGIN["choose"]); Xv, sv = features(h, d, rv, tv, hv, scale=True, lvl=lv); yv = h.X[rv, tv + hv - 1] / sv
    cats = ["wday", "month", "ev", "lvl"]
    m = lgb.train({**PARAMS, "objective": "regression", "metric": "rmse", "num_threads": threads, "seed": seed}, lgb.Dataset(X, y, categorical_feature=cats), rounds,
                  valid_sets=[lgb.Dataset(Xv, yv, categorical_feature=cats)], callbacks=[lgb.early_stopping(50, verbose=False)])
    preds = {}
    for role in ORIGIN:
        rr, tt, hh = full_rows(np.arange(len(A)), ORIGIN[role]); Xr, sr = features(h, d, rr, tt, hh, scale=True, lvl=lv)
        preds[role] = (np.clip(m.predict(Xr, num_iteration=m.best_iteration), 0, None) * sr).reshape(len(A), HZ)
    return preds, int(m.best_iteration), len(y)


def assemble(d, res, key):
    out = {role: np.zeros((d.n, HZ)) for role in ORIGIN}
    for _, o in res:
        for role in ORIGIN: out[role][o[key]["rows"] if key else o["rows"]] = (o[key] if key else o)["preds"][role]
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cores", type=int, default=16); ap.add_argument("--budget-min", type=float, default=22); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(); T0 = time.time(); d = load_prep(); log(f"prep loaded series={d.n} days={d.T} cores={a.cores} smoke={a.smoke}")
    stores = [0, 1] if a.smoke else list(range(10)); rounds = 20 if a.smoke else 400
    origins = np.arange(1129, 1857, 28 if a.smoke else 14); per = 1 if a.smoke else 2
    workers = max(1, min(len(stores), a.cores // 2)); thr = max(1, a.cores // workers)
    scorers = {r: Scorer(d, r) for r in ("choose", "report")}; hist = {"started": time.strftime("%FT%TZ"), "variants": {}}
    def finish(name, pred_by_role, extra=None):
        rec = {"choose": scorers["choose"].score(pred_by_role["choose"]), "report": scorers["report"].score(pred_by_role["report"], per_level=True)}
        rec["report_per_level"] = rec["report"][1].round(3).tolist(); rec["report"] = rec["report"][0]; rec.update(extra or {}); rec["elapsed_s"] = time.time() - T0
        hist["variants"][name] = rec; atomic_json(f"{OUT}/cpu_history.json", hist)
        for role in ORIGIN: np.save(f"{OUT}/preds/{name}_{role}.npy", pred_by_role[role].astype(np.float32))
        log(f"{name}: choose WRMSSE {rec['choose']:.4f} | report WRMSSE {rec['report']:.4f} | per level {rec['report_per_level']} | elapsed {rec['elapsed_s']:.0f}s")
    with ProcessPoolExecutor(workers, initializer=init, initargs=(PREP,)) as ex:
        res = list(ex.map(store_job, [(s, thr, origins, per, rounds, 1) for s in stores])); log(f"V1/V2 trained: {[(s, o['V1']['best_iter'], o['V2']['best_iter'], round(o['V1']['secs']), round(o['V2']['secs'])) for s, o in res]}")
        if time.time() - T0 > a.budget_min * 60: log("budget exhausted after V1/V2"); 
        fl = list(ex.map(lagfree_job, [(s, thr, rounds, 1) for s in stores])); log(f"V4 base trained in {time.time() - T0:.0f}s")
    v1 = assemble(d, res, "V1"); v2 = assemble(d, res, "V2"); v4 = assemble(d, fl, None)
    ao, ap_ = (np.arange(1129, 1857, 28), 1) if a.smoke else (np.arange(1129, 1857, 28), 4)
    agg, bi, nr = agg_model(d, a.cores, ao, ap_, rounds, 1); log(f"aggregate model trained: best_iter {bi}, rows {nr}, elapsed {time.time() - T0:.0f}s")
    lvl_rows = d.S[d.lvl > 0]; Sfull = d.S
    if True:  # smoke mode fits 2 stores only, so its scores are meaningless; it exercises the code path and timing
        finish("V1_lgb_squared", v1); finish("V2_lgb_tweedie", v2)
        # V3: reconcile V2 to the aggregate model at store x department groups (level 8), alpha chosen on the choose role
        g8 = np.array(pd.factorize(d.meta.store_id + "|" + d.meta.dept_id)[0]); ref8 = {r: agg[r][(d.lvl[d.lvl > 0] == 8)] for r in ORIGIN}
        order8 = np.array([k[1] for k in np.array(d.keys, dtype=object)[d.lvl == 8]]); lab = pd.factorize(d.meta.store_id + "|" + d.meta.dept_id)[1]
        ref8 = {r: agg[r][np.where(d.lvl[d.lvl > 0] == 8)[0]][pd.Index(order8).get_indexer(lab)] for r in ORIGIN}
        best = min((0.0, 0.5, 1.0), key=lambda al: scorers["choose"].score(align(v2["choose"], g8, ref8["choose"], al))); log(f"V3 alpha (choose) = {best}")
        finish("V3_lgb_tweedie_reconciled", {r: align(v2[r], g8, ref8[r], best) for r in ORIGIN}, dict(alpha=best))
        tot = {r: agg_total(d, agg, r) for r in ORIGIN}
        best4 = min((0.0, 0.5, 1.0), key=lambda al: scorers["choose"].score(align(v4["choose"], np.zeros(d.n, int), tot["choose"], al)))
        finish("V4_lagfree_aligned", {r: align(v4[r], np.zeros(d.n, int), tot[r], best4) for r in ORIGIN}, dict(alpha=best4))
    np.save(f"{OUT}/preds/AGG_levels1-11.npy", np.stack([agg[r] for r in ORIGIN])); log(f"done {time.time() - T0:.0f}s")


def agg_total(d, agg, role):
    """Independent total forecast: the level-0 series is not in the aggregate model's rows (levels 1-11); use the sum over the 3 states (level 1)."""
    lv = d.lvl[d.lvl > 0]; return agg[role][lv == 1].sum(0, keepdims=True)


if __name__ == "__main__":
    main()
