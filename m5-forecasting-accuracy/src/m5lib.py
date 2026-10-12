"""Shared M5 utilities: prepared arrays, roles, exact WRMSSE scorer, trailing-window features.
Roles (origin = number of known days): choose d_1886..1913 from origin 1885, report d_1914..1941 from origin 1913, submit d_1942..1969 from origin 1941.
Models are fitted only on targets up to d_1885; the same fitted model is rolled to later origins (no refit), so the report window is never used to fit or choose."""
import os, sys, types
import numpy as np, pandas as pd, scipy.sparse as sp

HZ = 28
ORIGIN = {"choose": 1885, "report": 1913, "submit": 1941}
FIT_END = 1885
PREP = os.environ.get("M5_PREP", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "prep"))
EVENT_TYPES = ["Sporting", "Cultural", "National", "Religious"]


def build_prep(data_dir=None):
    """Read the competition files (Kaggle layout: calendar.csv, sell_prices.csv, sales_train_evaluation.csv, sample_submission.csv) once and
    save compact arrays so every worker starts in seconds."""
    data_dir = data_dir or os.environ["M5_DATA_DIR"]
    cal = pd.read_csv(f"{data_dir}/calendar.csv"); pr = pd.read_csv(f"{data_dir}/sell_prices.csv")
    ev = pd.read_csv(f"{data_dir}/sales_train_evaluation.csv"); sub = pd.read_csv(f"{data_dir}/sample_submission.csv")
    meta = ev[["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]].astype(str).reset_index(drop=True)
    D = ev[[f"d_{i}" for i in range(1, 1942)]].to_numpy(np.int16)
    cal = cal.set_index("d").loc[[f"d_{i}" for i in range(1, 1970)]]
    piv = pr.set_index(["store_id", "item_id", "wm_yr_wk"])["sell_price"].unstack("wm_yr_wk")
    piv = piv.reindex(pd.MultiIndex.from_arrays([meta.store_id, meta.item_id]))
    P = piv.to_numpy(np.float32)[:, piv.columns.get_indexer(cal.wm_yr_wk.to_numpy())]
    date = pd.to_datetime(cal.date)
    ev_t = np.stack([((cal.event_type_1 == e) | (cal.event_type_2 == e)).to_numpy() for e in EVENT_TYPES]).astype(np.float32)
    out = dict(D=D, P=P, wday=cal.wday.to_numpy(np.int64), month=cal.month.to_numpy(np.int64), dom=date.dt.day.to_numpy(np.int64), doy=date.dt.dayofyear.to_numpy(np.int64),
               snap=np.stack([cal[f"snap_{s}"].to_numpy() for s in ("CA", "TX", "WI")]).astype(np.float32), ev_t=ev_t,
               codes=np.stack([pd.factorize(meta[c])[0] for c in ("item_id", "dept_id", "cat_id", "store_id", "state_id")]).astype(np.int64),
               snap_state=pd.Categorical(meta.state_id, categories=["CA", "TX", "WI"]).codes.astype(np.int64),
               ids=meta.id.to_numpy().astype(str), sub_ids=sub.id.to_numpy().astype(str))
    for c in ("item_id", "dept_id", "cat_id", "store_id", "state_id"): out["meta_" + c] = meta[c].to_numpy().astype(str)
    os.makedirs(PREP, exist_ok=True); np.savez_compressed(os.path.join(PREP, "m5_prep.npz"), **out)


def load_prep():
    z = np.load(os.path.join(PREP, "m5_prep.npz"))
    d = types.SimpleNamespace(**{k: z[k] for k in z.files})
    d.meta = pd.DataFrame({"id": d.ids, **{c: getattr(d, "meta_" + c) for c in ("item_id", "dept_id", "cat_id", "store_id", "state_id")}})
    d.n, d.T = d.D.shape
    d.Pf = np.nan_to_num(d.P).astype(np.float32); d.Pmax = np.maximum(d.Pf[:, :FIT_END].max(1), 1e-3)
    d.S, d.keys = aggregation_matrix(d.meta)
    d.lvl = np.array([k[0] for k in d.keys])
    return d


LEVELS = [[], ["state_id"], ["store_id"], ["cat_id"], ["dept_id"], ["state_id", "cat_id"], ["state_id", "dept_id"], ["store_id", "cat_id"],
          ["store_id", "dept_id"], ["item_id"], ["item_id", "state_id"], ["item_id", "store_id"]]


def aggregation_matrix(meta):
    rows, n = [], len(meta)
    keys = []
    for lv, cols in enumerate(LEVELS):
        if not cols: g = np.zeros(n, int); names = ["Total"]
        else:
            k = meta[cols].astype(str).agg("|".join, axis=1) if len(cols) > 1 else meta[cols[0]].astype(str)
            codes, uniq = pd.factorize(k); g = codes; names = list(uniq)
        rows.append(sp.csr_matrix((np.ones(n), (g, np.arange(n))), shape=(len(names), n)))
        keys += [(lv, nm) for nm in names]
    return sp.vstack(rows).tocsr(), keys


class WRMSSE:
    def __init__(self, meta, train, prices_last28):
        """train: [n_series, T] unit sales used for scaling; prices_last28: [n_series] dollar sales over the last 28 training days (weights)."""
        self.S, self.keys = aggregation_matrix(meta)
        self.lvl = np.array([k[0] for k in self.keys])
        agg = self.S @ train.astype(np.float64)
        scale = np.empty(agg.shape[0])
        for i in range(agg.shape[0]):
            s = agg[i]; nz = np.nonzero(s)[0]
            s = s[nz[0]:] if len(nz) else s
            scale[i] = np.mean(np.diff(s) ** 2) if len(s) > 1 else 1.0
        self.scale = np.maximum(scale, 1e-12)
        dollars = self.S @ prices_last28
        w = np.zeros_like(dollars)
        for lv in range(len(LEVELS)):
            m = self.lvl == lv; w[m] = dollars[m] / dollars[m].sum()
        self.w = w

    def score(self, y_true, y_pred, per_level=False):
        e = (self.S @ (y_true.astype(np.float64) - y_pred)) ** 2
        rmsse = np.sqrt(e.mean(1) / self.scale)
        per = np.array([np.sum(self.w[self.lvl == lv] * rmsse[self.lvl == lv]) for lv in range(len(LEVELS))])
        return (float(per.mean()), per) if per_level else float(per.mean())


def dollars(P, D, a, b):
    """dollar sales over days a..b (1-based, inclusive)"""
    return (np.nan_to_num(P[:, a - 1:b]) * D[:, a - 1:b]).sum(1).astype(np.float64)


class Scorer:
    """Exact WRMSSE for one role: scale from the history before the origin, weights from the 28 dollar-sales days before it."""
    def __init__(self, d, role):
        t = ORIGIN[role]; self.S, self.lvl = d.S, d.lvl
        f = os.path.join(PREP, f"scorer_{role}.npz")
        if os.path.exists(f):
            z = np.load(f); self.scale, self.w = z["scale"], z["w"]
        else:
            W = WRMSSE(d.meta, d.D[:, :t].astype(np.float64), dollars(d.P, d.D, t - 27, t)); self.scale, self.w = W.scale, W.w
            np.savez(f, scale=self.scale, w=self.w)
        self.y = d.D[:, t:t + HZ].astype(np.float64) if t + HZ <= d.T else None

    def score(self, pred, per_level=False):
        e = (self.S @ (self.y - pred)) ** 2
        r = np.sqrt(e.mean(1) / self.scale)
        per = np.array([np.sum(self.w[self.lvl == l] * r[self.lvl == l]) for l in range(len(LEVELS))])
        return (float(per.mean()), per) if per_level else float(per.mean())


class Hist:
    """Cumulative tables over a [n, T] matrix: trailing-window sums at any origin t (number of known days) in O(1)."""
    def __init__(self, X):
        n, T = X.shape; self.X = X.astype(np.float32); z = np.zeros((n, 1))
        Xd = X.astype(np.float64)
        self.c1 = np.concatenate([z, np.cumsum(Xd, 1)], 1); self.c2 = np.concatenate([z, np.cumsum(Xd ** 2, 1)], 1)
        self.cn = np.concatenate([z, np.cumsum(X > 0, 1)], 1)
        self.last = np.maximum.accumulate(np.where(X > 0, np.arange(T)[None, :], -1), 1)
        self.first = np.where((X > 0).any(1), np.argmax(X > 0, 1), T)


WIN = (7, 14, 28, 56, 112, 224)


def features(h, d, rows, t, hz, scale=False, glob=None, lvl=None):
    """Feature matrix for local series `rows` (indices into h), origins t and horizons hz (all 1-D, equal length).
    glob maps local rows to global bottom-series ids (adds price and id features); lvl gives an aggregate-level id per local row.
    Every value is computed from days < t only, except calendar and price of the target day (known in advance in M5)."""
    X = h.X; f = {}
    s = 1.0 + (h.c1[rows, t] - h.c1[rows, np.maximum(t - 28, 0)]) / np.maximum(np.minimum(t, 28), 1) if scale else np.ones(len(rows))
    for w in WIN:
        a = np.maximum(t - w, 0); f[f"m{w}"] = (h.c1[rows, t] - h.c1[rows, a]) / np.maximum(t - a, 1) / s
    a = np.maximum(t - 28, 0); n28 = np.maximum(t - a, 1)
    f["std28"] = np.sqrt(np.maximum((h.c2[rows, t] - h.c2[rows, a]) / n28 - ((h.c1[rows, t] - h.c1[rows, a]) / n28) ** 2, 0)) / s
    f["nz28"] = (h.cn[rows, t] - h.cn[rows, a]) / n28
    a = np.maximum(t - 112, 0); f["nz112"] = (h.cn[rows, t] - h.cn[rows, a]) / np.maximum(t - a, 1)
    f["dsl"] = np.minimum(t - 1 - h.last[rows, t - 1], 400)
    f["lastv"] = X[rows, t - 1] / s
    g = t + hz - 1; k0 = -(-hz // 7); acc = np.zeros(len(rows)); cnt = np.zeros(len(rows))
    for j in range(4):
        ix = g - 7 * (k0 + j); ok = ix >= 0; acc += np.where(ok, X[rows, np.maximum(ix, 0)], 0); cnt += ok
    f["wd4"] = acc / np.maximum(cnt, 1) / s
    f["trend"] = f["m28"] / (f["m224"] + 1e-3); f["age"] = np.log1p(np.maximum(t - h.first[rows], 0)); f["s"] = s
    f["h"] = hz; f["wday"] = d.wday[g]; f["month"] = d.month[g]; f["dom"] = d.dom[g]
    f["ev"] = np.where(d.ev_t[:, g].sum(0) > 0, d.ev_t[:, g].argmax(0) + 1, 0)
    if glob is not None:
        gi = glob[rows]; f["snap"] = d.snap[d.snap_state[gi], g]
        for nm, ci in (("item", 0), ("dept", 1), ("cat", 2), ("state", 4)): f[nm] = d.codes[ci][gi]
        f["price"] = d.Pf[gi, g] / d.Pmax[gi]; f["pr_ratio"] = d.Pf[gi, g] / (d.Pf[gi, t - 1] + 1e-3)
    else:
        f["snap"] = d.snap[:, g].mean(0); f["lvl"] = lvl[rows]
    return pd.DataFrame(f).astype(np.float32), s


def draw_rows(rows, origins, per, tmax, rng):
    """Random (series, origin, horizon) training cells whose target day is <= tmax."""
    r = np.tile(rows, len(origins) * per); t = np.repeat(origins, len(rows) * per)
    h = rng.integers(1, HZ + 1, size=len(r)); ok = t + h <= tmax
    return r[ok], t[ok], h[ok]


def full_rows(rows, t):
    """All 28 horizons of every series at one origin."""
    r = np.repeat(rows, HZ); return r, np.full(len(r), t), np.tile(np.arange(1, HZ + 1), len(rows))


def align(pred, group_of, ref_pred, alpha, clip=(0.7, 1.4)):
    """Scale bottom forecasts so group sums move toward an independent forecast: ratio^alpha per (group, horizon). pred [n,28], ref_pred [G,28]."""
    G = ref_pred.shape[0]; M = sp.csr_matrix((np.ones(len(pred)), (group_of, np.arange(len(pred)))), shape=(G, len(pred)))
    cur = M @ pred; r = np.clip(ref_pred / np.maximum(cur, 1e-6), *clip) ** alpha
    return pred * r[group_of]


def atomic_json(path, obj):
    import json; tmp = path + ".tmp"; open(tmp, "w").write(json.dumps(obj, default=float)); os.replace(tmp, path)
