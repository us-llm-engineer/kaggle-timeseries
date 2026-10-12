"""Per-stock feature + sequence builder (vectorised), pre-averaging estimators, cross-stock aggregates.

build_stock(book_df, trade_df, stock_id) -> (features DataFrame, sequence float16 [G, N_BINS, 13]); rows aligned by time_id order.
Only the stock's own window data is used: no target, no other window, so no leakage by construction (see tests)."""
import numpy as np
import pandas as pd
from .common import N_BINS, BIN_S

BOOK_COLS = ["time_id", "seconds_in_bucket", "bid_price1", "ask_price1", "bid_price2", "ask_price2",
             "bid_size1", "ask_size1", "bid_size2", "ask_size2"]
TRADE_COLS = ["time_id", "seconds_in_bucket", "price", "size", "order_count"]
N_CH = 13
WINDOWS = (100, 200, 300, 400, 500, 600)       # cumulative sub-windows (1/6 .. 6/6 of the window)


def _ffill_bfill(a):
    """Forward then backward fill NaN along axis 1 of [G, T]; rows that are all NaN become 0."""
    G, T = a.shape
    valid = ~np.isnan(a)
    idx = np.where(valid, np.arange(T)[None, :], 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    out = np.take_along_axis(a, idx, axis=1)
    bad = np.isnan(out)
    if bad.any():
        ridx = np.where(valid, np.arange(T)[None, :], T - 1)
        ridx = np.minimum.accumulate(ridx[:, ::-1], axis=1)[:, ::-1]
        out = np.where(bad, np.take_along_axis(a, ridx, axis=1), out)
    return np.nan_to_num(out, nan=0.0)


def preavg(Y, c=0.2, i_n=3):
    """Pre-averaging estimators on Y [G, N+1] (log prices), T = 1 per window.
    E[sum_m bar_m^2] = IV/3 + M*2k*sigU2/(k+1)^2  (the edge noise term U_{a+k} cancels; derived and checked by simulation in tests); two-step noise correction as in
    Li, Laeven, Vellekoop (dependent noise): lag-j_n noise variance, lag autocovariances up to i_n, finite-sample bias term j*IV/(2(N-j+1)).
    Returns dict of arrays [G]: pav_raw (no noise correction), iv1 (i.i.d.-noise step 1), iv2 (step 2, i.i.d. noise, bias-corrected),
    iv3 (step 2, dependent noise), sigu (sqrt of the i.i.d. noise variance estimate)."""
    G, n1 = Y.shape
    N = n1 - 1
    nan = np.full(G, np.nan)
    k = max(1, int(round(c * np.sqrt(max(N, 1)))))
    M = N // (2 * k) if N > 0 else 0
    if M < 1 or N < 8:
        return {"pav_raw": nan, "iv1": nan, "iv2": nan, "iv3": nan, "sigu": nan}
    D = Y[:, k:] - Y[:, :-k]
    cs = np.concatenate([np.zeros((G, 1)), np.cumsum(D, axis=1)], axis=1)
    starts = (2 * np.arange(1, M + 1) - 2) * k
    bar = (cs[:, starts + k + 1] - cs[:, starts]) / (k + 1)
    pav = (bar ** 2).sum(1)

    def yy(j):                                                   # <Y,Y>_n(j)
        return ((Y[:, j:] - Y[:, :-j]) ** 2).sum(1) / (2.0 * (N - j + 1))

    sig1 = yy(1)
    coef = 2.0 * M * k / (k + 1) ** 2
    iv1 = 3.0 * (pav - coef * sig1)                                  # step 1: i.i.d. noise, no finite-sample correction
    ivp = np.maximum(iv1, 0.0)
    var_u2 = sig1 - ivp / (2.0 * N)                                  # j = 1 finite-sample bias term j*IV/(2(N-j+1))
    iv2 = 3.0 * (pav - coef * var_u2)                                # step 2: i.i.d. noise with the bias correction (stable)
    jn = int(min(20, max(2, N // 4)))
    var_u = yy(jn) - jn * ivp / (2.0 * (N - jn + 1))                 # dependent noise: lag-j_n variance ...
    s3 = var_u.copy()
    for j in range(1, i_n + 1):                                      # ... plus autocovariances up to i_n
        s3 = s3 + 2.0 * (var_u - (yy(j) - j * ivp / (2.0 * (N - j + 1))))
    iv3 = 3.0 * (pav - coef * s3)                                    # step 2, dependent noise (high variance on short windows)
    return {"pav_raw": np.sqrt(3.0 * pav), "iv1": iv1, "iv2": iv2, "iv3": iv3, "sigu": np.sqrt(np.maximum(var_u2, 0.0))}


def _signed_sqrt(x):
    return np.sqrt(np.maximum(x, 0.0))


def build_stock(book, trade, stock_id, want_seq=True):
    b = book[BOOK_COLS].to_numpy(dtype=np.float64)
    o = np.lexsort((b[:, 1], b[:, 0]))
    b = b[o]
    tid, sec = b[:, 0].astype(np.int64), b[:, 1]
    b1, a1, b2, a2, bs1, as1, bs2, as2 = (b[:, i] for i in range(2, 10))
    uniq, start, counts = np.unique(tid, return_index=True, return_counts=True)
    G, n = len(uniq), len(tid)
    gidx = np.repeat(np.arange(G), counts)
    first = np.zeros(n, bool); first[start] = True

    def gsum(x): return np.add.reduceat(x, start)
    def gmean(x): return gsum(x) / counts
    def gstd(x): return np.sqrt(np.maximum(gsum(x * x) / counts - gmean(x) ** 2, 0.0))

    wap1 = (b1 * as1 + a1 * bs1) / (bs1 + as1)
    wap2 = (b2 * as2 + a2 * bs2) / (bs2 + as2)
    lw1, lw2 = np.log(wap1), np.log(wap2)
    r1 = np.diff(lw1, prepend=lw1[0]); r1[first] = 0.0
    r2 = np.diff(lw2, prepend=lw2[0]); r2[first] = 0.0
    mid = (a1 + b1) / 2
    spread = (a1 - b1) / mid
    spread2 = (a2 - b2) / mid
    imb1 = (bs1 - as1) / (bs1 + as1)
    imb2 = (bs2 - as2) / (bs2 + as2)
    d1, d2 = bs1 + as1, bs2 + as2
    wdiff = (wap1 - mid) / mid
    chg_b = np.diff(b1, prepend=b1[0]) != 0; chg_b[first] = False
    chg_a = np.diff(a1, prepend=a1[0]) != 0; chg_a[first] = False

    F = {"stock_id": np.full(G, stock_id, dtype=np.int16), "time_id": uniq.astype(np.int32)}
    r1sq = r1 * r1
    for i, w in enumerate(WINDOWS, 1):
        F[f"rv_w{i}"] = _signed_sqrt(gsum(r1sq * (sec < w)))
    F["rv_all"] = F["rv_w6"]
    F["rv_late300"] = _signed_sqrt(gsum(r1sq * (sec >= 300)))
    F["rv_late450"] = _signed_sqrt(gsum(r1sq * (sec >= 450)))
    F["rv_wap2"] = _signed_sqrt(gsum(r2 * r2))
    bp = np.abs(r1) * np.abs(np.concatenate([[0.0], r1[:-1]])); bp[first] = 0.0
    F["bpv"] = _signed_sqrt(gsum(bp) * np.pi / 2)
    F["rq"] = np.sqrt(gsum(r1sq * r1sq))
    F["nz_ret_frac"] = gsum((r1 != 0).astype(float)) / counts
    F["max_abs_ret"] = np.maximum.reduceat(np.abs(r1), start)
    F["n_book"] = counts.astype(np.float32)
    F["first_sec"] = sec[start]
    F["last_sec"] = np.maximum.reduceat(sec, start)
    for name, x in (("spread", spread), ("spread2", spread2), ("imb1", imb1), ("imb2", imb2), ("wdiff", wdiff),
                    ("depth1", np.log1p(d1)), ("depth2", np.log1p(d2))):
        F[f"{name}_m"], F[f"{name}_s"] = gmean(x), gstd(x)
    F["depth_ratio_m"] = gmean(np.log1p(d2) - np.log1p(d1))
    F["chg_bid_frac"], F["chg_ask_frac"] = gsum(chg_b.astype(float)) / counts, gsum(chg_a.astype(float)) / counts
    F["wap_drift"] = lw1[start + counts - 1] - lw1[start]
    F["size1_sum"] = np.log1p(gsum(d1))

    # ---- trades
    t = trade[TRADE_COLS].to_numpy(dtype=np.float64) if trade is not None and len(trade) else np.zeros((0, 5))
    tcnt = np.zeros(G); tsz = np.zeros(G); tor = np.zeros(G); trv2 = np.zeros(G); tmaxsz = np.zeros(G); tchg = np.zeros(G)
    prog = {w: (np.zeros(G), np.zeros(G), np.zeros(G)) for w in WINDOWS}
    tg = np.zeros(0, np.int64); tsec = np.zeros(0); tlp = np.zeros(0); tsize = np.zeros(0); tord = np.zeros(0)
    if len(t):
        t = t[np.lexsort((t[:, 1], t[:, 0]))]
        ttid = t[:, 0].astype(np.int64)
        pos = np.searchsorted(uniq, ttid); pos = np.minimum(pos, G - 1)
        ok = uniq[pos] == ttid
        t, pos = t[ok], pos[ok]
        tg, tsec, tsize, tord = pos, t[:, 1], t[:, 3], t[:, 4]
        tlp = np.log(t[:, 2])
        tfirst = np.ones(len(tg), bool); tfirst[1:] = tg[1:] != tg[:-1]
        tret = np.diff(tlp, prepend=tlp[0]); tret[tfirst] = 0.0
        tcnt = np.bincount(tg, minlength=G).astype(float)
        tsz = np.bincount(tg, weights=tsize, minlength=G)
        tor = np.bincount(tg, weights=tord, minlength=G)
        trv2 = np.bincount(tg, weights=tret * tret, minlength=G)
        tchg = np.bincount(tg, weights=(tret != 0).astype(float), minlength=G)
        np.maximum.at(tmaxsz, tg, tsize)
        for w in WINDOWS:
            m = tsec < w
            prog[w] = (np.bincount(tg, weights=m.astype(float), minlength=G),
                       np.bincount(tg, weights=tsize * m, minlength=G),
                       np.bincount(tg, weights=tord * m, minlength=G))
    F["n_trade"], F["trade_size_sum"], F["trade_orders"] = tcnt.astype(np.float32), tsz, tor
    F["trade_size_max"], F["trade_rv"] = tmaxsz, _signed_sqrt(trv2)
    F["trade_chg_frac"] = tchg / np.maximum(tcnt, 1)
    F["trade_size_per_order"] = tsz / np.maximum(tor, 1)
    F["trade_size_mean"] = tsz / np.maximum(tcnt, 1)
    for i, w in enumerate(WINDOWS, 1):
        F[f"tcnt_w{i}"], F[f"tsize_w{i}"], F[f"tord_w{i}"] = prog[w]

    # ---- 1-second grid (forward filled, the paper's 'quote stays until new information' rule) and subsampled RV
    key1 = gidx * 600 + np.minimum(sec.astype(np.int64), 599)
    grid = np.full(G * 600, np.nan)
    grid[key1] = lw1
    grid = _ffill_bfill(grid.reshape(G, 600))
    dg = np.diff(grid, axis=1)
    F["rv_grid1"] = _signed_sqrt((dg ** 2).sum(1))
    for step, name in ((5, "rv_sub5"), (30, "rv_sub30")):
        acc = np.zeros(G)
        for off in range(step):
            acc += (np.diff(grid[:, off::step], axis=1) ** 2).sum(1)
        F[name] = _signed_sqrt(acc / step)
    pg = preavg(grid)
    F["pavg_raw"], F["pavg_iv1"], F["pavg_iv2"], F["pavg_iv3"], F["pavg_sigu"] = pg["pav_raw"], _signed_sqrt(pg["iv1"]), _signed_sqrt(pg["iv2"]), _signed_sqrt(pg["iv3"]), pg["sigu"]

    # ---- pre-averaging on the observation index (tick time): per window, variable length
    tick = {k: np.full(G, np.nan) for k in ("pav_raw", "iv1", "iv2", "iv3", "sigu")}
    for g in range(G):
        s, c = start[g], counts[g]
        if c >= 17:
            res = preavg(lw1[s:s + c][None, :])
            for k in tick: tick[k][g] = res[k][0]
    F["pavt_raw"], F["pavt_iv1"], F["pavt_iv2"], F["pavt_iv3"], F["pavt_sigu"] = tick["pav_raw"], _signed_sqrt(tick["iv1"]), _signed_sqrt(tick["iv2"]), _signed_sqrt(tick["iv3"]), tick["sigu"]

    feat = pd.DataFrame({k: (v.astype(np.float32) if v.dtype.kind == "f" else v) for k, v in F.items()})

    seq = None
    if want_seq:
        K = gidx * N_BINS + (sec.astype(np.int64) // BIN_S)
        GB = G * N_BINS
        cnt = np.bincount(K, minlength=GB).astype(float)
        def bmean(x):
            s_ = np.bincount(K, weights=x, minlength=GB)
            return np.where(cnt > 0, s_ / np.maximum(cnt, 1), np.nan).reshape(G, N_BINS)
        ret_sq = np.bincount(K, weights=r1sq, minlength=GB).reshape(G, N_BINS)
        spr, im, dd1, dd2 = (_ffill_bfill(bmean(x)) for x in (spread * 1e3, imb1, np.log1p(d1) / 6, np.log1p(d2) / 6))
        last = np.full(GB, np.nan); last[K] = lw1
        last = _ffill_bfill(last.reshape(G, N_BINS))
        ret_bin = np.diff(last, axis=1, prepend=last[:, :1])
        if len(tg):
            TK = tg * N_BINS + (tsec.astype(np.int64) // BIN_S)
            tc = np.bincount(TK, minlength=GB).astype(float).reshape(G, N_BINS)
            ts_ = np.bincount(TK, weights=tsize, minlength=GB).reshape(G, N_BINS)
            to_ = np.bincount(TK, weights=tord, minlength=GB).reshape(G, N_BINS)
            tl = np.full(GB, np.nan); tl[TK] = tlp
            tl = _ffill_bfill(tl.reshape(G, N_BINS))
            tr_bin = np.diff(tl, axis=1, prepend=tl[:, :1])
        else:
            tc = ts_ = to_ = tr_bin = np.zeros((G, N_BINS))
        chans = [np.log1p(1e4 * np.sqrt(ret_sq)), np.clip(ret_bin * 1e4, -30, 30) / 10, spr, im, dd1, dd2,
                 cnt.reshape(G, N_BINS) / 5, np.log1p(tc) / 3, np.log1p(ts_) / 8, np.log1p(to_) / 5,
                 np.clip(tr_bin * 1e4, -30, 30) / 10, (cnt.reshape(G, N_BINS) > 0).astype(float), (tc > 0).astype(float)]
        seq = np.stack(chans, axis=2).astype(np.float16)
    return feat, seq


CROSS_COLS = ["rv_all", "rv_sub5", "pavt_iv2", "spread_m", "depth1_m", "n_trade", "trade_size_sum", "rv_late300", "imb1_s", "nz_ret_frac"]


def add_cross(df):
    """Same-time_id cross-stock aggregates and unsupervised per-stock level; touches no target column."""
    out = df.copy()
    g = out.groupby("time_id")
    for c in CROSS_COLS:
        x = np.log(out[c].clip(lower=1e-12)) if c.startswith(("rv", "pav", "trade_size", "n_trade")) else out[c]
        gx = x.groupby(out["time_id"])
        mu, sd = gx.transform("mean"), gx.transform("std")
        out[f"x_{c}_mean"], out[f"x_{c}_std"] = mu.astype(np.float32), sd.astype(np.float32)
        out[f"x_{c}_rank"] = gx.rank(pct=True).astype(np.float32)
        out[f"x_{c}_dev"] = (x - mu).astype(np.float32)
    out["x_n_stocks"] = g["stock_id"].transform("size").astype(np.float32)
    out["stock_logrv_mean"] = np.log(out["rv_all"].clip(lower=1e-12)).groupby(out["stock_id"]).transform("mean").astype(np.float32)
    return out
