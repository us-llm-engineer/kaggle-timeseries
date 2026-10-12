"""Final fit on all data + incremental day-ahead serving through the Kaggle time-series API.
Blend = mean of available models among {V1 LightGBM-L1, V3 TiDE capacity-relative ensemble, V4 TiDE window-normalised ensemble}.
Everything the API delivers for day D is used only as the back-test pipeline would have it (revealed targets <= D-2, client <= D-2,
forecast origin D-1 02:00, day-ahead price of D-1)."""
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch

from . import data as D
from . import gbdt as G
from . import nn as N
from . import nn_ens as NE

EXTRA_DAYS = 420


def _pad(a, extra, fill):
    shape = list(a.shape)
    shape[1] = extra
    return np.concatenate([a, np.full(shape, fill, dtype=a.dtype)], 1)


def train_final(store, quick=False, v1_rounds=(67, 592), dl_epochs=(46, 40), log=print):
    """Fit V1/V3/V4 on every labelled day. Returns a Server."""
    t0 = time.time()
    panel = D.load_panel(store)
    an, ar, s, ti, sid, t0g = D.series_grid(panel)
    panel = D.add_lag_features(panel, an, s, ti)
    log(f"panel {len(panel)} rows, {len(sid)} series, features built in {time.time() - t0:.0f}s")
    # --- V1: global LightGBM L1 on y/scale, weight = scale (== raw MAE), one booster per target
    feats = G.v1_features()
    boosters = {}
    valid = panel.y_norm.notna()
    for c, r in zip((0, 1), v1_rounds):
        df = panel[valid & (panel.is_consumption == c)]
        boosters[c] = lgb.train(G.PARAMS_L1, G._dataset(df, feats), num_boost_round=(30 if quick else r))
        log(f"V1 booster c{c}: {len(df)} rows, {30 if quick else r} rounds, {time.time() - t0:.0f}s")
    # --- DL ensembles
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    td = N.build_tensors(panel, an, ar, s, ti, sid, t0g)
    sidx, didx = N.day_samples(td)
    models = {}
    drops, seeds = ([0.5], [0]) if quick else ([0.5, 0.2], [0, 1, 2, 3])
    for v, mode, ep in (("v3", "rel", dl_epochs[0]), ("v4", "raw", dl_epochs[1])):
        _, info, m = NE.train_ensemble(td, mode, dev, drops, seeds, (sidx, didx), None, None, "/tmp/enefit_final", v,
                                       epochs=(2 if quick else ep), budget_sec=None, final=True)
        models[v] = (m, mode)
        n_ep, wall_s, n_mem = info['epochs_run'], info['wall_sec'], len(info['members'])
        log(f"{v}: {n_ep} epochs, {wall_s:.0f}s, members {n_mem}")
    # extend grids so the API days can be appended
    for k, fill in (("an", np.nan), ("ar", np.nan), ("sc", 1.0)):
        td[k] = _pad(td[k], EXTRA_DAYS * 24, fill)
    td["wx"] = _pad(td["wx"], EXTRA_DAYS * 24, 0.0)
    td["T"] = td["an"].shape[1]
    st = D.station_map(store)
    last_client = {}
    cl = store.read("client.parquet")
    for r in cl.sort_values("date").itertuples():
        last_client[(int(r.county), int(r.is_business), int(r.product_type))] = (float(r.installed_capacity), float(r.eic_count))
    ep = store.read("electricity_prices.parquet")
    price = {pd.Timestamp(r.forecast_date): float(r.euros_per_mwh) for r in ep.itertuples()}
    return Server(td, sid, boosters, models, dev, st, last_client, price, t0g)


class Server:
    def __init__(self, td, sid, boosters, models, dev, st, last_client, price, t0):
        self.td, self.sid, self.boosters, self.models, self.dev = td, sid, boosters, models, dev
        self.st, self.last_client, self.price, self.t0 = st, last_client, price, pd.Timestamp(t0)
        self.helpers = {}
        self.n_calls = 0

    # ---- state updates from one API iteration -------------------------------------------------------------
    def _tidx(self, dt):
        return ((pd.to_datetime(dt) - self.t0) / pd.Timedelta(hours=1)).astype(int).values

    def _series(self, df):
        return np.array([self.sid.get((int(u), int(c)), -1) for u, c in zip(df.prediction_unit_id, df.is_consumption)])

    def update_revealed(self, rev):
        if rev is None or len(rev) == 0:
            return
        r = rev.dropna(subset=["target"])
        s, t = self._series(r), self._tidx(r.datetime)
        ok = (s >= 0) & (t >= 0) & (t < self.td["T"])
        s, t, y = s[ok], t[ok], r.target.values[ok].astype("float32")
        self.td["ar"][s, t] = y
        self.td["an"][s, t] = y / self.td["sc"][s, t]

    def update_client(self, client):
        for r in client.itertuples():
            self.last_client[(int(r.county), int(r.is_business), int(r.product_type))] = (float(r.installed_capacity), float(r.eic_count))

    def update_price(self, ep):
        for r in ep.itertuples():
            self.price[pd.Timestamp(r.forecast_date)] = float(r.euros_per_mwh)

    # ---- one forecast day -------------------------------------------------------------------------------------
    def predict_day(self, test, forecast_weather):
        test = test.copy()
        test["datetime"] = pd.to_datetime(test.prediction_datetime)
        w = D.county_weather_from_frame(forecast_weather, self.st)
        test = test.merge(w, on=["county", "datetime"], how="left")
        test["county"] = test.county.astype("int16")
        keys = list(zip(test.county, test.is_business, test.product_type))
        cap = np.array([self.last_client.get((int(a), int(b), int(c)), (np.nan, np.nan))[0] for a, b, c in keys])
        eic = np.array([self.last_client.get((int(a), int(b), int(c)), (np.nan, np.nan))[1] for a, b, c in keys])
        test["installed_capacity"], test["eic_count"] = cap, eic
        scale = np.where(test.is_consumption.values == 1, eic, cap)
        test["scale"] = np.clip(np.nan_to_num(scale, nan=1.0), 1.0, None).astype("float32")
        test["price_lag24"] = [self.price.get(t - pd.Timedelta(hours=24), np.nan) for t in test.datetime]
        dt = test.datetime
        test["hour"], test["dow"], test["month"] = dt.dt.hour.astype("int8"), dt.dt.dayofweek.astype("int8"), dt.dt.month.astype("int8")
        doy = dt.dt.dayofyear.values
        test["doy_s"], test["doy_c"] = np.sin(2 * np.pi * doy / 365.25).astype("float32"), np.cos(2 * np.pi * doy / 365.25).astype("float32")
        s, t = self._series(test), self._tidx(test.datetime)
        known = (s >= 0) & (t >= 0) & (t < self.td["T"])
        # state: scale + (standardised) weather for the day, then lags from the grids (all >= 48 h old)
        self.td["sc"][s[known], t[known]] = test.scale.values[known]
        wv = test[D.WEATHER].values.astype("float32")
        wv = np.where(np.isnan(wv), self.td["w_mu"], wv)
        self.td["wx"][s[known], t[known]] = (wv[known] - self.td["w_mu"]) / self.td["w_sd"]
        ti_known = t.copy()
        test = D.add_lag_features(test, self.td["an"], np.where(known, s, 0), np.where(known, t, 0))
        preds = {}
        # V1
        p1 = np.full(len(test), np.nan)
        for c, b in self.boosters.items():
            m = (test.is_consumption.values == c) & known
            if m.any():
                p1[m] = b.predict(test.loc[m, G.v1_features()]) * test.scale.values[m]
        preds["v1"] = np.clip(p1, 0, None)
        # DL: one sample per series for this day
        day = int(np.floor(t[known].min() / 24)) if known.any() else None
        if day is not None:
            sidx = np.unique(s[known])
            didx = np.full(len(sidx), day)
            for v, (model, mode) in self.models.items():
                if v not in self.helpers:
                    self.helpers[v] = N.Runner(self.td, mode, self.dev, 0, "/tmp", f"serve_{v}")
                    del self.helpers[v].model
                data = self.helpers[v].materialize(sidx, didx)
                out = NE.predict_members(model, data, mode, self.dev).mean(0).cpu().numpy()   # (nser, 24)
                pos = {int(k): i for i, k in enumerate(sidx)}
                arr = np.full(len(test), np.nan)
                for j in np.where(known)[0]:
                    arr[j] = out[pos[int(s[j])], int(t[j] - day * 24)]
                preds[v] = arr
        stack = np.stack([preds[k] for k in preds])
        blend = np.nanmean(stack, 0) if np.isfinite(stack).any() else np.zeros(len(test))
        blend = np.nan_to_num(blend, nan=0.0)
        self.n_calls += 1
        self.last_features = test
        return test.row_id.values, np.clip(blend, 0, None), {k: v for k, v in preds.items()}


def run_api(env, server, log=print, max_days=None):
    """Standard time-series API loop. Predictions always cover every row (fallback 0)."""
    n = 0
    for (test, revealed_targets, client, hist_w, forecast_weather, el_prices, gas_prices, sample_prediction) in env.iter_test():
        try:
            server.update_revealed(revealed_targets)
            server.update_client(client)
            server.update_price(el_prices)
            rid, yhat, _ = server.predict_day(test, forecast_weather)
            out = pd.DataFrame({"row_id": rid, "target": yhat})
            sample_prediction = sample_prediction.drop(columns=["target"]).merge(out, on="row_id", how="left")
            sample_prediction["target"] = sample_prediction.target.fillna(0.0)
        except Exception as e:                                       # never fail the whole run on one day
            log(f"day {n}: fallback to zeros ({type(e).__name__}: {str(e)[:200]})")
            sample_prediction = sample_prediction.copy()
            sample_prediction["target"] = 0.0
        env.predict(sample_prediction[["row_id", "target"]] if "data_block_id" not in sample_prediction else sample_prediction)
        n += 1
        if n % 10 == 0:
            log(f"served {n} days")
        if max_days and n >= max_days:
            break
    return n
