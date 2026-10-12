"""Shared data layer for the four variants (leak-free alignment checked on the data, see README).

Alignment facts (checked on the data):
- weather forecast for target day D: rows with origin_datetime == (D-1) 02:00 (hours_ahead 22..45);
- client (installed_capacity, eic_count) is only known up to day D-2;
- day-ahead electricity price for day D is not yet known at forecast time: use day D-1 (24 h lag);
- revealed targets lag 2 days: for hour h of day D the newest actual of the same hour is at day D-2 (lag 48 h).
"""
import hashlib
import json
import os

import numpy as np
import pandas as pd

HF_REPO = os.environ.get("ENEFIT_HF_REPO", "")  # optional private parquet mirror of the competition data
WCOLS = ["temperature", "dewpoint", "cloudcover_low", "cloudcover_mid", "cloudcover_high", "cloudcover_total",
         "10_metre_u_wind_component", "10_metre_v_wind_component", "direct_solar_radiation",
         "surface_solar_radiation_downwards", "snowfall", "total_precipitation"]
WSHORT = ["temp", "dew", "cl_low", "cl_mid", "cl_high", "cl_tot", "wind_u", "wind_v", "dir_rad", "ssr", "snow", "precip"]
LAGS = (48, 72, 96, 120, 168, 336)
REPORT_START = pd.Timestamp("2023-04-02")   # same 60-day window as the baselines in results/data/data_profile.txt
CHOOSE_START = pd.Timestamp("2023-03-05")   # 28 days before the report window: early stopping / epoch choice only


class Store:
    """Reads parquet from a local tree if present, else from the private HF dataset (sha256 checked against MANIFEST)."""

    def __init__(self, root=None, repo=HF_REPO):
        self.root, self.repo, self._man = root, repo, None

    def manifest(self):
        if self._man is None:
            self._man = json.load(open(self.path("MANIFEST.json", verify=False)))
        return self._man

    def path(self, rel, verify=True):
        if self.root and os.path.exists(os.path.join(self.root, rel)):
            return os.path.join(self.root, rel)
        from huggingface_hub import hf_hub_download
        p = hf_hub_download(self.repo, rel, repo_type="dataset")
        if verify and rel != "MANIFEST.json":
            want = self.manifest().get(rel, {}).get("sha256")
            h = hashlib.sha256(open(p, "rb").read()).hexdigest()
            assert want is None or h == want, f"sha256 mismatch for {rel}"
        return p

    def read(self, rel, columns=None):
        return pd.read_parquet(self.path(rel), columns=columns)


def months_between(a, b):
    return [str(p) for p in pd.period_range(a, b, freq="M")]


def county_weather_from_frame(f, st):
    """Forecast frame (any months / one API iteration) -> [county, datetime, w_*, n_*], forecast available at 11:00 of day D-1."""
    f = f.copy()
    f["forecast_datetime"] = pd.to_datetime(f["forecast_datetime"])
    f["origin_datetime"] = pd.to_datetime(f["origin_datetime"])
    keep = f.origin_datetime.dt.normalize() == (f.forecast_datetime.dt.normalize() - pd.Timedelta(days=1))
    f = f[keep].copy()
    f["lat"], f["lon"] = f.latitude.astype("float32").astype("float64").round(2), f.longitude.astype("float32").astype("float64").round(2)
    f = f.drop_duplicates(["forecast_datetime", "lat", "lon"])
    nat = f.groupby("forecast_datetime")[WCOLS].mean()
    nat.columns = ["n_" + c for c in WSHORT]
    f = f.merge(st, on=["lat", "lon"], how="left")
    g = f.dropna(subset=["county"]).groupby(["county", "forecast_datetime"])[WCOLS].mean()
    g.columns = ["w_" + c for c in WSHORT]
    g = g.reset_index()
    g["county"] = g.county.astype("int16")
    miss = [c for c in range(16) if c not in set(g.county.unique())]
    if miss:                                   # counties without any mapped station use the national mean
        base = nat.reset_index().rename(columns={"n_" + c: "w_" + c for c in WSHORT})
        for c in miss:
            extra = base.copy()
            extra["county"] = np.int16(c)
            g = pd.concat([g, extra[g.columns]], ignore_index=True)
    g = g.merge(nat.reset_index(), on="forecast_datetime", how="left").rename(columns={"forecast_datetime": "datetime"})
    for c in g.columns:
        if c.startswith(("w_", "n_")):
            g[c] = g[c].astype("float32")
    return g


def station_map(store):
    st = store.read("weather_station_to_county_mapping.parquet")
    return st.assign(lat=st.latitude.round(2), lon=st.longitude.round(2))[["lat", "lon", "county"]]


def county_weather(store, months):
    st = station_map(store)
    out = []
    for m in months:
        try:
            f = store.read(f"forecast_weather/{m}.parquet")
        except Exception:
            continue
        out.append(county_weather_from_frame(f, st))
    return pd.concat(out, ignore_index=True)


class CsvStore:
    """Same read(rel) interface as Store, backed by the competition CSVs (Kaggle input / local data/)."""

    def __init__(self, root):
        self.root, self._fw = root, None

    def read(self, rel, columns=None):
        r = self.root
        if rel == "train.parquet":
            return pd.read_csv(f"{r}/train.csv", parse_dates=["datetime"])
        if rel == "client.parquet":
            return pd.read_csv(f"{r}/client.csv", parse_dates=["date"])
        if rel == "electricity_prices.parquet":
            return pd.read_csv(f"{r}/electricity_prices.csv", parse_dates=["forecast_date", "origin_date"])
        if rel == "weather_station_to_county_mapping.parquet":
            return pd.read_csv(f"{r}/weather_station_to_county_mapping.csv")
        if rel.startswith("forecast_weather/"):
            if self._fw is None:
                self._fw = pd.read_csv(f"{r}/forecast_weather.csv", parse_dates=["origin_datetime", "forecast_datetime"])
                for c in WCOLS + ["latitude", "longitude"]:
                    self._fw[c] = self._fw[c].astype("float32")
            m = rel.split("/")[1].split(".")[0]
            return self._fw[self._fw.forecast_datetime.dt.strftime("%Y-%m") == m]
        raise FileNotFoundError(rel)


def load_panel(store, first_month="2021-09", last_month="2023-05"):
    """One row per (prediction_unit_id, is_consumption, datetime) with target, scale, calendar, weather, price, static columns."""
    tr = store.read("train.parquet")
    tr = tr.drop(columns=["row_id"], errors="ignore")
    tr["date"] = tr.datetime.dt.normalize()
    # scale: client row of day D-2
    cl = store.read("client.parquet")[["county", "is_business", "product_type", "date", "installed_capacity", "eic_count"]]
    cl = cl.assign(date=cl.date + pd.Timedelta(days=2))
    tr = tr.merge(cl, on=["county", "is_business", "product_type", "date"], how="left")
    tr = tr.sort_values(["prediction_unit_id", "is_consumption", "datetime"]).reset_index(drop=True)
    for c in ("installed_capacity", "eic_count"):
        tr[c] = tr.groupby("prediction_unit_id")[c].transform(lambda s: s.ffill().bfill())
    tr["scale"] = np.where(tr.is_consumption == 1, tr.eic_count, tr.installed_capacity).astype("float32")
    tr["scale"] = tr["scale"].clip(lower=1.0)
    # price of the previous day, same hour
    ep = store.read("electricity_prices.parquet")[["forecast_date", "euros_per_mwh"]]
    ep = ep.rename(columns={"forecast_date": "datetime", "euros_per_mwh": "price_lag24"})
    ep["datetime"] = ep.datetime + pd.Timedelta(hours=24)
    tr = tr.merge(ep, on="datetime", how="left")
    w = county_weather(store, months_between(first_month, last_month))
    tr = tr.merge(w, on=["county", "datetime"], how="left")
    dt = tr.datetime
    tr["hour"] = dt.dt.hour.astype("int8")
    tr["dow"] = dt.dt.dayofweek.astype("int8")
    tr["month"] = dt.dt.month.astype("int8")
    doy = dt.dt.dayofyear.values
    tr["doy_s"] = np.sin(2 * np.pi * doy / 365.25).astype("float32")
    tr["doy_c"] = np.cos(2 * np.pi * doy / 365.25).astype("float32")
    tr["y"] = tr.target.astype("float32")
    tr["y_norm"] = (tr.target / tr.scale).astype("float32")
    return tr


def series_grid(panel):
    """Wide array [series, hour] of y_norm and raw y on the full hourly grid; returns (arr_norm, arr_raw, series_index, t0)."""
    t0 = panel.datetime.min()
    T = int((panel.datetime.max() - t0) / pd.Timedelta(hours=1)) + 1
    keys = panel[["prediction_unit_id", "is_consumption"]].drop_duplicates().sort_values(["prediction_unit_id", "is_consumption"])
    sid = {(int(a), int(b)): i for i, (a, b) in enumerate(keys.itertuples(index=False))}
    s = np.fromiter((sid[(int(a), int(b))] for a, b in zip(panel.prediction_unit_id.values, panel.is_consumption.values)), dtype=np.int32, count=len(panel))
    ti = ((panel.datetime - t0) / pd.Timedelta(hours=1)).astype(int).values
    an = np.full((len(sid), T), np.nan, dtype=np.float32)
    ar = np.full((len(sid), T), np.nan, dtype=np.float32)
    an[s, ti] = panel.y_norm.values
    ar[s, ti] = panel.y.values
    return an, ar, s, ti, sid, t0


def add_lag_features(panel, an, s, ti):
    """Target-history features (normalised). Every lag is >= 48 h, i.e. what the API reveals (no leak)."""
    out = {}
    for k in LAGS:
        j = ti - k
        v = np.where(j >= 0, an[s, np.clip(j, 0, None)], np.nan)
        out[f"lag{k}"] = v.astype("float32")
    # same hour, days D-2..D-8 (mean/std), daily level of D-2 and last-7-days level
    stack = np.stack([np.where(ti - 24 * d >= 0, an[s, np.clip(ti - 24 * d, 0, None)], np.nan) for d in range(2, 9)])
    with np.errstate(all="ignore"):
        out["same_hour_mean7"] = np.nanmean(stack, 0).astype("float32")
        out["same_hour_std7"] = np.nanstd(stack, 0).astype("float32")
        out["same_hour_max7"] = np.nanmax(stack, 0).astype("float32")
    day0 = (ti // 24) * 24                      # index of 00:00 of the target day (grid starts at 00:00)
    lvl = []
    for d in (2, 3, 4, 5, 6, 7, 8):
        cols = np.stack([np.where(day0 - 24 * d + h >= 0, an[s, np.clip(day0 - 24 * d + h, 0, None)], np.nan) for h in range(24)])
        with np.errstate(all="ignore"):
            lvl.append(np.nanmean(cols, 0))
    lvl = np.stack(lvl)
    with np.errstate(all="ignore"):
        out["day_level_d2"] = lvl[0].astype("float32")
        out["week_level"] = np.nanmean(lvl, 0).astype("float32")
    for k, v in out.items():
        panel[k] = v
    return panel


STATIC = ["county", "is_business", "product_type", "is_consumption", "prediction_unit_id"]
CAL = ["hour", "dow", "month", "doy_s", "doy_c"]
WEATHER = ["w_" + c for c in WSHORT] + ["n_temp", "n_ssr", "n_cl_tot"]
SCALE = ["installed_capacity", "eic_count"]
LAGCOLS = [f"lag{k}" for k in LAGS] + ["same_hour_mean7", "same_hour_std7", "same_hour_max7", "day_level_d2", "week_level"]


def roles(panel):
    d = panel.datetime
    return {
        "fit": (d < CHOOSE_START).values,
        "choose": ((d >= CHOOSE_START) & (d < REPORT_START)).values,
        "report": (d >= REPORT_START).values,
    }
