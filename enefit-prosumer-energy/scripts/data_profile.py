"""Data profile (no models). Writes results/data/data_profile.txt.

Run: python3 scripts/data_profile.py   (pandas/numpy only; forecast_weather is read in float32)
"""
import json, os, sys, hashlib
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "results", "data", "data_profile.txt")
lines = []


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    lines.append(s)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()[:16]


P("== FILES ==")
for f in sorted(os.listdir(D)):
    p = os.path.join(D, f)
    if os.path.isfile(p):
        P(f"{f:42s} {os.path.getsize(p):>12d}  sha256[:16]={sha(p)}")

tr = pd.read_csv(f"{D}/train.csv", parse_dates=["datetime"])
cl = pd.read_csv(f"{D}/client.csv", parse_dates=["date"])
ep = pd.read_csv(f"{D}/electricity_prices.csv", parse_dates=["forecast_date", "origin_date"])
gp = pd.read_csv(f"{D}/gas_prices.csv", parse_dates=["forecast_date", "origin_date"])
hw = pd.read_csv(f"{D}/historical_weather.csv", parse_dates=["datetime"])
fw = pd.read_csv(f"{D}/forecast_weather.csv", parse_dates=["origin_datetime", "forecast_datetime"],
                 dtype={c: "float32" for c in ["latitude", "longitude", "temperature", "dewpoint",
                        "cloudcover_high", "cloudcover_low", "cloudcover_mid", "cloudcover_total",
                        "10_metre_u_wind_component", "10_metre_v_wind_component",
                        "direct_solar_radiation", "surface_solar_radiation_downwards",
                        "snowfall", "total_precipitation"]})
st = pd.read_csv(f"{D}/weather_station_to_county_mapping.csv")
te = pd.read_csv(f"{D}/example_test_files/test.csv", parse_dates=["prediction_datetime"])
rv = pd.read_csv(f"{D}/example_test_files/revealed_targets.csv", parse_dates=["datetime"])
cmap = json.load(open(f"{D}/county_id_to_name_map.json"))

P("\n== SHAPES / RANGE / MISSING ==")
for n, df in [("train", tr), ("client", cl), ("elec", ep), ("gas", gp), ("hist_w", hw), ("fcst_w", fw),
              ("station_map", st), ("test_example", te), ("revealed", rv)]:
    na = df.isna().sum()
    na = {k: int(v) for k, v in na.items() if v}
    P(f"{n:12s} shape={df.shape} dup_rows={int(df.duplicated().sum())} NA={na}")
P("train datetime", tr.datetime.min(), "->", tr.datetime.max(), " data_block_id", tr.data_block_id.min(), "->", tr.data_block_id.max())
P("test  datetime", te.prediction_datetime.min(), "->", te.prediction_datetime.max(), "cols", list(te.columns))
P("hist_w", hw.datetime.min(), "->", hw.datetime.max(), " stations(lat,lon)=", hw[["latitude", "longitude"]].drop_duplicates().shape[0])
P("fcst_w origin", fw.origin_datetime.min(), "->", fw.origin_datetime.max(), " grid points=", fw[["latitude", "longitude"]].drop_duplicates().shape[0],
  " hours_ahead", fw.hours_ahead.min(), fw.hours_ahead.max())
P("county map", cmap)
P("station->county: counties with no station =", sorted(set(range(16)) - set(st.county.dropna().astype(int))),
  " stations w/o county =", int(st.county.isna().sum()), "of", len(st))

P("\n== TRAIN STRUCTURE ==")
P("counties", tr.county.nunique(), "units", tr.prediction_unit_id.nunique(),
  "business", sorted(tr.is_business.unique()), "product", sorted(tr.product_type.unique()))
P("unit key unique mapping:", tr.groupby("prediction_unit_id")[["county", "is_business", "product_type"]].nunique().max().to_dict())
P("rows per (unit,is_consumption) : min/med/max", tr.groupby(["prediction_unit_id", "is_consumption"]).size().describe()[["min", "50%", "max"]].tolist())
full = pd.date_range(tr.datetime.min(), tr.datetime.max(), freq="h")
P("hours in full range", len(full), " hours present", tr.datetime.nunique())
cnt = tr.groupby(["prediction_unit_id", "is_consumption"]).size()
P("units short of full hours:", int((cnt < len(full)).sum()), "of", len(cnt))
miss_h = sorted(set(full) - set(tr.datetime.unique()))
P("hours absent for ALL units:", len(miss_h), miss_h[:5])
first = tr.groupby("prediction_unit_id").datetime.min()
P("units whose first obs is > start+30d:", int((first > tr.datetime.min() + pd.Timedelta(days=30)).sum()))
P("target NA rows", int(tr.target.isna().sum()), " target==0:", float((tr.target == 0).mean()))
for c, g in tr.groupby("is_consumption"):
    t = g.target
    P(f"is_consumption={c}: n={len(g)} mean={t.mean():.1f} median={t.median():.1f} p90={t.quantile(.9):.0f} p99={t.quantile(.99):.0f} max={t.max():.0f} zero_frac={(t==0).mean():.3f}")

P("\n== SEGMENT SHARE OF TOTAL |TARGET| (what MAE rewards) ==")
tr["abs"] = tr.target.abs()
tot = tr["abs"].sum()
P("by is_consumption:", (tr.groupby("is_consumption")["abs"].sum() / tot).round(3).to_dict())
P("by is_business   :", (tr.groupby("is_business")["abs"].sum() / tot).round(3).to_dict())
P("by product_type  :", (tr.groupby("product_type")["abs"].sum() / tot).round(3).to_dict())
by_county = (tr.groupby("county")["abs"].sum() / tot).sort_values(ascending=False)
P("by county (top 6):", by_county.head(6).round(3).to_dict())
by_unit = (tr.groupby("prediction_unit_id")["abs"].sum() / tot).sort_values(ascending=False)
P("by unit: top5 share", by_unit.head(5).round(3).to_dict(), " top10 cum", round(float(by_unit.head(10).sum()), 3),
  " top20 cum", round(float(by_unit.head(20).sum()), 3))

P("\n== TIME STRUCTURE ==")
tr["hour"] = tr.datetime.dt.hour
tr["month"] = tr.datetime.dt.to_period("M")
P("mean target by month, production | consumption (sum over units, per hour):")
m = tr.groupby(["month", "is_consumption"]).target.sum().unstack() / tr.groupby("month").datetime.nunique().values[:, None]
P(m.round(0).to_string())
prod = tr[tr.is_consumption == 0]
P("production by hour (mean over units/days):", prod.groupby("hour").target.mean().round(0).to_dict())
P("production zero-fraction at night(0-3h):", float((prod[prod.hour <= 3].target == 0).mean()))
dst = tr.groupby(tr.datetime.dt.date).datetime.nunique()
P("days with !=24 distinct hours (DST):", dst[dst != 24].to_dict())

P("\n== CLIENT (installed capacity / eic_count) ==")
P("client rows", len(cl), " dates", cl.date.min(), "->", cl.date.max(), " groups", cl.groupby(["county", "is_business", "product_type"]).ngroups)
cs = cl.groupby("date")[["installed_capacity", "eic_count"]].sum()
P("total installed capacity: first / mid / last:", cs.installed_capacity.iloc[[0, len(cs) // 2, -1]].round(0).tolist(),
  " eic_count:", cs.eic_count.iloc[[0, len(cs) // 2, -1]].tolist())
g = cl.sort_values("date").groupby(["county", "is_business", "product_type"]).installed_capacity
P("capacity step events (day-to-day change>0.5%): per group avg", float(g.apply(lambda s: (s.pct_change().abs() > .005).sum()).mean()))
P("capacity growth first->last (%):", round(100 * (cs.installed_capacity.iloc[-1] / cs.installed_capacity.iloc[0] - 1), 1))

P("\n== PRICES ==")
P("elec euros_per_mwh: min/mean/max", ep.euros_per_mwh.min(), round(ep.euros_per_mwh.mean(), 1), ep.euros_per_mwh.max(),
  " negative frac", float((ep.euros_per_mwh < 0).mean()))
P("gas: rows", len(gp), "range", gp.forecast_date.min(), "->", gp.forecast_date.max())

P("\n== WEATHER <-> TARGET (does the forecast carry the production signal?) ==")
# county-level mean of forecast solar radiation at hours_ahead 24-47 (the available day-ahead forecast), joined by hour
fwh = fw[(fw.hours_ahead >= 22) & (fw.hours_ahead <= 45)]
agg = fwh.groupby("forecast_datetime")[["surface_solar_radiation_downwards", "direct_solar_radiation", "cloudcover_total", "temperature"]].mean().reset_index()
agg.columns = ["datetime", "fc_ssr", "fc_dir", "fc_cloud", "fc_temp"]
hag = hw.groupby("datetime")[["shortwave_radiation", "direct_solar_radiation", "cloudcover_total", "temperature"]].mean().reset_index()
hag.columns = ["datetime", "h_sw", "h_dir", "h_cloud", "h_temp"]
cap = cl.copy()
cap["datetime0"] = cap.date
tp = prod.merge(agg, on="datetime", how="left").merge(hag, on="datetime", how="left")
tp = tp[(tp.hour >= 6) & (tp.hour <= 20)]
# normalise by size: per-unit target / unit's own p99 so units are comparable
tp["norm"] = tp.target / tp.groupby("prediction_unit_id").target.transform(lambda s: s.quantile(.99) + 1e-9)
for c in ["fc_ssr", "fc_dir", "fc_cloud", "h_sw", "h_dir", "h_cloud"]:
    P(f"corr(norm production, {c}) = {tp['norm'].corr(tp[c]):.3f}   [n={tp[c].notna().sum()}]")
P("corr(h_sw, fc_ssr) =", round(float(tp[["h_sw", "fc_ssr"]].corr().iloc[0, 1]), 3))

P("\n== TRIVIAL BASELINES (MAE; last 60 days of train as pseudo-test; no peeking at the example test) ==")
cut = tr.datetime.max() - pd.Timedelta(days=60)
fit, ev = tr[tr.datetime <= cut], tr[tr.datetime > cut].copy()
ev["hour"] = ev.datetime.dt.hour
P("eval rows", len(ev), " scale: mean|y| =", round(float(ev.target.mean()), 1))
P("zero predictor          MAE =", round(float(ev.target.abs().mean()), 2))
P("global median           MAE =", round(float((ev.target - fit.target.median()).abs().mean()), 2))
key = ["prediction_unit_id", "is_consumption"]
med = fit.assign(hour=fit.datetime.dt.hour).groupby(key + ["hour"]).target.median().rename("m")
e2 = ev.join(med, on=key + ["hour"])
P("unit x hour median      MAE =", round(float((e2.target - e2.m).abs().mean()), 2))
# lag-48h same unit/hour: what the API gives (revealed targets are 2 days behind)
lag = tr[key + ["datetime", "target"]].copy()
lag["datetime"] = lag["datetime"] + pd.Timedelta(hours=48)
e3 = ev.merge(lag.rename(columns={"target": "lag48"}), on=key + ["datetime"], how="left")
P("lag-48h (same unit/hr)  MAE =", round(float((e3.target - e3.lag48).abs().mean()), 2), " coverage", round(float(e3.lag48.notna().mean()), 3))
lag7 = lag.copy(); lag7["datetime"] = lag7.datetime + pd.Timedelta(hours=120)  # 48h+120h = 168h
e4 = ev.merge(lag7.rename(columns={"target": "lag168"}), on=key + ["datetime"], how="left")
P("lag-168h (same unit/hr) MAE =", round(float((e4.target - e4.lag168).abs().mean()), 2), " coverage", round(float(e4.lag168.notna().mean()), 3))
for c, g in e3.groupby("is_consumption"):
    P(f"  lag-48 MAE is_consumption={c}: {float((g.target - g.lag48).abs().mean()):.2f}  (mean|y|={g.target.mean():.1f})")

P("\n== TEST EXAMPLE vs TRAIN (shift / new entities) ==")
P("test rows", len(te), " units", te.prediction_unit_id.nunique(), " units not in train:", sorted(set(te.prediction_unit_id) - set(tr.prediction_unit_id)))
tu = set(tr.prediction_unit_id)
P("train units missing from test:", sorted(tu - set(te.prediction_unit_id)))
P("test current_date range:", te.prediction_datetime.min(), te.prediction_datetime.max(), " data_block_id", te.data_block_id.min(), te.data_block_id.max())
P("train last 3 data_block_ids:", sorted(tr.data_block_id.unique())[-3:], " first test blocks:", sorted(te.data_block_id.unique())[:3])
# seasonal extrapolation: is the unseen test season inside the training support?
P("train covers months:", sorted(tr.month.astype(str).unique())[0], "..", sorted(tr.month.astype(str).unique())[-1])
P("production in June: train years =", sorted({str(p)[:4] for p in tr.month.unique() if str(p)[5:] == '06'}))
P("production level: Jun-2022 vs May-2023 mean by-hour sum (growth in capacity => scale shift):")
ms = prod.groupby("month").target.sum() / prod.groupby("month").datetime.nunique()
P(ms.round(0).tail(14).to_string())

P("\n== FORECAST-WEATHER REVISION (how noisy is the day-ahead forecast vs measurement?) ==")
x = fwh.merge(hw.groupby("datetime")[["temperature"]].mean().reset_index().rename(columns={"temperature": "h_t"}),
              left_on="forecast_datetime", right_on="datetime", how="inner")
P("MAE forecast temperature vs historical mean temperature (any station):", round(float((x.temperature - x.h_t).abs().mean()), 2))
P("forecast_weather duplicates on (origin,forecast,lat,lon):", int(fw.duplicated(["origin_datetime", "forecast_datetime", "latitude", "longitude"]).sum()))

os.makedirs(os.path.dirname(OUT), exist_ok=True)
open(OUT, "w").write("\n".join(lines) + "\n")
print("wrote", OUT)
