"""Capacity facts: capacity normalisation, scale shift, consumption drivers, test-API facts. Appends to results/data/data_profile.txt."""
import os, numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); D = f"{ROOT}/data"
out = []
def P(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); out.append(s)
tr = pd.read_csv(f"{D}/train.csv", parse_dates=["datetime"]); cl = pd.read_csv(f"{D}/client.csv", parse_dates=["date"])
te = pd.read_csv(f"{D}/example_test_files/test.csv", parse_dates=["prediction_datetime"])
rv = pd.read_csv(f"{D}/example_test_files/revealed_targets.csv", parse_dates=["datetime"])
st = pd.read_csv(f"{D}/weather_station_to_county_mapping.csv")
P("\n== CAPACITY AND TEST-API FACTS ==")
# client row is dated D and applies to target hours of D+? : check which join gives the tighter capacity relation
tr["date"] = tr.datetime.dt.normalize()
k = ["county", "is_business", "product_type"]
for lag in (0, 1, 2):
    c = cl.copy(); c["date"] = c.date + pd.Timedelta(days=lag)
    m = tr[tr.is_consumption == 0].merge(c[k + ["date", "installed_capacity", "eic_count"]], on=k + ["date"], how="left")
    P(f"client joined with date+{lag}d: coverage {m.installed_capacity.notna().mean():.4f}")
c = cl.copy(); c["date"] = c.date + pd.Timedelta(days=1)
m = tr.merge(c[k + ["date", "installed_capacity", "eic_count"]], on=k + ["date"], how="left")
pr = m[(m.is_consumption == 0) & m.installed_capacity.notna()].copy()
pr["h"] = pr.datetime.dt.hour
mid = pr[(pr.h >= 10) & (pr.h <= 14)].copy()
mid["ratio"] = mid.target / mid.installed_capacity
P("production/capacity at 10-14h: quantiles", mid.ratio.quantile([.5, .9, .99, 1.0]).round(3).to_dict())
P("  by month median:", mid.groupby(mid.datetime.dt.to_period("M")).ratio.median().round(3).to_dict())
P("  units with ratio p99 > 1.2 (output above nameplate):", int((mid.groupby("prediction_unit_id").ratio.quantile(.99) > 1.2).sum()))
cons = m[m.is_consumption == 1].copy()
cons["per_eic"] = cons.target / cons.eic_count
P("consumption per eic: by month median:", cons.groupby(cons.datetime.dt.to_period("M")).per_eic.median().round(2).tail(14).to_dict())
P("consumption hour-of-day profile (sum):", (cons.groupby(cons.datetime.dt.hour).target.sum() / cons.target.sum()).round(3).to_dict())
P("consumption weekday profile:", (cons.groupby(cons.datetime.dt.dayofweek).target.sum() / cons.target.sum()).round(3).to_dict())
# target scale shift relative to capacity: is the *unit* relation stable year over year?
pr["ym"] = pr.datetime.dt.to_period("M")
a = pr[pr.ym == "2022-05"].groupby("prediction_unit_id").agg(t=("target", "sum"), c=("installed_capacity", "mean"))
b = pr[pr.ym == "2023-05"].groupby("prediction_unit_id").agg(t=("target", "sum"), c=("installed_capacity", "mean"))
j = a.join(b, lsuffix="_22", rsuffix="_23").dropna()
P("May-22 vs May-23 per unit: total production x%.2f, capacity x%.2f, production/capacity x%.2f (median over units)" % (
    j.t_23.sum() / j.t_22.sum(), j.c_23.sum() / j.c_22.sum(), float((j.t_23 / j.c_23 / (j.t_22 / j.c_22)).median())))
# test API facts
P("test currently_scored share:", float(te.currently_scored.mean()), " by block:", te.groupby("data_block_id").currently_scored.mean().to_dict())
P("revealed_targets: data_block_ids", sorted(rv.data_block_id.unique()), " rows", len(rv))
t = te.merge(rv.rename(columns={"datetime": "prediction_datetime"})[["prediction_unit_id", "is_consumption", "prediction_datetime", "target"]],
             on=["prediction_unit_id", "is_consumption", "prediction_datetime"], how="left")
P("test rows whose own target already in revealed (overlap with train tail):", float(t.target.notna().mean()))
P("stations with a county:", int(st.county.notna().sum()), "of", len(st), "; counties covered:", sorted(st.county.dropna().astype(int).unique()))
P("units in county 12 (UNKNOWN):", sorted(tr[tr.county == 12].prediction_unit_id.unique()))
# new-unit risk: units that start late in train
first = tr.groupby("prediction_unit_id").datetime.min().sort_values()
P("latest-starting units:", first.tail(6).to_dict())
last = tr.groupby("prediction_unit_id").datetime.max()
P("units ending before train end:", {int(u): str(v) for u, v in last[last < tr.datetime.max()].items()})
# zeros in production at daylight hours: dead/offline signal?
dz = tr[(tr.is_consumption == 0) & (tr.datetime.dt.hour.between(10, 14)) & (tr.datetime.dt.month.between(5, 8))]
P("production==0 at 10-14h in May-Aug share:", float((dz.target == 0).mean()))
with open(f"{ROOT}/results/data/data_profile.txt", "a") as f: f.write("\n".join(out) + "\n")
