"""End-to-end check of the serving path on the cloud host: final fit (quick or full) -> mock time-series API over the example days ->
feature equivalence vs the training pipeline -> MAE of served days vs train actuals. Persists results.json (+ per-day predictions)."""
import argparse, json, os, sys, time
import numpy as np, pandas as pd
ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True); ap.add_argument("--root", default=None)
ap.add_argument("--quick", action="store_true")
ap.add_argument("--pkg-script", default=None, help="test the exact package embedded in the Kaggle script instead of src/enefit13")
a = ap.parse_args()
if a.pkg_script:
    import tempfile
    _d = tempfile.mkdtemp()
    _s = open(a.pkg_script).read()
    exec(_s.split("sys.path.insert(0, '/kaggle/working/pkg')")[0].replace("/kaggle/working/pkg", _d))
    sys.path.insert(0, _d)
else:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from enefit13 import data as D, serve as S, gbdt as G
os.makedirs(a.out, exist_ok=True)
T0 = time.time()
res = {"quick": a.quick}
def mark(stage):
    json.dump({"stage": stage, "t": round(time.time() - T0, 1), **res}, open(os.path.join(a.out, "progress.json.tmp"), "w"))
    os.replace(os.path.join(a.out, "progress.json.tmp"), os.path.join(a.out, "progress.json"))

store = D.Store(a.root)

class FakeEnv:
    def __init__(self):
        self.out = []
        self.t = {k: store.read(f"example_test_files/{k}.parquet") for k in
                  ("test", "revealed_targets", "client", "historical_weather", "forecast_weather", "electricity_prices", "gas_prices", "sample_submission")}
    def iter_test(self):
        for b in sorted(self.t["test"].data_block_id.unique()):
            g = lambda k: self.t[k][self.t[k].data_block_id == b].reset_index(drop=True)
            yield (g("test"), g("revealed_targets"), g("client"), g("historical_weather"), g("forecast_weather"), g("electricity_prices"), g("gas_prices"), g("sample_submission"))
    def predict(self, df):
        assert list(df.columns) == ["row_id", "data_block_id", "target"] and df.target.notna().all()
        self.out.append(df)

mark("train_final")
server = S.train_final(store, quick=a.quick, log=lambda m: print(m, flush=True))
res["train_sec"] = round(time.time() - T0, 1)
mark("serve")
feats = []
orig = server.predict_day
def spy(test, fw):
    t = time.time(); r = orig(test, fw); feats.append(server.last_features.copy()); res.setdefault("day_sec", []).append(round(time.time() - t, 2)); return r
server.predict_day = spy
env = FakeEnv()
n = S.run_api(env, server, log=lambda m: print(m, flush=True))
sub = pd.concat(env.out)
sub.to_csv(os.path.join(a.out, "submission_mock.csv"), index=False)
res.update(days=n, rows=int(len(sub)), target_min=float(sub.target.min()), target_max=float(sub.target.max()), nan=int(sub.target.isna().sum()))
mark("equivalence")
panel = D.load_panel(store)
an, ar, s_, ti, sid, t0g = D.series_grid(panel)
panel = D.add_lag_features(panel, an, s_, ti)
f = pd.concat(feats)
m = f.merge(panel, on=["prediction_unit_id", "is_consumption", "datetime"], suffixes=("_srv", ""))
res["matched_rows"], res["served_rows"] = int(len(m)), int(len(f))
bad = {}
for c in G.v1_features():
    if c in ("prediction_unit_id", "is_consumption"):
        continue
    x, y = m[c + "_srv"].astype(float).values, m[c].astype(float).values
    ok = np.isclose(x, y, equal_nan=True, rtol=1e-4, atol=1e-3)
    if not ok.all():
        bad[c] = round(float((~ok).mean()), 4)
res["feature_mismatch_fraction"] = bad
test = store.read("example_test_files/test.parquet")[["row_id", "prediction_unit_id", "is_consumption", "prediction_datetime"]]
sc = sub.merge(test, on="row_id").merge(panel[["prediction_unit_id", "is_consumption", "datetime", "target"]].rename(columns={"target": "truth"}),
                                          left_on=["prediction_unit_id", "is_consumption", "prediction_datetime"],
                                          right_on=["prediction_unit_id", "is_consumption", "datetime"])
res["mae_served_vs_train_actuals"] = round(float((sc.target - sc.truth).abs().mean()), 2)
res["scored_rows"] = int(len(sc))
res["wall_sec"] = round(time.time() - T0, 1)
json.dump(res, open(os.path.join(a.out, "results.json"), "w"), indent=1)
mark("done")
print(json.dumps({k: v for k, v in res.items() if k != "day_sec"}, indent=1))
