"""Aggregate error tables for the report role (origin 1913) and the choose role (origin 1885), from the saved forecasts.
Run: python scripts/analyze.py --preds results/preds --out results   -> per_level.csv, per_horizon.csv, bias_total.csv, bias_by_volume.csv, store_iterations.csv"""
import argparse, os, re, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from m5lib import *

ap = argparse.ArgumentParser(); ap.add_argument("--preds", default="results/preds"); ap.add_argument("--out", default="results"); ap.add_argument("--log", default="results/cpu_run.log"); a = ap.parse_args()
V = ["V1_lgb_squared", "V2_lgb_tweedie", "V3_lgb_tweedie_reconciled", "V4_lagfree_aligned"]
d = load_prep(); sc = {r: Scorer(d, r) for r in ("choose", "report")}
# reference forecast: mean of the last 28 days times the weekday profile of the last 52 weeks
def weekday_profile(role):
    t = ORIGIN[role]; tr = d.D[:, :t].astype(float); wd = d.wday[:t][-364:]
    prof = np.array([tr[:, -364:][:, wd == w].sum(1) for w in range(1, 8)]).T; prof = prof / np.maximum(prof.mean(1, keepdims=True), 1e-9)
    return tr[:, -28:].mean(1, keepdims=True) * prof[:, d.wday[t:t + HZ] - 1]
P = {"weekday_profile_baseline": {r: weekday_profile(r) for r in ("choose", "report")}}
for v in V: P[v] = {r: np.load(f"{a.preds}/{v}_{r}.npy").astype(float) for r in ("choose", "report")}
rows = []
for v, pr in P.items():
    for r in ("choose", "report"):
        tot, per = sc[r].score(pr[r], per_level=True); rows.append(dict(variant=v, role=r, wrmsse=tot, **{f"level{i}": per[i] for i in range(12)}))
pd.DataFrame(rows).round(4).to_csv(f"{a.out}/per_level.csv", index=False)
# error by horizon day: WRMSSE computed from day h alone
rows = []
for v, pr in P.items():
    e = (sc["report"].S @ (sc["report"].y - pr["report"])) ** 2; r = np.sqrt(e / sc["report"].scale[:, None])
    for h in range(HZ):
        per = [np.sum(sc["report"].w[sc["report"].lvl == l] * r[sc["report"].lvl == l, h]) for l in range(12)]; rows.append(dict(variant=v, day=h + 1, wrmsse_day=np.mean(per)))
pd.DataFrame(rows).round(4).to_csv(f"{a.out}/per_horizon.csv", index=False)
# bias of the total forecast per day (sum of forecasts / sum of actuals)
rows = []
for v, pr in P.items():
    for h in range(HZ): rows.append(dict(variant=v, day=h + 1, ratio=pr["report"][:, h].sum() / sc["report"].y[:, h].sum()))
pd.DataFrame(rows).round(4).to_csv(f"{a.out}/bias_total.csv", index=False)
# bias and share of squared error by sales-volume decile of the item-store series (volume = last 28 known days)
vol = d.D[:, ORIGIN["report"] - 28:ORIGIN["report"]].sum(1); dec = pd.qcut(pd.Series(vol).rank(method="first"), 10, labels=False).to_numpy()
rows = []
for v, pr in P.items():
    err = ((sc["report"].y - pr["report"]) ** 2 / sc["report"].scale[d.lvl == 11][:, None]).sum(1)
    for k in range(10):
        m = dec == k; rows.append(dict(variant=v, decile=k, ratio=pr["report"][m].sum() / max(sc["report"].y[m].sum(), 1e-9), share_scaled_sq_error=err[m].sum() / err.sum()))
pd.DataFrame(rows).round(4).to_csv(f"{a.out}/bias_by_volume.csv", index=False)
# trees fitted per store (early stopping on the choose role) from the run log
line = [l for l in open(a.log) if "V1/V2 trained" in l][0]; tup = re.findall(r"\((\d+), (\d+), (\d+), (\d+), (\d+)\)", line)
pd.DataFrame(tup, columns=["store", "v1_trees", "v2_trees", "v1_secs", "v2_secs"]).astype(int).to_csv(f"{a.out}/store_iterations.csv", index=False)
print(pd.read_csv(f"{a.out}/per_level.csv")[["variant", "role", "wrmsse"]].to_string(index=False))
