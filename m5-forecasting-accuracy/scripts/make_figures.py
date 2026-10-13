"""Draw the six figures from the CSV files in results/ (run scripts/analyze.py first). Run: python scripts/make_figures.py"""
import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, pandas as pd

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"); F = os.path.join(R, "..", "figures"); os.makedirs(F, exist_ok=True)
NAMES = {"weekday_profile_baseline": "baseline", "V1_lgb_squared": "V1", "V2_lgb_tweedie": "V2", "V3_lgb_tweedie_reconciled": "V3", "V4_lagfree_aligned": "V4", "V5_hl_mixer_ens": "V5", "V6_hurdle_mixer_ens": "V6"}
COL = {"baseline": "#777777", "V1": "#1f77b4", "V2": "#2ca02c", "V3": "#9467bd", "V4": "#d62728", "V5": "#ff7f0e", "V6": "#8c564b"}
pl = pd.read_csv(f"{R}/per_level.csv"); pl["v"] = pl.variant.map(NAMES)

fig, ax = plt.subplots(figsize=(6.4, 3.8)); x = np.arange(len(COL)); w = 0.38
for i, role in enumerate(("choose", "report")):
    s = pl[pl.role == role].set_index("v").loc[list(COL)]["wrmsse"]; b = ax.bar(x + (i - .5) * w, s, w, label=role, color=["#8ab6d6" if i == 0 else "#1f4e79"] * len(COL))
    for xx, v in zip(x + (i - .5) * w, s): ax.text(xx, v + .008, f"{v:.3f}", ha="center", fontsize=7)
ax.set_xticks(x); ax.set_xticklabels(list(COL)); ax.set_ylabel("WRMSSE (lower is better)"); ax.set_ylim(0.5, 0.88); ax.legend(); ax.set_title("Choose and report window per variant")
fig.tight_layout(); fig.savefig(f"{F}/fig1_choose_vs_report.png", dpi=140); plt.close(fig)

fig, ax = plt.subplots(figsize=(6.4, 3.8)); rep = pl[pl.role == "report"].set_index("v")
for v in COL: ax.plot(range(1, 13), rep.loc[v, [f"level{i}" for i in range(12)]].to_numpy(float), marker="o", ms=3, label=v, color=COL[v])
ax.set_xlabel("aggregation level (1 = total, 12 = item x store)"); ax.set_ylabel("level WRMSSE, report window"); ax.legend(ncol=6, fontsize=7); ax.set_title("Error by aggregation level")
fig.tight_layout(); fig.savefig(f"{F}/fig2_error_by_level.png", dpi=140); plt.close(fig)

ph = pd.read_csv(f"{R}/per_horizon.csv"); ph["v"] = ph.variant.map(NAMES); fig, ax = plt.subplots(figsize=(6.4, 3.8))
for v in COL: s = ph[ph.v == v]; ax.plot(s.day, s.wrmsse_day, label=v, color=COL[v], lw=1.5)
ax.set_xlabel("forecast day"); ax.set_ylabel("WRMSSE from that day alone"); ax.legend(ncol=6, fontsize=7); ax.set_title("Error by horizon day (report window)")
fig.tight_layout(); fig.savefig(f"{F}/fig3_error_by_horizon.png", dpi=140); plt.close(fig)

bt = pd.read_csv(f"{R}/bias_total.csv"); bt["v"] = bt.variant.map(NAMES); fig, ax = plt.subplots(figsize=(6.4, 3.8))
for v in COL: s = bt[bt.v == v]; ax.plot(s.day, s.ratio, label=v, color=COL[v], lw=1.5)
ax.axhline(1, color="k", lw=.6); ax.set_xlabel("forecast day"); ax.set_ylabel("sum of forecasts / sum of actuals"); ax.legend(ncol=6, fontsize=7); ax.set_title("Bias of the total forecast")
fig.tight_layout(); fig.savefig(f"{F}/fig4_total_bias.png", dpi=140); plt.close(fig)

bv = pd.read_csv(f"{R}/bias_by_volume.csv"); bv["v"] = bv.variant.map(NAMES); fig, ax = plt.subplots(figsize=(6.4, 3.8))
for v in COL: s = bv[bv.v == v]; ax.plot(s.decile + 1, s.ratio, marker="o", ms=3, label=v, color=COL[v])
ax.axhline(1, color="k", lw=.6); ax.set_xlabel("sales-volume decile of the item-store series (10 = highest)"); ax.set_ylabel("sum of forecasts / sum of actuals"); ax.legend(ncol=6, fontsize=7)
ax.set_title("Bias by sales volume (report window)"); fig.tight_layout(); fig.savefig(f"{F}/fig5_bias_by_volume.png", dpi=140); plt.close(fig)

it = pd.read_csv(f"{R}/store_iterations.csv"); fig, ax = plt.subplots(figsize=(6.4, 3.8)); x = np.arange(len(it))
ax.bar(x - .2, it.v1_trees, .4, label="V1 squared loss", color=COL["V1"]); ax.bar(x + .2, it.v2_trees, .4, label="V2 Tweedie loss", color=COL["V2"])
ax.set_xticks(x); ax.set_xticklabels([f"store {s}" for s in it.store], rotation=45, fontsize=7); ax.set_ylabel("trees kept by early stopping (choose role)"); ax.legend(); ax.set_title("Boosting rounds per store")
fig.tight_layout(); fig.savefig(f"{F}/fig6_trees_per_store.png", dpi=140); plt.close(fig); import glob, json
fig, axs = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
for f in sorted(glob.glob(f"{R}/gpu/V[56]_*_s[0-9]_history.json")):
    h = pd.DataFrame(json.load(open(f))); v = "V5" if "V5" in f else "V6"
    axs[0].plot(h.step, h.choose, color=COL[v], alpha=.7, lw=1.2); axs[1].plot(h.step, h.report_monitor, color=COL[v], alpha=.7, lw=1.2)
for a, t in zip(axs, ("choose window (used for stopping)", "report window (monitor only)")): a.set_title(t); a.set_xlabel("training step"); a.axhline(0.716, color="k", lw=.6, ls=":")
axs[0].set_ylabel("WRMSSE"); axs[0].set_ylim(0.6, 1.2); axs[0].plot([], [], color=COL["V5"], label="V5 members"); axs[0].plot([], [], color=COL["V6"], label="V6 members"); axs[0].legend(fontsize=7)
fig.tight_layout(); fig.savefig(f"{F}/fig7_neural_training.png", dpi=140); plt.close(fig); print(sorted(os.listdir(F)))
