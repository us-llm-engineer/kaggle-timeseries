"""Draw every figure from the files in results/. Run from the project root: python scripts/make_figures.py"""
import json, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

R, F = "results", "figures"
os.makedirs(F, exist_ok=True)
C = {"v1": "#4c72b0", "v2": "#8c8c8c", "v3": "#dd8452", "v4": "#55a868", "blend": "#c44e52"}
NAME = {"v1": "V1 LightGBM-L1", "v2": "V2 LightGBM-Tweedie", "v3": "V3 TiDE capacity-relative", "v4": "V4 TiDE window-normalised", "blend": "Mean of V1+V3+V4"}
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": .25, "axes.spines.top": False, "axes.spines.right": False})
J = lambda p: json.load(open(p))


# ---------- 0. architecture -------------------------------------------------------------
def box(ax, x, y, w, h, text, fc="#eef3fa", ec="#4c72b0", fs=8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06", fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def arrow(ax, a, b):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=10, lw=1.1, color="#444"))


def line(ax, pts):
    for p0, p1 in zip(pts[:-2], pts[1:-1]):
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color="#444", lw=1.1)
    arrow(ax, pts[-2], pts[-1])


fig, ax = plt.subplots(figsize=(13, 6.4)); ax.set_xlim(0, 13); ax.set_ylim(0, 6.4); ax.axis("off")
box(ax, .2, 4.7, 2.5, .9, "look-back target\n168 h ending D-2 23:00\n(V3: y / capacity, V4: window\nmean/std + 2 scalars)", "#fdf0e3", "#dd8452")
box(ax, .2, 3.4, 2.5, .9, "known-future weather (15)\n+ calendar sin/cos (6)\nfor the 24 target hours", "#e9f5ec", "#55a868")
box(ax, .2, 2.1, 2.5, .9, "static: series id (138) emb 16,\ncounty emb 4, business flag,\nproduct type, is_consumption", "#f2eaf7", "#8172b3")
box(ax, .2, .8, 2.5, .9, "log capacity (V3) /\nlog mean, log std (V4)", "#f0f0f0", "#888")
box(ax, 3.3, 3.2, 2.1, 1.0, "feature projection\nResidual block\n(21 -> 64 -> 4) per hour", "#e9f5ec", "#55a868")
box(ax, 6.0, 2.3, 2.2, 2.0, "dense encoder\n2 x Residual block\nhidden 1024\n(concat: look-back, 24x4\nprojected covariates,\nstatic, extra)")
box(ax, 8.8, 2.3, 1.8, 2.0, "dense decoder\nResidual block\n1024 -> 24 x 8")
box(ax, 11.0, 3.2, 1.9, 1.0, "temporal decoder\nResidual block\n(8 + 4 -> 64 -> 1)\nper target hour", "#e9f5ec", "#55a868")
box(ax, 6.0, 4.8, 2.2, .9, "global linear skip\nlook-back (168) -> 24 h", "#fdf0e3", "#dd8452")
box(ax, 8.8, 4.8, 4.1, .9, "sum -> x capacity (V3)  |  x std + mean (V4)\n-> clamp at 0 -> kWh", "#fbe9e9", "#c44e52")
arrow(ax, (2.7, 5.15), (6.0, 5.25))
arrow(ax, (2.7, 4.95), (6.0, 4.0))
arrow(ax, (2.7, 3.85), (3.3, 3.7)); arrow(ax, (5.4, 3.7), (6.0, 3.6))
arrow(ax, (2.7, 2.55), (6.0, 3.0)); arrow(ax, (2.7, 1.25), (6.0, 2.5))
arrow(ax, (8.2, 3.3), (8.8, 3.3)); arrow(ax, (10.6, 3.3), (11.0, 3.6))
line(ax, [(4.35, 3.2), (4.35, 1.9), (11.95, 1.9), (11.95, 3.2)])
arrow(ax, (11.95, 4.2), (11.95, 4.8)); arrow(ax, (8.2, 5.25), (8.8, 5.25))
ax.text(6.5, 6.15, "TiDE-style day-ahead forecaster: one sample = (series, day D) -> 24 hourly values; loss = L1 on raw kWh (= the competition MAE)",
        ha="center", fontsize=10, weight="bold")
ax.text(.2, .1, "Trained as a batched ensemble: E = 8 independent members (dropout 0.5 and 0.2 x 4 seeds) in one process; each member has its own weights, "
        "data order, dropout mask and early stopping.", fontsize=8, color="#444")
fig.savefig(f"{F}/arch_tide.png", dpi=150, bbox_inches="tight"); plt.close(fig)

# ---------- 1. in-training: DL members ---------------------------------------------------
fig, axs = plt.subplots(2, 2, figsize=(12, 7.5))
for j, v in enumerate(("v3", "v4")):
    h = J(f"{R}/dl/run2_amp/{v}/{v}_history.json"); rows = h["rows"]; E = len(rows[0]["train"])
    ep = [r["epoch"] for r in rows]
    tr = np.array([r["train"] for r in rows]); ch = np.array([r["choose_mae"] for r in rows]); rp = np.array([r["report_mae_monitor_only"] for r in rows])
    ax = axs[0, j]
    for e in range(E):
        ax.plot(ep, ch[:, e], color=C[v], alpha=.35, lw=1)
    ax.plot(ep, ch.mean(1), color=C[v], lw=2, label="choose MAE (mean of 8 members)")
    ax.plot(ep, rp.mean(1), color="k", lw=1.2, ls="--", label="report MAE (monitor only, never used to stop)")
    for e in range(E):
        se = next((r["epoch"] for r in rows if not r["active"][e]), None)
        if se:
            ax.axvline(se, color="#888", lw=.6, alpha=.6)
    ax.set_title(f"{NAME[v]}: held-out MAE (grey = member stops)", fontsize=9); ax.set_xlabel("epoch"); ax.set_ylabel("MAE (kWh)"); ax.set_ylim(40, 120); ax.legend(fontsize=7)
    ax = axs[1, j]
    ax.plot(ep, tr.mean(1), color=C[v]); ax.set_ylabel("train L1 (scaled)", color=C[v]); ax.set_xlabel("epoch")
    ax2 = ax.twinx(); ax2.plot(ep, [r["lr"] for r in rows], color="#888", lw=1.2); ax2.set_ylabel("learning rate (cosine)", color="#888"); ax2.grid(False)
    best = np.array(h["best_choose"]); ax.set_title(f"train loss + LR; best choose MAE {best.min():.1f}-{best.max():.1f}; stop: {', '.join(sorted(set(h['stop_reason'])))}", fontsize=9)
fig.tight_layout(); fig.savefig(f"{F}/train_dl_curves.png", dpi=150); plt.close(fig)

# ---------- 2. in-training: LightGBM -----------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
for nm, c in (("c0", "production"), ("c1", "consumption")):
    h = J(f"{R}/backtest/v1/v1_{nm}_history.json"); r = pd.DataFrame(h["rows"])
    axs[0].plot(r.iter, r.fit_l1, label=f"{c} fit", color=C["v1"], ls="-" if nm == "c0" else "--")
    axs[0].plot(r.iter, r.choose_l1, label=f"{c} choose", color=C["v3"], ls="-" if nm == "c0" else "--")
    axs[0].axvline(h["best_iteration"], color="#888", lw=.8)
axs[0].set_yscale("log"); axs[0].set_xlabel("boosting round"); axs[0].set_ylabel("capacity-relative L1 (weighted)"); axs[0].legend(fontsize=7)
axs[0].set_title("V1: fit vs choose loss; vertical = early-stopped round (61 / 538)")
for g in ("c0_b0", "c0_b1", "c1_b0", "c1_b1"):
    r = pd.DataFrame(J(f"{R}/backtest/v2/v2_{g}_history.json")["rows"])
    col = "choose_tweedie" if "choose_tweedie" in r else [c for c in r.columns if c.startswith("choose")][0]
    axs[1].plot(r.iter, r[col], label=g)
axs[1].set_yscale("log"); axs[1].set_xlabel("boosting round"); axs[1].legend(fontsize=7); axs[1].set_title("V2 (Tweedie, no target lags): choose loss per group")
fig.tight_layout(); fig.savefig(f"{F}/train_gbdt_curves.png", dpi=150); plt.close(fig)

# ---------- 3. in-training: GPU efficiency -----------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(12, 3.8))
for nm, lab in (("run1_fp32", "run 1 (fp32)"), ("run2_amp", "run 2 (fp16 autocast)")):
    s = J(f"{R}/dl/{nm}/gpu_utilisation.json"); t = np.array([x["t"] for x in s]); t = t - t[0]; u = np.array([x["util"] for x in s])
    axs[0].plot(t, u, lw=1, label=f"{lab}: mean {u.mean():.0f}%, {100 * (u >= 80).mean():.0f}% of samples >= 80%")
axs[0].set_xlabel("seconds"); axs[0].set_ylabel("GPU utilisation (%)"); axs[0].legend(fontsize=7); axs[0].set_title("Tesla T4 utilisation sampled every 5 s")
s = J(f"{R}/dl/run2_amp/gpu_utilisation.json"); axs[1].plot([x["t"] - s[0]["t"] for x in s], [x["mem_mib"] for x in s], color="#8172b3")
axs[1].axhline(s[0]["mem_total_mib"], color="#c44e52", ls="--"); axs[1].set_ylabel("memory used (MiB)"); axs[1].set_xlabel("seconds")
axs[1].set_title("GPU memory: about 2.4 GB of 15.4 GB (one batched process)")
fig.tight_layout(); fig.savefig(f"{F}/train_gpu_efficiency.png", dpi=150); plt.close(fig)

# ---------- 4. post: method comparison ---------------------------------------------------
T = pd.read_csv(f"{R}/post/by_target.csv").set_index("is_consumption")
fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
base = {"lag-168": 105.3, "lag-48": 123.8}
ks = ["v2", "v1", "v3", "v4", "blend"]
vals = {k: float(((T[f"mae_{k}"] * T.rows).sum()) / T.rows.sum()) for k in ks}
bars = axs[0].bar(list(base) + [NAME[k] for k in ks], list(base.values()) + [vals[k] for k in ks], color=["#bbb", "#bbb"] + [C[k] for k in ks])
for r_, v_ in zip(bars, list(base.values()) + [vals[k] for k in ks]):
    axs[0].text(r_.get_x() + r_.get_width() / 2, v_ + 1, f"{v_:.1f}", ha="center", fontsize=8)
axs[0].set_ylabel("report-window MAE (kWh)"); axs[0].tick_params(axis="x", rotation=30); axs[0].set_title("Overall MAE, 2023-04-02 .. 2023-05-31")
x = np.arange(2); w = .16
for i, k in enumerate(ks):
    axs[1].bar(x + (i - 2) * w, [T.loc[0, f"mae_{k}"], T.loc[1, f"mae_{k}"]], w, color=C[k], label=NAME[k])
axs[1].set_xticks(x); axs[1].set_xticklabels(["production", "consumption"]); axs[1].set_ylabel("MAE (kWh)"); axs[1].legend(fontsize=7); axs[1].set_title("Per target")
fig.tight_layout(); fig.savefig(f"{F}/post_method_comparison.png", dpi=150); plt.close(fig)

# ---------- 5. post: error by hour (production) and bias --------------------------------
H = pd.read_csv(f"{R}/post/by_hour.csv")
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
P = H[H.is_consumption == 0]
for k in ("v1", "v3", "v4", "blend"):
    axs[0].plot(P.hour, P[f"mae_{k}"], color=C[k], label=NAME[k], lw=2 if k == "blend" else 1.3)
axs[0].set_title("Production: MAE by hour of day"); axs[0].set_xlabel("hour"); axs[0].set_ylabel("MAE (kWh)"); axs[0].legend(fontsize=7)
for k in ("v1", "v3", "v4", "blend"):
    axs[1].plot(P.hour, P[f"bias_{k}"], color=C[k], label=NAME[k])
axs[1].axhline(0, color="k", lw=.6); axs[1].set_title("Production: mean error (prediction - actual) by hour"); axs[1].set_xlabel("hour"); axs[1].set_ylabel("bias (kWh)")
fig.tight_layout(); fig.savefig(f"{F}/post_error_by_hour.png", dpi=150); plt.close(fig)

# ---------- 6. post: weekly error (the capacity shift) -------------------------------------
W = pd.read_csv(f"{R}/post/by_week.csv", parse_dates=["week"])
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
for i, (c, t) in enumerate(((0, "production"), (1, "consumption"))):
    d = W[W.is_consumption == c]
    for k in ("v1", "v3", "v4", "blend"):
        axs[i].plot(d.week, d[f"mae_{k}"], marker="o", ms=3, color=C[k], label=NAME[k])
    axs[i].set_title(f"{t}: weekly MAE"); axs[i].tick_params(axis="x", rotation=30); axs[i].set_ylabel("MAE (kWh)")
axs[0].legend(fontsize=7)
fig.tight_layout(); fig.savefig(f"{F}/post_weekly.png", dpi=150); plt.close(fig)

# ---------- 7. post: per-unit error against size ----------------------------------------
U = pd.read_csv(f"{R}/post/by_unit.csv")
fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
for i, (c, t) in enumerate(((0, "production"), (1, "consumption"))):
    d = U[U.is_consumption == c]
    axs[i].scatter(d.mean_abs_y, d.mae_blend, s=18, color=C["blend"], label="blend")
    axs[i].scatter(d.mean_abs_y, d.mae_v1, s=10, color=C["v1"], alpha=.6, label="V1")
    axs[i].set_xscale("log"); axs[i].set_yscale("log"); axs[i].set_xlabel("mean |actual| of the unit (kWh)"); axs[i].set_ylabel("unit MAE (kWh)")
    axs[i].set_title(f"{t}: error grows with unit size; labels = 5 largest units")
    for _, r in d.sort_values("sum_abs_y", ascending=False).head(5).iterrows():
        axs[i].annotate(int(r.prediction_unit_id), (r.mean_abs_y, r.mae_blend), fontsize=7)
axs[0].legend(fontsize=7)
fig.tight_layout(); fig.savefig(f"{F}/post_per_unit.png", dpi=150); plt.close(fig)

# ---------- 8. post: model disagreement ----------------------------------------------------
D = pd.read_csv(f"{R}/post/disagreement_by_hour.csv"); EC = pd.read_csv(f"{R}/post/error_correlation.csv")
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
for c, ls in ((0, "-"), (1, "--")):
    d = D[D.is_consumption == c]
    for col, lab in (("d13", "V1 vs V3"), ("d14", "V1 vs V4"), ("d34", "V3 vs V4")):
        axs[0].plot(d.hour, d[col], ls=ls, label=f"{lab} ({'prod' if c == 0 else 'cons'})")
axs[0].set_xlabel("hour"); axs[0].set_ylabel("mean |difference| (kWh)"); axs[0].legend(fontsize=6); axs[0].set_title("Disagreement between model families by hour")
cols = ["corr_err_v1_v3", "corr_err_v1_v4", "corr_err_v3_v4"]; x = np.arange(3)
axs[1].bar(x - .18, EC.loc[EC.is_consumption == 0, cols].values[0], .36, label="production")
axs[1].bar(x + .18, EC.loc[EC.is_consumption == 1, cols].values[0], .36, label="consumption")
axs[1].set_xticks(x); axs[1].set_xticklabels(["V1-V3", "V1-V4", "V3-V4"]); axs[1].set_ylim(0, 1); axs[1].set_ylabel("correlation of errors"); axs[1].legend(fontsize=7)
axs[1].set_title("Error correlation 0.68-0.85: the blend gain is modest")
fig.tight_layout(); fig.savefig(f"{F}/post_disagreement.png", dpi=150); plt.close(fig)
print(sorted(os.listdir(F)))
