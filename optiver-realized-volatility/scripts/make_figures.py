"""Draw the six figures from the shipped result files only (no retraining, no competition rows).
Usage: python scripts/make_figures.py   (run from the project folder)"""
import glob, json, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R, F = "results", "figures"; os.makedirs(F, exist_ok=True)
COL = {"v1_est": "#8c8c8c", "v2_nocross": "#c9a227", "v2_cross": "#d95f02", "v3_ens": "#1b9e77", "v4_ens": "#7570b3"}
NAME = {"v1_est": "V1 ridge + estimators", "v2_nocross": "V2 LightGBM, per-stock", "v2_cross": "V2 LightGBM + cross-stock", "v3_ens": "V3 CNN (3 seeds)", "v4_ens": "V4 set transformer (3 seeds)"}

# 1: train vs choose gap per epoch / round, with the stop reason
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for a_, v in zip(ax[:2], ("v3", "v4")):
    for f in sorted(glob.glob(f"{R}/gpu/{v}_s*_history.json")):
        h = json.load(open(f)); r = h["rows"]; n = os.path.basename(f)[:5]
        a_.plot([x["epoch"] for x in r], [x["choose_rmspe"] for x in r], label=f"{n} choose (stop: {h['meta'].get('stop')}, best epoch {h['meta'].get('best_epoch')})")
        a_.plot([x["epoch"] for x in r], [x["fit_rmspe"] for x in r], "--", alpha=.5)
    a_.set_title(f"{v.upper()}: choose (solid) vs fit (dashed) RMSPE"); a_.set_xlabel("epoch"); a_.legend(fontsize=7); a_.set_ylim(0.17, 0.26)
for k in ("v2_cross", "v2_nocross"):
    h = json.load(open(f"{R}/cpu/{k}_history.json")); r = [x for x in h["rows"] if "round" in x]
    ax[2].plot([x["round"] for x in r], [x["choose_rmspe"] for x in r], label=f"{k} choose (stop: {h['meta'].get('stop')})")
    ax[2].plot([x["round"] for x in r], [x["fit_rmspe"] for x in r], "--", alpha=.5)
ax[2].set_title("V2: choose (solid) vs fit (dashed)"); ax[2].set_xlabel("boosting round"); ax[2].legend(fontsize=7); ax[2].set_ylim(0.15, 0.26)
fig.tight_layout(); fig.savefig(f"{F}/fig1_gap_per_epoch.png", dpi=130); plt.close(fig)

# 2, 3: error by stock volatility level and by market regime
for fn, src, title in (("fig2_error_by_stock_level.png", "by_stock_level", "RMSPE by stock volatility-level decile (report role)"),
                       ("fig3_error_by_market_regime.png", "by_regime", "RMSPE by market-wide volatility decile of the time_id (report role)")):
    d = pd.read_csv(f"{R}/post/{src}.csv"); col = d.columns[1]
    fig, a_ = plt.subplots(figsize=(7, 4))
    for m in COL:
        g = d[d.model == m]; a_.plot(g[col], g.rmspe, marker="o", color=COL[m], label=NAME[m])
    a_.set_title(title, fontsize=9); a_.set_xlabel("decile (0 = lowest)"); a_.set_ylabel("RMSPE"); a_.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(f"{F}/{fn}", dpi=130); plt.close(fig)

# 4: residual distribution and calibration
rh, cal = pd.read_csv(f"{R}/post/residual_hist.csv"), pd.read_csv(f"{R}/post/calibration.csv")
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for m in COL:
    g = rh[rh.model == m]; ax[0].step(g.bin_left, g["count"], where="post", color=COL[m], label=f"{NAME[m]} (mean {g.mean_log_ratio.iloc[0]:+.3f})")
    c = cal[cal.model == m]; ax[1].plot(c.mean_pred, c.mean_target, marker=".", color=COL[m], label=NAME[m])
lim = [cal.mean_pred.min(), cal.mean_pred.max()]; ax[1].plot(lim, lim, "k:", lw=1)
ax[0].set_title("log(prediction / target), report role"); ax[0].legend(fontsize=6); ax[1].set_title("calibration: mean target per prediction bin")
ax[1].set_xscale("log"); ax[1].set_yscale("log"); ax[1].set_xlabel("mean prediction"); ax[1].legend(fontsize=7)
fig.tight_layout(); fig.savefig(f"{F}/fig4_residuals_calibration.png", dpi=130); plt.close(fig)

# 5: member disagreement against error, and correlation of member predictions
dis, cor = pd.read_csv(f"{R}/post/disagreement.csv"), pd.read_csv(f"{R}/post/member_correlation.csv", index_col=0)
fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
ax[0].plot(dis["std"], dis.abs_err, marker="o"); ax[0].set_xlabel("std of log-prediction across members"); ax[0].set_ylabel("mean |log error| of the member mean"); ax[0].set_title("disagreement vs error (report role)")
im = ax[1].imshow(cor.to_numpy(), vmin=cor.to_numpy().min(), vmax=1)
ax[1].set_xticks(range(len(cor))); ax[1].set_xticklabels(cor.columns, rotation=70, fontsize=7); ax[1].set_yticks(range(len(cor))); ax[1].set_yticklabels(cor.index, fontsize=7)
ax[1].set_title("correlation of log-predictions"); fig.colorbar(im, ax=ax[1]); fig.tight_layout(); fig.savefig(f"{F}/fig5_member_disagreement.png", dpi=130); plt.close(fig)

# 6: attribution - LightGBM gain with and without cross-stock features
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
for a_, k in zip(ax, ("v2_cross", "v2_nocross")):
    imp = pd.read_csv(f"{R}/cpu/{k}_importance.csv", index_col=0).iloc[:, 0].sort_values(ascending=False).head(20)[::-1]
    a_.barh(imp.index, imp.values / imp.values.sum(), color=["#d95f02" if i.startswith("x_") or i == "stock_logrv_mean" else "#555" for i in imp.index])
    a_.set_title(f"{k}: top-20 gain share (orange = cross-stock)"); a_.tick_params(labelsize=7)
fig.tight_layout(); fig.savefig(f"{F}/fig6_feature_attribution.png", dpi=130); plt.close(fig)
print(sorted(os.listdir(F)))
