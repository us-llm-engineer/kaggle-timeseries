"""Regenerate every figure in figures/ from the logs and CSVs in results/ (no data or GPU needed).

Usage (from rogii-wellbore-geology/): python scripts/make_figures.py
"""
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "figures"
FIG.mkdir(exist_ok=True)
WCOL = {16: "tab:blue", 24: "tab:orange", 32: "tab:green"}
plt.rcParams.update({"figure.dpi": 120, "axes.grid": True, "grid.alpha": 0.3, "font.size": 9})


def jl(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def fold_logs():
    rows = []
    for f in sorted(glob.glob(str(RES / "training" / "train_fold*.jsonl"))):
        fold = int(Path(f).stem.replace("train_fold", ""))
        rows += [{**r, "fold": fold} for r in jl(f)]
    return pd.DataFrame(rows)


def full_logs():
    return pd.DataFrame(sum([jl(f) for f in sorted(glob.glob(str(RES / "training" / "full_w*.jsonl")))], []))


def save(fig, name):
    fig.tight_layout(); fig.savefig(FIG / name, bbox_inches="tight"); plt.close(fig); print(FIG / name)


# ------------------------------------------------------------------ in-training
def fig_heldout_curves(df):
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.4), sharey=True)
    for ax, w in zip(axs, (16, 24, 32)):
        for fold, g in df[df.width == w].groupby("fold"):
            ax.plot(g.epoch, g.heldout_row_rmse_monitor, marker="o", ms=3, label=f"fold {fold}")
        ax.set_title(f"width {w}: held-out row RMSE per epoch"); ax.set_xlabel("epoch")
    axs[0].set_ylabel("RMSE (ft)"); axs[0].legend(fontsize=7)
    save(fig, "train_heldout_rmse.png")


def fig_loss(df, full):
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.4))
    for w, g in df.groupby("width"):
        m = g.groupby("epoch").train_loss.agg(["mean", "min", "max"])
        axs[0].plot(m.index, m["mean"], color=WCOL[w], label=f"w{w}")
        axs[0].fill_between(m.index, m["min"], m["max"], color=WCOL[w], alpha=0.2)
    axs[0].set_title("fold runs: training loss (mean, min-max over 5 folds)")
    for w, g in full.groupby("width"):
        axs[1].plot(g.epoch, g.train_loss, marker="o", ms=3, color=WCOL[w], label=f"w{w}")
    axs[1].axvspan(6.5, 12.5, color="grey", alpha=0.12, label="snapshot epochs 7-12")
    axs[1].set_title("final models (all 770 wells): training loss")
    for ax in axs:
        ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.legend(fontsize=7)
    save(fig, "train_loss.png")


def fig_grad_lr(df):
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.4))
    for w, g in df.groupby("width"):
        m = g.groupby("epoch").grad_norm.mean(); axs[0].plot(m.index, m.values, color=WCOL[w], marker="o", ms=3, label=f"w{w}")
    axs[0].set_title("mean pre-clip gradient norm per epoch (update clipped to 1.0)"); axs[0].set_yscale("log")
    g = df[(df.width == 16) & (df.fold == 1)]
    axs[1].plot(g.epoch, g.lr, marker="o", ms=3, color="k"); axs[1].set_title("learning rate (one-cycle, end of epoch)")
    for ax in axs:
        ax.set_xlabel("epoch"); ax.legend(fontsize=7) if ax is axs[0] else None
    save(fig, "train_grad_lr.png")


def fig_calibration(df):
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.4))
    for w, g in df.groupby("width"):
        m = g.groupby("epoch")[["temp", "heldout_mean_sd"]].mean()
        axs[0].plot(m.index, m.temp, color=WCOL[w], marker="o", ms=3, label=f"w{w}")
        axs[1].plot(m.index, m.heldout_mean_sd, color=WCOL[w], marker="o", ms=3, label=f"w{w}")
    axs[0].set_title("learned softmax temperature"); axs[1].set_title("held-out mean posterior s.d. of the offset (ft)")
    for ax in axs:
        ax.set_xlabel("epoch"); ax.legend(fontsize=7)
    save(fig, "train_calibration.png")


def fig_throughput(df, full):
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.4))
    a = pd.concat([df.assign(kind="fold"), full.assign(kind="full")])
    for i, col in enumerate(["samples_per_s", "gpu_mem_gb"]):
        s = a[a.epoch > 1].groupby(["width", "kind"])[col].median().unstack()
        s.plot.bar(ax=axs[i], rot=0, color=["tab:gray", "tab:purple"])
        axs[i].set_title({"samples_per_s": "training throughput (wells/s, median, epoch>1)",
                          "gpu_mem_gb": "peak GPU memory (GB)"}[col]); axs[i].set_xlabel("U-Net width")
    save(fig, "train_throughput_memory.png")


# ------------------------------------------------------------------ post-training
def last(path, role):
    return [r for r in jl(path) if r.get("role") == role][-1]


def fig_methods():
    B = RES / "baselines"; st = jl(RES / "stack" / "stack_log.jsonl")[-1]["folds2to5"]
    report = [("carry-forward", st["carry"]),
              ("grid tracker V1", last(B / "tracker_v1_report.jsonl", "report_oof_folds2to5")["pooled_rmse"]),
              ("grid tracker V2", last(B / "tracker_v2_report.jsonl", "report_oof_folds2to5")["pooled_rmse"]),
              ("coarsened tracker", last(B / "coarsened_tracker_report.jsonl", "report_oof_folds2to5")["pooled_rmse"]),
              ("A: grid U-Net blend", st["A_blend"]), ("B: A + GBM margin", st["B_margin"]),
              ("gated A/B", st["gated"]), ("gated + scale dir.", st["final"])]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    names, vals = zip(*report)
    cols = ["tab:gray"] + ["tab:olive"] * 3 + ["tab:blue", "tab:green", "tab:cyan", "tab:cyan"]
    bars = ax.barh(names, vals, color=cols); ax.invert_yaxis()
    for b, v in zip(bars, vals):
        ax.text(v + 0.1, b.get_y() + b.get_height() / 2, f"{v:.2f}", va="center", fontsize=8)
    ax.set_xlabel("pooled row RMSE, out-of-fold, folds 2-5 (ft)"); ax.set_title("methods on the report folds (lower is better)")
    save(fig, "post_method_comparison.png")


def fig_per_fold():
    rows = [r for r in jl(RES / "stack" / "stack_log.jsonl") if "fold" in r]
    df = pd.DataFrame(rows).set_index("fold")[["carry", "A_blend", "B_margin", "gated", "final"]]
    ax = df.plot.bar(rot=0, figsize=(9, 3.4), color=["tab:gray", "tab:blue", "tab:green", "tab:cyan", "tab:purple"])
    ax.set_ylabel("pooled RMSE (ft)"); ax.set_title("stack stages per fold (fold 1 = choose role, 2-5 = report)")
    ax.legend(fontsize=7, ncol=5)
    save(ax.figure, "post_stack_per_fold.png")


def fig_per_well():
    d = pd.read_csv(RES / "stack" / "final_per_well.csv")
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
    axs[0].scatter(d.carry_rmse, d.rmse, s=6, alpha=0.5)
    lim = [0.5, max(d.carry_rmse.max(), d.rmse.max()) * 1.1]
    axs[0].plot(lim, lim, "k--", lw=1); axs[0].set_xscale("log"); axs[0].set_yscale("log")
    axs[0].set_xlabel("carry-forward RMSE per well (ft)"); axs[0].set_ylabel("stack RMSE per well (ft)")
    axs[0].set_title(f"per well: {int((d.rmse < d.carry_rmse).sum())}/{len(d)} wells beat carry-forward")
    axs[1].hist(d.rmse, bins=np.logspace(-0.5, 2.2, 40), alpha=0.6, label="stack")
    axs[1].hist(d.carry_rmse, bins=np.logspace(-0.5, 2.2, 40), alpha=0.6, label="carry-forward")
    axs[1].set_xscale("log"); axs[1].set_xlabel("RMSE per well (ft)"); axs[1].legend(); axs[1].set_title("distribution over 770 wells")
    save(fig, "post_per_well.png")


def fig_horizon():
    h = pd.read_csv(RES / "stack" / "horizon_rmse.csv")
    x = (h.rows_after_cut_lo + h.rows_after_cut_hi) / 2
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.plot(x, h.carry_rmse, "o-", color="tab:gray", ms=3, label="carry-forward")
    ax.plot(x, h.final_rmse, "o-", color="tab:purple", ms=3, label="stack")
    ax2 = ax.twinx(); ax2.bar(x, h.points, width=200, alpha=0.12, color="k"); ax2.set_ylabel("rows in bin"); ax2.grid(False)
    ax.set_xlabel("rows after the last known TVT"); ax.set_ylabel("RMSE (ft)"); ax.legend(loc="upper left")
    ax.set_title("error grows with the distance from the cut (all 5 folds, out-of-fold)")
    save(fig, "post_horizon.png")


def fig_missing():
    d = pd.read_csv(RES / "stack" / "final_per_well.csv")
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.scatter(d.gr_missing, d.rmse - d.carry_rmse, s=6, alpha=0.5)
    ax.axhline(0, color="k", lw=1); ax.set_xlabel("fraction of missing GR after the cut")
    ax.set_ylabel("stack RMSE - carry RMSE (ft)"); ax.set_title("gain over carry-forward vs GR missingness")
    save(fig, "post_gr_missing.png")


def fig_cv_lb():
    lb = pd.read_csv(RES / "leaderboard.csv")
    fig, ax = plt.subplots(figsize=(6, 3.6))
    x = np.arange(len(lb)); wd = 0.2
    for i, (col, lab) in enumerate([("oof_rmse_all_folds", "OOF, all folds"), ("oof_rmse_folds2to5", "OOF, folds 2-5"),
                                    ("public_rmse", "public LB"), ("private_rmse", "private LB")]):
        b = ax.bar(x + (i - 1.5) * wd, lb[col], wd, label=lab)
        ax.bar_label(b, fmt="%.2f", fontsize=7)
    ax.set_xticks(x, ["A: blend", "B: blend + margin"]); ax.set_ylabel("RMSE (ft)"); ax.set_ylim(0, 15)
    ax.legend(fontsize=7, ncol=2); ax.set_title("local validation vs leaderboard")
    save(fig, "post_cv_vs_leaderboard.png")


if __name__ == "__main__":
    df, full = fold_logs(), full_logs()
    fig_heldout_curves(df); fig_loss(df, full); fig_grad_lr(df); fig_calibration(df); fig_throughput(df, full)
    fig_methods(); fig_per_fold(); fig_per_well(); fig_horizon(); fig_missing(); fig_cv_lb()
