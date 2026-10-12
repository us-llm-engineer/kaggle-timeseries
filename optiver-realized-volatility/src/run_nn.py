"""One neural worker (own process, own CUDA context): V3 or V4. Data lives on the GPU; batches are index gathers (no host sync in the step).
Per epoch: atomic history JSON + one log line + last/best checkpoints; time-based cosine LR; deadline stops starting epochs and leaves a usable result."""
import argparse, json, math, os, sys, time
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from opt14 import common as C, data as D, obs
from opt14.nn_models import V3, V4, pct_loss

ap = argparse.ArgumentParser()
ap.add_argument("--variant", required=True, choices=["v3", "v4"]); ap.add_argument("--prep", required=True)   # dir with features.parquet, seq.npy, train.csv
ap.add_argument("--out", required=True); ap.add_argument("--frac", type=float, default=0.425); ap.add_argument("--budget-min", type=float, default=10.0)
ap.add_argument("--max-epochs", type=int, default=400); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--patience", type=int, default=25)
ap.add_argument("--batch", type=int, default=0); ap.add_argument("--smoke", action="store_true")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True); os.makedirs(f"{a.out}/logs", exist_ok=True)
name = f"{a.variant}_s{a.seed}"; LOG = f"{a.out}/logs/{name}.log"
torch.manual_seed(a.seed); np.random.seed(a.seed)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if dev.type == "cuda":
    torch.cuda.set_per_process_memory_fraction(a.frac)
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
use_amp = dev.type == "cuda" and torch.cuda.get_device_capability()[0] >= 8      # native bf16 (L4 = 8.9); Turing and older fall back to fp32
amp = dict(device_type=dev.type, dtype=torch.bfloat16, enabled=use_amp)

# ---------------- data
df = D.load_table(f"{a.prep}/features.parquet", f"{a.prep}/train.csv")
cols = D.feature_columns(df, True)
prep = D.Prep().fit(df.loc[df.role == 0, [c for c in cols if c != "stock_id"]].to_numpy())
fc = [c for c in cols if c != "stock_id"]
X = torch.from_numpy(prep.transform(df[fc].to_numpy())).to(dev)
y = torch.from_numpy(df.target.to_numpy(np.float32)).to(dev)
log_ref = torch.log(torch.from_numpy(df.rv_all.to_numpy(np.float32)).clamp(min=1e-5)).to(dev)
stock = torch.from_numpy(df.stock_id.to_numpy(np.int64)).to(dev)
role = df.role.to_numpy()
if a.variant == "v3":
    keys = np.load(f"{a.prep}/seq_keys.npy"); seq_all = np.load(f"{a.prep}/seq.npy", mmap_mode="r")
    kdf = pd.DataFrame({"stock_id": keys[:, 0], "time_id": keys[:, 1], "i": np.arange(len(keys))})
    pos = df[["stock_id", "time_id"]].merge(kdf, on=["stock_id", "time_id"], how="left")["i"].to_numpy()
    assert not np.isnan(pos).any(), "sequence rows missing for some table rows"
    SEQ = torch.from_numpy(np.ascontiguousarray(seq_all[pos.astype(np.int64)])).to(dev)
n_feat = X.shape[1]
obs.log(LOG, f"{name} amp_bf16={use_amp} data on {dev}: X {tuple(X.shape)} roles {np.bincount(role)} gpu_mem {torch.cuda.memory_allocated()/2**20 if dev.type=='cuda' else 0:.0f}MB")

# time_id sets for V4 (up to 112 stocks each, padded)
tids = df.time_id.to_numpy()
if a.variant == "v4":
    order = np.argsort(tids, kind="stable"); ut, st, ct = np.unique(tids[order], return_index=True, return_counts=True)
    S = int(ct.max()); sets = np.full((len(ut), S), -1, np.int64)
    for i, (s0, c0) in enumerate(zip(st, ct)): sets[i, :c0] = order[s0:s0 + c0]
    set_role = role[order[st]]
    SETS = torch.from_numpy(sets).to(dev)

model = (V3(n_feat) if a.variant == "v3" else V4(n_feat)).to(dev)
lr0 = 2e-3 if a.variant == "v3" else 8e-4
opt = torch.optim.AdamW(model.parameters(), lr=lr0, weight_decay=1e-4 if a.variant == "v3" else 1e-2)
bs = a.batch or (2048 if a.variant == "v3" else 48)
fit_idx = np.where(role == 0)[0]; choose_idx = np.where(role == 1)[0]; report_idx = np.where(role == 2)[0]
fit_sets = np.where(set_role == 0)[0] if a.variant == "v4" else None
hist = obs.History(f"{a.out}/{name}_history.json", {"variant": name, "n_params": sum(p.numel() for p in model.parameters()), "n_feat": n_feat, "batch": bs,
                                                       "gpu_frac": a.frac, "budget_min": a.budget_min, "lr0": lr0})


def forward_rows(idx_t):
    if a.variant == "v3":
        return model(X[idx_t], SEQ[idx_t].float(), stock[idx_t], log_ref[idx_t])
    raise RuntimeError


def forward_sets(set_ids):
    ix = SETS[set_ids]; pad = ix < 0; ixc = ix.clamp(min=0)
    return model(X[ixc], stock[ixc], log_ref[ixc], pad), ix, pad


@torch.no_grad()
def predict(idx):
    model.eval(); out = np.zeros(len(df), np.float32); idx_t = torch.from_numpy(idx).to(dev)
    with torch.autocast(**amp):
        if a.variant == "v3":
            for i in range(0, len(idx), 8192):
                b = idx_t[i:i + 8192]; out[idx[i:i + 8192]] = torch.exp(forward_rows(b).float()).cpu().numpy()
        else:
            sel = np.unique(np.searchsorted(ut, tids[idx])); st_ = torch.from_numpy(sel).to(dev)
            for i in range(0, len(sel), 128):
                lp, ix, pad = forward_sets(st_[i:i + 128]); m = ~pad
                out[ix[m].cpu().numpy()] = torch.exp(lp.float())[m].cpu().numpy()
    model.train()
    return out[idx]


def step_loss(batch_ids):
    with torch.autocast(**amp):
        if a.variant == "v3":
            lp = forward_rows(batch_ids); return pct_loss(lp, y[batch_ids])
        lp, ix, pad = forward_sets(batch_ids); m = ~pad
        return pct_loss(lp[m], y[ix.clamp(min=0)][m])


dl = obs.Deadline(a.budget_min * 60); t0 = time.time()
best, best_ep, since, ema, ep, halvings = np.inf, 0, 0, None, 0, 0
ycs, yrs = df.target.to_numpy()[choose_idx], df.target.to_numpy()[report_idx]
fit_sub = np.random.default_rng(0).choice(fit_idx, size=min(40000, len(fit_idx)), replace=False)
stop = "max_epochs"
while ep < a.max_epochs:
    ep += 1; te = time.time(); model.train()
    n_units = len(fit_idx) if a.variant == "v3" else len(fit_sets)
    perm = np.random.permutation(fit_idx) if a.variant == "v3" else np.random.permutation(fit_sets)
    perm_t = torch.from_numpy(perm).to(dev); loss_sum = torch.zeros((), device=dev); nb = 0; gnorm = None
    try:
        for i in range(0, n_units, bs):
            frac_t = min(1.0, (time.time() - t0) / (a.budget_min * 60))
            for g in opt.param_groups: g["lr"] = lr0 * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * frac_t)))
            loss = step_loss(perm_t[i:i + bs])
            opt.zero_grad(set_to_none=True); loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); loss_sum += loss.detach(); nb += 1
    except torch.OutOfMemoryError:
        ep -= 1; bs = max(8, bs // 2); halvings += 1; torch.cuda.empty_cache(); obs.log(LOG, f"{name} OOM -> batch {bs} (halving {halvings})"); continue
    pc = predict(choose_idx); c = C.best_scalar(ycs, pc); cr = C.rmspe(ycs, c * pc)
    ps = predict(fit_sub); fr = C.rmspe(df.target.to_numpy()[fit_sub], C.best_scalar(df.target.to_numpy()[fit_sub], ps) * ps)
    pr = predict(report_idx); rr = C.rmspe(yrs, c * pr)                      # report role: monitor only, never used to choose
    ema = cr if ema is None else 0.6 * ema + 0.4 * cr
    if ema < best - 1e-5:
        best, best_ep, since = ema, ep, 0; torch.save({"model": model.state_dict(), "epoch": ep, "scalar": c}, f"{a.out}/{name}_best.pt")
    else: since += 1
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "epoch": ep}, f"{a.out}/{name}_last.pt")
    sec = time.time() - te; eta = max(0.0, dl.left())
    row = {"epoch": ep, "fit_loss": float(loss_sum / max(nb, 1)), "choose_rmspe": cr, "choose_rmspe_ema": ema, "fit_rmspe": fr, "gap": cr - fr, "report_rmspe_monitor_only": rr,
           "scalar": c, "lr": opt.param_groups[0]["lr"], "grad_norm": float(gnorm) if gnorm is not None else None, "since_best": since, "epoch_sec": round(sec, 2),
           "samples_per_s": round(n_units / sec), "gpu_peak_mb": round(torch.cuda.max_memory_allocated() / 2**20) if dev.type == "cuda" else 0, "halvings": halvings,
           "sec": round(time.time() - t0, 1), "budget_left_s": round(eta, 1), "batch": bs}
    hist.add(row)
    obs.log(LOG, f"{name} ep {ep} {100*min(1,(time.time()-t0)/(a.budget_min*60)):.0f}% choose {cr:.4f} (best-ema {best:.4f}@{best_ep}) fit {fr:.4f} gap {cr-fr:+.4f} lr {row['lr']:.2e} gn {row['grad_norm']:.2f} {row['samples_per_s']}/s peak {row['gpu_peak_mb']}MB left {eta:.0f}s")
    if since >= a.patience: stop = "early_stop"; break
    if dl.hit() or dl.left() < sec * 1.2: stop = "deadline"; break

# ---------------- final: best checkpoint -> predictions for choose + report
ck = torch.load(f"{a.out}/{name}_best.pt", map_location=dev); model.load_state_dict(ck["model"])
pc = predict(choose_idx); c = C.best_scalar(ycs, pc); pr = predict(report_idx) * c
out = pd.concat([pd.DataFrame({"stock_id": df.stock_id.to_numpy()[i], "time_id": df.time_id.to_numpy()[i], "target": df.target.to_numpy()[i], "pred": p, "role": r})
                 for i, p, r in ((choose_idx, c * pc, 1), (report_idx, pr, 2))])
out.to_parquet(f"{a.out}/{name}_predictions.parquet")
res = {"variant": name, "stop": stop, "best_epoch": best_ep, "epochs": ep, "scalar": c, "choose_rmspe": C.rmspe(ycs, c * pc), "report_rmspe": C.rmspe(yrs, pr), "seconds": round(time.time() - t0, 1), "halvings": halvings}
hist.meta.update(res); hist.flush(); obs.log(LOG, f"{name} FINAL {json.dumps(res)}")
