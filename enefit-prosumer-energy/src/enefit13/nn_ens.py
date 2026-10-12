"""Batched-ensemble TiDE: E independent members (seeds x dropout settings) trained in ONE process with batched matmuls.
Why: the cloud GPU host used here exposes a single vCPU, so separate worker processes cannot keep a T4 busy; one process with
member-batched kernels does. Members never share parameters or batches (own permutation, own dropout rate and mask).
Per-member early stopping (plateau / overfitting) keeps each member's best snapshot; training ends when every member stopped."""
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

from .nn import H, L, Runner, _atomic_json


class EL(nn.Module):
    """Ensemble linear: weight (E, i, o)."""

    def __init__(self, E, i, o):
        super().__init__()
        k = 1.0 / np.sqrt(i)
        self.w = nn.Parameter(torch.empty(E, i, o).uniform_(-k, k))
        self.b = nn.Parameter(torch.empty(E, 1, o).uniform_(-k, k))

    def forward(self, x):                                   # x (E,B,i)
        return torch.baddbmm(self.b, x, self.w)


class ERes(nn.Module):
    def __init__(self, E, i, h, o, keep, ln=True):
        super().__init__()
        self.f1, self.f2, self.sk = EL(E, i, h), EL(E, h, o), EL(E, i, o)
        self.register_buffer("keep", keep.clone())           # (E,1,1) per-member keep probability
        self.ln = ln and o > 1                                # LayerNorm over one feature would be a constant
        if self.ln:
            self.g, self.bb = nn.Parameter(torch.ones(E, 1, o)), nn.Parameter(torch.zeros(E, 1, o))

    def forward(self, x):
        h = self.f2(torch.relu(self.f1(x)))
        if self.training:
            h = h * (torch.rand_like(h) < self.keep).to(h.dtype) / self.keep
        y = (h + self.sk(x)).float()                          # statistics / residual sum in fp32 under autocast
        if self.ln:
            y = (y - y.mean(-1, keepdim=True)) / torch.sqrt(y.var(-1, unbiased=False, keepdim=True) + 1e-5) * self.g + self.bb
        return y


class EnsTiDE(nn.Module):
    def __init__(self, E, nw, nser, mode, drops, hid=1024, r=4, p=8, emb=16):
        super().__init__()
        keep = torch.tensor([1.0 - d for d in drops]).view(E, 1, 1)
        self.register_buffer("keep", keep)
        self.E, self.mode, self.p, self.r = E, mode, p, r
        self.ser_emb = nn.Parameter(torch.randn(E, nser, emb) * 0.1)
        self.cnt_emb = nn.Parameter(torch.randn(E, 16, 4) * 0.1)
        extra = 2 if mode == "raw" else 1
        enc_in = L + H * r + emb + 4 + 2 + extra
        self.proj = ERes(E, nw + 6, 64, r, keep)
        self.enc1, self.enc2 = ERes(E, enc_in, hid, hid, keep), ERes(E, hid, hid, hid, keep)
        self.dec = ERes(E, hid, hid, H * p, keep)
        self.tdec = ERes(E, p + r, 64, 1, keep, ln=False)
        self.skip = EL(E, L, H)

    def forward(self, lb, wx, cal, sid, static, extra):     # all tensors (E,B,...)
        E, B = lb.shape[:2]
        x = self.proj(torch.cat([wx, cal], -1).reshape(E, B * H, -1)).reshape(E, B, H, self.r)
        se = torch.gather(self.ser_emb, 1, sid.unsqueeze(-1).expand(-1, -1, self.ser_emb.shape[-1]))
        ce = torch.gather(self.cnt_emb, 1, static[..., 0].unsqueeze(-1).expand(-1, -1, 4))
        e = torch.cat([lb, x.reshape(E, B, -1), se, ce, static[..., 1:3].float(), extra], -1)
        d = self.dec(self.enc2(self.enc1(e))).reshape(E, B, H, self.p)
        out = self.tdec(torch.cat([d, x], -1).reshape(E, B * H, -1)).reshape(E, B, H)
        return out + self.skip(lb)


def _scaler(enabled):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except Exception:                                           # older torch (Kaggle image)
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _gather(data, idx):
    """idx (E,B) long -> dict of (E,B,...) tensors."""
    return {k: v[idx] for k, v in data.items()}


def train_ensemble(td, mode, device, drops, seeds, tr, ch, rp, out_dir, name, epochs=100, bs=512, lr=1e-3, budget_sec=None,
                   patience=12, min_rel=0.002, overfit_gap=0.05, overfit_run=5, min_epochs=20, amp=True, final=False):
    """drops/seeds define E = len(drops) * len(seeds) members (order: drop-major). Returns (preds_eval, info)."""
    os.makedirs(out_dir, exist_ok=True)
    members = [(d, s) for d in drops for s in seeds]
    E = len(members)
    torch.manual_seed(1234 + sum(seeds))
    helper = Runner(td, mode, device, seeds[0], out_dir, name)       # only used for materialize/_predict helpers
    dtr = helper.materialize(*tr)
    dch = helper.materialize(*ch) if (ch is not None and len(ch[0])) else None
    drp = helper.materialize(*rp) if (rp is not None and len(rp[0])) else None
    del helper.model
    n = dtr["lb"].shape[0]
    model = EnsTiDE(E, td["wx"].shape[-1], td["S"], mode, [m[0] for m in members]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    const = float(torch.nanmean(torch.abs(dtr["tg"][:, 0]))) + 1.0
    spe = int(np.ceil(n / bs))

    use_amp = bool(amp and device == "cuda")
    scaler = _scaler(use_amp)

    def fwd(b):
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            out = model(b["lb"], b["wx"], b["cal"], b["sid"], b["static"], b["extra"])
        out = out.float()
        return out * b["sc"] if mode == "rel" else out * b["std"].unsqueeze(-1) + b["mean"].unsqueeze(-1)

    def step(idx=None):
        if idx is None:
            idx = torch.randint(0, n, (E, bs), device=device)
        b = _gather(dtr, idx)
        pred = fwd(b)
        m = ~torch.isnan(b["tg"])
        per = (torch.abs(pred - torch.nan_to_num(b["tg"])) * m).sum((1, 2)) / m.sum((1, 2)).clamp(min=1) / const
        opt.zero_grad(set_to_none=True)
        scaler.scale(per.sum()).backward()
        scaler.unscale_(opt)
        # per-member gradient clipping (members are independent models)
        sq = torch.zeros(E, device=device)
        for p_ in model.parameters():
            if p_.grad is not None and p_.shape[0] == E:
                sq += p_.grad.pow(2).flatten(1).sum(1)
        scale = (1.0 / (sq.sqrt() + 1e-6)).clamp(max=1.0)
        for p_ in model.parameters():
            if p_.grad is not None and p_.shape[0] == E:
                p_.grad.mul_(scale.view(E, *([1] * (p_.dim() - 1))))
        scaler.step(opt)
        scaler.update()
        return per.detach()

    def predict(data, bs_=2048):
        model.eval()
        outs = []
        with torch.no_grad():
            for i in range(0, data["lb"].shape[0], bs_):
                sl = {k: v[i : i + bs_].unsqueeze(0).expand(E, *v[i : i + bs_].shape) for k, v in data.items()}
                outs.append(fwd(sl).clamp(min=0))
        return torch.cat(outs, 1)                                   # (E,N,H)

    def mae(data):
        if data is None:
            return np.full(E, np.nan)
        p = predict(data)
        m = (~torch.isnan(data["tg"])).unsqueeze(0)
        return ((torch.abs(p - torch.nan_to_num(data["tg"]).unsqueeze(0)) * m).sum((1, 2)) / m.sum((1, 2))).cpu().numpy()

    # one-unit timing -> epochs that fit the stage budget
    epochs_eff = epochs
    if budget_sec:
        sd = {k: v.clone() for k, v in model.state_dict().items()}
        model.train()
        for _ in range(6):                                          # untimed warm-up: CUDA init, allocator, autotune
            step()
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(15):
            step()
        torch.cuda.synchronize()
        per_epoch = (time.time() - t0) / 15 * spe * 1.2 + 1.0
        epochs_eff = int(max(5, min(epochs, budget_sec // per_epoch)))
        model.load_state_dict(sd)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scaler = _scaler(use_amp)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs_eff * spe)
    best = np.full(E, 1e18); sm = np.full(E, np.nan); best_sm = np.full(E, 1e18)
    since = np.zeros(E, int); over = np.zeros(E, int); active = np.ones(E, bool); reason = ["max_epochs"] * E
    stopped_at = np.zeros(E, int)
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    last_train = np.full(E, np.nan)
    hist, t0 = [], time.time()
    for ep in range(epochs_eff):
        model.train()
        tot = torch.zeros(E, device=device)
        perms = torch.rand(E, n, device=device).argsort(1)            # own data order per member
        for i in range(spe):
            tot += step(perms[:, i * bs : (i + 1) * bs])
            sch.step()
        trn = (tot / spe).cpu().numpy()
        cm, rm = (mae(dch), mae(drp)) if not final else (trn.copy(), np.full(E, np.nan))
        improved = (cm < best) & active if not final else np.zeros(E, bool)
        if improved.any():
            for k, v in model.state_dict().items():
                if v.dim() > 0 and v.shape[0] == E:
                    sel = torch.as_tensor(improved, device=device).view(E, *([1] * (v.dim() - 1)))
                    best_state[k] = torch.where(sel, v, best_state[k])
        best = np.where(improved, cm, best)
        sm = np.where(np.isnan(sm), cm, 0.6 * sm + 0.4 * cm)
        gain = sm < best_sm * (1 - min_rel)
        best_sm = np.where(gain, sm, best_sm)
        since = np.where(gain, 0, since + 1)
        falling = trn <= last_train
        over = np.where((sm > best_sm * (1 + overfit_gap)) & falling, over + 1, 0)
        last_train = trn
        for e in range(E):
            if active[e] and not final:
                if over[e] >= overfit_run:
                    active[e], reason[e], stopped_at[e] = False, "overfitting", ep + 1
                elif since[e] >= patience and ep + 1 >= min_epochs:
                    active[e], reason[e], stopped_at[e] = False, "plateau", ep + 1
        hist.append({"epoch": ep + 1, "sec": round(time.time() - t0, 1), "train": trn.round(5).tolist(), "choose_mae": cm.round(2).tolist(),
                     "report_mae_monitor_only": rm.round(2).tolist(), "active": active.tolist(), "lr": sch.get_last_lr()[0]})
        _atomic_json(os.path.join(out_dir, f"{name}_history.json"),
                     {"members": [{"drop": d, "seed": s} for d, s in members], "rows": hist, "best_choose": best.tolist(),
                      "stop_reason": reason, "epochs_planned": epochs_eff})
        if improved.any():
            torch.save(best_state, os.path.join(out_dir, f"{name}_best.pt"))
        if (ep + 1) % 5 == 0:
            torch.save(model.state_dict(), os.path.join(out_dir, f"{name}_last.pt"))
        if not active.any():
            break
    stopped_at = np.where(stopped_at == 0, len(hist), stopped_at)
    if final:                                                        # fixed schedule: keep the last state, no selection set exists
        info_final = {"members": [{"drop": d, "seed": s_, "stop_reason": "fixed_epochs", "stopped_epoch": len(hist)} for d, s_ in members],
                      "epochs_run": len(hist), "epochs_planned": epochs_eff, "wall_sec": round(time.time() - t0, 1), "peak_mem_mib": 0.0}
        model.eval()
        return None, info_final, model
    model.load_state_dict(best_state)
    ev = {k: torch.cat([dch[k], drp[k]], 0) for k in dch}
    preds = predict(ev).cpu().numpy()                               # (E, Nch+Nrp, H)
    info = {"members": [{"drop": d, "seed": s, "best_choose": float(best[i]), "stop_reason": reason[i], "stopped_epoch": int(stopped_at[i])}
                        for i, (d, s) in enumerate(members)], "epochs_run": len(hist), "epochs_planned": epochs_eff,
            "peak_mem_mib": torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else 0.0, "wall_sec": round(time.time() - t0, 1)}
    model.eval()
    return preds, info, model


def predict_members(model, data, mode, device, amp=True, bs=2048):
    """All members on the same samples -> (E, N, H) raw-unit predictions (clamped at 0)."""
    model.eval()
    E = model.E
    outs = []
    use_amp = bool(amp and device == "cuda")
    with torch.no_grad():
        for i in range(0, data["lb"].shape[0], bs):
            b = {k: v[i : i + bs].unsqueeze(0).expand(E, *v[i : i + bs].shape) for k, v in data.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                out = model(b["lb"], b["wx"], b["cal"], b["sid"], b["static"], b["extra"])
            out = out.float()
            out = out * b["sc"] if mode == "rel" else out * b["std"].unsqueeze(-1) + b["mean"].unsqueeze(-1)
            outs.append(out.clamp(min=0))
    return torch.cat(outs, 1)
