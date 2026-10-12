"""Base model: 2-D 'candidate x position' U-Net over a whole well.

Image per well and cut point: rows = candidate offsets from the anchor TVT (256 rows, 0.75 ft, +-96 ft),
columns = 16-row blocks (16 history blocks before the cut, up to 640 after). Channels:
  horizontal GR, calibrated typewell GR at the candidate depth, their difference, the heavy-tailed match score
  -log(1+|r|) (Dieckow et al. 2025, Eq. 4), a long-window smoothed difference, a history line
  (known path before the cut), a constant-dip prior line (surface Z+TVT extrapolated linearly in MD), masks,
  geometry rows and coordinates.
Output: logits per (row, column); decode = expected offset under a softmax over rows (learned temperature).
Loss: soft-target cross entropy (sigma 1 ft) + 0.1 L1 on the expected offset + 0.1 slope L1; history columns 0.25.
Training: random cut points (natural cut with p 0.5), synthetic suffixes with p 0.5 (typewell along a perturbed path
drawn from our fitted AR(1) slope prior + the well's own rolled residual), 50% of the fit wells (reduced size).
For the stack: every epoch predicts the held-out fold (natural cut, full suffix) and saves the block offsets.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import tracker as lt
from wells import append_jsonl, eligible_ids, fold_of, load_wells

BS, HC, FC = 16, 16, 640                 # block rows, history columns, future columns
NR, DR = 256, 0.75                       # candidate rows, row step (ft)
TW_RES = 0.25
OFFS = (np.arange(NR) - (NR - 1) / 2) * DR
FIT_FRACTION = 0.5


class WellBank:
    """Row-level arrays of a set of wells padded on the GPU, plus calibrated z-scored typewells."""

    def __init__(self, wells, ids, device):
        self.ids = ids; n = len(ids)
        N = max(wells[i]["n"] for i in ids) + (HC + FC + 2) * BS
        f = lambda: np.full((n, N), np.nan, np.float32)
        gr, tvt, z, x, y, md = f(), f(), f(), f(), f(), f()
        self.n = np.zeros(n, int); self.p = np.zeros(n, int)
        refs, tw0 = [], np.zeros(n)
        for r, wid in enumerate(ids):
            w = wells[wid]; m = w["n"]
            a, b, _ = lt.calibrate(w)
            grid = np.arange(w["tw_tvt"][0], w["tw_tvt"][-1], TW_RES)
            ref = a + b * np.interp(grid, w["tw_tvt"], w["tw_gr"]); mu, sd = ref.mean(), max(ref.std(), 1e-6)
            refs.append(((ref - mu) / sd).astype(np.float32)); tw0[r] = grid[0]
            gr[r, :m] = (w["gr"] - mu) / sd
            t = w["tvt"] if np.isfinite(w["tvt"]).all() else w["tvt_input"]
            tvt[r, :m] = t; z[r, :m] = w["z"]; x[r, :m] = w["x"]; y[r, :m] = w["y"]; md[r, :m] = w["md"]
            self.n[r], self.p[r] = m, w["p"]
        Lr = max(len(v) for v in refs)
        rm = np.full((n, Lr), np.nan, np.float32); rl = np.zeros(n, int)
        for r, v in enumerate(refs):
            rm[r, :len(v)] = v; rl[r] = len(v)
        T = lambda a: torch.as_tensor(a, device=device)
        self.gr, self.tvt, self.z, self.x, self.y, self.md = T(gr), T(tvt), T(z), T(x), T(y), T(md)
        self.ref, self.rlen, self.tw0 = T(rm), T(rl), T(tw0).float()
        self.nt, self.pt = T(self.n), T(self.p)
        self.offs = T(OFFS.astype(np.float32)); self.device = device

    def ref_at(self, rows, T):
        pos = (T - self.tw0[rows].view(-1, *[1] * (T.dim() - 1))) / TW_RES
        lo = pos.floor().long(); fr = pos - lo
        n = self.rlen[rows].view(-1, *[1] * (T.dim() - 1))
        ok = (lo >= 0) & (lo + 1 < n)
        lo = lo.clamp(0, self.ref.shape[1] - 2)
        rr = rows.view(-1, *[1] * (T.dim() - 1)).expand_as(lo)
        v = self.ref[rr, lo] * (1 - fr) + self.ref[rr, lo + 1] * fr
        return torch.where(ok, v, torch.full_like(v, float("nan")))


def nanmean_blocks(v):                                        # (b, C*BS) -> (b, C)
    v = v.view(v.shape[0], -1, BS); f = torch.isfinite(v)
    return torch.where(f.any(-1), torch.nansum(v, -1) / f.sum(-1).clamp(min=1), torch.full_like(v[..., 0], float("nan")))


def make_batch(bank: WellBank, rows, cuts, synth: torch.Tensor | None, gen, prior):
    """Build images (b, CH, NR, C), targets (b, C), masks; cuts are row indices of the first unknown row."""
    dev = bank.device; b = len(rows); C = HC + FC
    start = cuts - HC * BS
    idx = start[:, None] + torch.arange(C * BS, device=dev)[None]
    inwell = (idx >= 0) & (idx < bank.nt[rows][:, None])
    idc = idx.clamp(min=0)
    g = torch.where(inwell, bank.gr[rows[:, None], idc], torch.full_like(idc, float("nan"), dtype=torch.float32))
    t = torch.where(inwell, bank.tvt[rows[:, None], idc], torch.full_like(g, float("nan")))
    z = bank.z[rows[:, None], idc]; md = bank.md[rows[:, None], idc]; x = bank.x[rows[:, None], idc]; y = bank.y[rows[:, None], idc]
    ai = (cuts - 1)
    t_anchor = bank.tvt[rows, ai]; z_anchor = bank.z[rows, ai]; md_anchor = bank.md[rows, ai]
    x_anchor, y_anchor = bank.x[rows, ai], bank.y[rows, ai]
    after = idx >= cuts[:, None]
    if synth is not None and synth.any():
        # perturbed path from the AR(1) slope prior (per 16-row block), applied after the cut only
        nb = C
        a = prior["alpha"] ** (BS / 10); sig = prior["sigma"] * lt.DT * BS / 10 * (0.5 + torch.rand(b, 1, device=dev, generator=gen))
        v = torch.zeros(b, device=dev); steps = []
        for _ in range(nb):
            v = a * v + torch.randn(b, device=dev, generator=gen) * sig[:, 0]; steps.append(v)
        pert = torch.cumsum(torch.stack(steps, 1), 1)                          # (b, C) ft
        pert = pert - pert[:, HC:HC + 1]
        pert_rows = pert.repeat_interleave(BS, 1)
        t_syn = torch.where(after, t + pert_rows, t)
        resid = g - bank.ref_at(rows, t)
        shift = torch.randint(0, C * BS, (b, 1), device=dev, generator=gen)
        resid = torch.gather(torch.nan_to_num(resid), 1, (torch.arange(C * BS, device=dev)[None] + shift) % (C * BS))
        g_syn = bank.ref_at(rows, t_syn) + resid * (0.55 + 0.65 * torch.rand(b, 1, device=dev, generator=gen))
        g_syn = torch.where(torch.isnan(g), g, g_syn)                          # keep the missing pattern
        s = synth[:, None]
        t = torch.where(s & after, t_syn, t); g = torch.where(s & after, g_syn, g)
    gb = nanmean_blocks(g)
    gl = nanmean_blocks(g)                                                     # long smooth over 5 blocks
    gl = torch.where(torch.isfinite(gl), gl, torch.zeros_like(gl))
    gl = F.avg_pool1d(gl[:, None], 5, 1, padding=2, count_include_pad=False)[:, 0]
    tb = nanmean_blocks(t); zb = nanmean_blocks(z); mdb = nanmean_blocks(md); xb = nanmean_blocks(x); yb = nanmean_blocks(y)
    target = tb - t_anchor[:, None]
    pos_mask = torch.isfinite(zb) & (torch.arange(C, device=dev)[None] >= 0)
    hist = torch.zeros_like(pos_mask); hist[:, :HC] = True; hist &= pos_mask
    # constant-dip prior: surface u = z + tvt linear in md over the last 300 rows before the cut
    lo = (cuts - 300).clamp(min=0)
    u1 = bank.z[rows, ai] + t_anchor; u0 = bank.z[rows, lo] + bank.tvt[rows, lo]
    dip = (u1 - u0) / (bank.md[rows, ai] - bank.md[rows, lo]).clamp(min=1.0)
    prior_off = (u1[:, None] + dip[:, None] * (mdb - md_anchor[:, None]) - zb) - t_anchor[:, None]
    prior_off = torch.nan_to_num(prior_off).clamp(-96, 96)
    cand = t_anchor[:, None] + bank.offs[None]                                 # (b, NR)
    tw = bank.ref_at(rows, cand)                                               # (b, NR)
    twl = F.avg_pool1d(torch.nan_to_num(tw)[:, None], 7, 1, padding=3, count_include_pad=False)[:, 0]
    tvalid = torch.isfinite(tw)
    H = gb[:, None, :].expand(-1, NR, -1); TW = torch.nan_to_num(tw)[:, :, None].expand(-1, -1, C)
    gmask = torch.isfinite(H)
    d = torch.where(gmask, torch.nan_to_num(H) - TW, torch.zeros_like(TW))
    dl = gl[:, None, :] - twl[:, :, None]
    score = torch.where(gmask, -torch.log1p(d.abs()), torch.zeros_like(d))
    off = bank.offs[None, :, None]
    hist_line = torch.exp(-0.5 * ((off - torch.nan_to_num(target)[:, None, :]) / 1.0) ** 2) * hist[:, None, :]
    prior_line = torch.exp(-0.5 * ((off - prior_off[:, None, :]) / 4.0) ** 2) * pos_mask[:, None, :]
    dzb = torch.nan_to_num(zb - z_anchor[:, None]) / 128.0
    incl = torch.nan_to_num(torch.diff(zb, dim=1, prepend=zb[:, :1]) / torch.diff(mdb, dim=1, prepend=mdb[:, :1] - 1).clamp(min=1e-3)).clamp(-2, 2)
    az = torch.atan2(torch.nan_to_num(torch.diff(yb, dim=1, prepend=yb[:, :1])), torch.nan_to_num(torch.diff(xb, dim=1, prepend=xb[:, :1])) + 1e-6)
    geo = torch.stack([dzb, incl, torch.sin(az), torch.cos(az), torch.nan_to_num(mdb - md_anchor[:, None]) / 10000.0, prior_off / 96.0,
                       torch.nan_to_num(xb - x_anchor[:, None]) / 10000.0, torch.nan_to_num(yb - y_anchor[:, None]) / 10000.0], 1)
    G = geo[:, :, None, :].expand(-1, -1, NR, -1)
    rc = torch.linspace(-1, 1, NR, device=dev)[None, None, :, None].expand(b, 1, NR, C)
    cc = torch.linspace(-1, 1, C, device=dev)[None, None, None, :].expand(b, 1, NR, C)
    img = torch.cat([torch.nan_to_num(H)[:, None], TW[:, None], d[:, None], score[:, None], dl[:, None].clamp(-6, 6),
                     hist_line[:, None], prior_line[:, None], hist[:, None, None, :].expand(-1, 1, NR, -1).float(),
                     pos_mask[:, None, None, :].expand(-1, 1, NR, -1).float(), gmask[:, None].float(),
                     tvalid[:, None, :, None].expand(-1, 1, -1, C).float(), G, rc, cc], 1)
    tmask = pos_mask & torch.isfinite(target) & (target.abs() < 96)
    return img, torch.nan_to_num(target), tmask, hist, after.view(b, C, BS).any(-1)


NCH = 11 + 8 + 2


class RB(nn.Module):
    def __init__(self, ci, co, drop):
        super().__init__()
        g = lambda c: nn.GroupNorm(min(8, c // 4), c)
        self.b = nn.Sequential(nn.Conv2d(ci, co, 3, padding=1), g(co), nn.GELU(), nn.Dropout2d(drop),
                               nn.Conv2d(co, co, 3, padding=1), g(co))
        self.s = nn.Conv2d(ci, co, 1) if ci != co else nn.Identity()

    def forward(self, x):
        return F.gelu(self.b(x) + self.s(x))


class GridUNet(nn.Module):
    def __init__(self, width, drop=0.1):
        super().__init__()
        ch = [width, width * 2, width * 4, width * 8]
        self.inc = RB(NCH, ch[0], drop)
        self.down = nn.ModuleList([RB(ch[i], ch[i + 1], drop) for i in range(3)])
        self.up = nn.ModuleList([RB(ch[i + 1] + ch[i], ch[i], drop) for i in reversed(range(3))])
        self.head = nn.Conv2d(ch[0], 1, 1)
        self.log_temp = nn.Parameter(torch.zeros(()))

    def forward(self, x):
        s = [self.inc(x)]; h = s[0]
        for d in self.down:
            h = d(F.max_pool2d(h, 2)); s.append(h)
        for i, u in enumerate(self.up):
            h = F.interpolate(h, size=s[-2 - i].shape[-2:], mode="bilinear", align_corners=False)
            h = u(torch.cat([h, s[-2 - i]], 1))
        return self.head(h)[:, 0]                                              # (b, NR, C)


def decode(logits, offs, temp):
    p = torch.softmax(logits.float() / temp, 1)
    mean = (p * offs[None, :, None]).sum(1)
    sd = torch.sqrt(((offs[None, :, None] - mean[:, None]) ** 2 * p).sum(1).clamp(min=0))
    return mean, sd, p


def loss_fn(logits, net, offs, target, tmask, hist):
    temp = torch.exp(net.log_temp).clamp(0.25, 4.0)
    logp = torch.log_softmax(logits.float() / temp, 1)
    soft = torch.exp(-0.5 * ((offs[None, :, None] - target[:, None, :]) / 1.0) ** 2)
    soft = soft / soft.sum(1, keepdim=True).clamp(min=1e-9)
    w = tmask.float() * torch.where(hist, 0.25, 1.0)
    ce = -(soft * logp).sum(1)
    mean, _, _ = decode(logits, offs, temp)
    reg = (mean - target).abs()
    slope = (torch.diff(mean, dim=1) - torch.diff(target, dim=1)).abs()
    sw = (w[:, 1:] * w[:, :-1])
    den = w.sum().clamp(min=1)
    return (ce * w).sum() / den + 0.1 * (reg * w).sum() / den + 0.1 * (slope * sw).sum() / sw.sum().clamp(min=1)


@torch.no_grad()
def predict(net, bank, bs=4):
    """Natural-cut, full-suffix expected offsets per future block: returns list of (L,) arrays and sds."""
    net.eval(); outs, sds = [], []
    temp = torch.exp(net.log_temp).clamp(0.25, 4.0)
    for i in range(0, len(bank.ids), bs):
        rows = torch.arange(i, min(i + bs, len(bank.ids)), device=bank.device)
        cuts = bank.pt[rows]
        img, _, _, _, _ = make_batch(bank, rows, cuts, None, None, None)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = net(img)
        m, s, _ = decode(lg, bank.offs, temp)
        for j, r in enumerate(rows.tolist()):
            L = int(math.ceil((bank.n[r] - bank.p[r]) / BS))
            outs.append(m[j, HC:HC + L].cpu().numpy()); sds.append(s[j, HC:HC + L].cpu().numpy())
    net.train()
    return outs, sds


def rows_rmse(wells, bank, preds):
    se = n = 0.0
    for r, wid in enumerate(bank.ids):
        w = wells[wid]; p = w["p"]; L = len(preds[r])
        centers = p + np.arange(L) * BS + (BS - 1) / 2
        pred = float(w["tvt_input"][p - 1]) + np.interp(np.arange(p, w["n"]), np.r_[p - 1, centers], np.r_[0.0, preds[r]])
        e = pred - w["tvt"][p:]; se += float((e ** 2).sum()); n += len(e)
    return math.sqrt(se / n)


def train_run(wells, fit, ev, device, width, seed, epochs, per_well, bsz, lr, log, tag, out, wall, save_half=False):
    torch.manual_seed(seed)
    gen = torch.Generator(device=device); gen.manual_seed(1000 + seed)
    bank, ebank = WellBank(wells, fit, device), WellBank(wells, ev, device)
    prior = lt.fit_prior([wells[i] for i in fit], 10)
    net = GridUNet(width).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    steps = max(1, len(fit) * per_well // bsz)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * steps, pct_start=0.1)
    snaps, t0 = {}, time.time()
    for ep in range(epochs):
        te = time.time(); acc = gn = 0.0
        for _ in range(steps):
            rows = torch.randint(0, len(fit), (bsz,), device=device, generator=gen)
            n, p = bank.nt[rows], bank.pt[rows]
            lo = torch.maximum(p // 2, torch.full_like(p, HC * BS)); hi = (n - 4 * BS).clamp(min=lo + 1)
            rnd = lo + (torch.rand(bsz, device=device, generator=gen) * (hi - lo).float()).long()
            cuts = torch.where(torch.rand(bsz, device=device, generator=gen) < 0.5, p, rnd)
            synth = torch.rand(bsz, device=device, generator=gen) < 0.5
            img, tgt, tm, hist, _ = make_batch(bank, rows, cuts, synth, gen, prior)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = net(img)
            l = loss_fn(lg, net, bank.offs, tgt, tm, hist)
            opt.zero_grad(set_to_none=True); l.backward()
            gn += float(torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)); opt.step(); sch.step(); acc += float(l.detach())
        if save_half and ep + 1 > epochs // 2:   # snapshot-blend members for the final (full-data) model
            torch.save({"model": net.state_dict(), "epoch": ep + 1, "width": width, "seed": seed,
                        "temp": float(torch.exp(net.log_temp))}, out / f"{tag}_ep{ep + 1}.pt")
        preds, sds = predict(net, ebank)
        snaps[f"ep{ep + 1}"] = preds
        rec = {"event": "epoch", "tag": tag, "width": width, "seed": seed, "epoch": ep + 1, "train_loss": acc / steps, "grad_norm": gn / steps,
               "lr": sch.get_last_lr()[0], "temp": float(torch.exp(net.log_temp)), "epoch_s": round(time.time() - te, 1),
               "samples_per_s": round(steps * bsz / (time.time() - te), 1), "gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
               "heldout_row_rmse_monitor": rows_rmse(wells, ebank, preds),
               "heldout_mean_sd": float(np.mean([s.mean() for s in sds]))}
        append_jsonl(log, rec); print(json.dumps(rec), flush=True)
        if time.time() - t0 > wall:
            break
    torch.save({"model": net.state_dict(), "epoch": ep + 1, "width": width, "seed": seed}, out / f"{tag}_last.pt")
    return snaps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="data/stage")
    ap.add_argument("--fold", type=int, default=0, help="0 = full mode: train on every eligible well")
    ap.add_argument("--runs", default="16:0,24:0,32:0", help="width:seed list")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--per-well", type=int, default=2)
    ap.add_argument("--bsz", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wall", type=float, default=1500)
    ap.add_argument("--limit", type=int, default=0, help="smoke: limit wells")
    ap.add_argument("--out", default="runs/grid")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda"); torch.backends.cudnn.benchmark = True
    wells = load_wells(args.stage); ids = eligible_ids(wells); folds = fold_of(ids)
    if args.fold == 0:   # full mode: all eligible wells train; a few wells kept only as a smoke monitor
        fit, ev = list(ids), list(ids[:8])
    else:
        k = args.fold - 1
        fit = [i for i in ids if folds[i] != k]; ev = [i for i in ids if folds[i] == k]
        rng = np.random.default_rng(0)
        fit = sorted(rng.choice(fit, size=int(len(fit) * FIT_FRACTION), replace=False).tolist())
    if args.limit:
        fit, ev = fit[:args.limit], ev[:max(2, args.limit // 4)]
    for spec in args.runs.split(","):
        width, seed = map(int, spec.split(":"))
        tag = f"fold{args.fold}_w{width}_s{seed}"
        tag = tag if args.fold else f"full_w{width}_s{seed}"
        snaps = train_run(wells, fit, ev, device, width, seed, args.epochs, args.per_well, args.bsz, args.lr,
                          out / "train.jsonl", tag, out, args.wall, save_half=(args.fold == 0))
        np.savez_compressed(out / f"{tag}_snapshots.npz", well_ids=np.array(ev),
                            **{f"{e}__{w}": a for e, arrs in snaps.items() for w, a in zip(ev, arrs)})
        append_jsonl(out / "done.jsonl", {"tag": tag, "fit_wells": len(fit), "eval_wells": len(ev)})


if __name__ == "__main__":
    main()
