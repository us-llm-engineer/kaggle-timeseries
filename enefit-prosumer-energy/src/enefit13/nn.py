"""DL variants (TiDE-style MLP encoder/decoder, day-ahead: one sample = (series, day D) -> 24 hours).
V3: targets and look-back are capacity-relative (y / scale); L1 loss weighted by scale (== raw MAE).
V4: scale handled by window statistics only (reversible instance normalisation of the look-back, de-normalised output), no capacity division.
Persistence: checkpoint last.pt every epoch, best.pt by choose-role MAE, atomic history JSON every epoch. No live prints needed."""
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .data import WEATHER

L, H = 168, 24          # look-back ends at D-2 23:00 (index ds-25): 168 hours = 7 days
GAP = 24                # hours between the end of the look-back and the target day (day D-1 is unknown)


def _atomic_json(path, obj):
    tmp = path + ".tmp"
    json.dump(obj, open(tmp, "w"))
    os.replace(tmp, path)


def build_tensors(panel, an, ar, s, ti, sid, t0):
    """Dense per-series arrays on the full hourly grid, then day samples."""
    S, T = an.shape
    wx = np.zeros((S, T, len(WEATHER)), dtype=np.float32)
    wvals = panel[WEATHER].values.astype("float32")
    mu = np.nanmean(wvals, 0)
    wvals = np.where(np.isnan(wvals), mu, wvals)
    wx[s, ti] = wvals
    sd = wx.reshape(-1, wx.shape[-1]).std(0) + 1e-6
    wx = (wx - mu) / sd
    sc = np.ones((S, T), dtype=np.float32)
    sc[s, ti] = panel.scale.values
    # series-level static ids
    ser = panel.drop_duplicates(["prediction_unit_id", "is_consumption"]).copy()
    ser["sid"] = [sid[(int(a), int(b))] for a, b in zip(ser.prediction_unit_id, ser.is_consumption)]
    ser = ser.sort_values("sid")
    static = ser[["county", "is_business", "product_type", "is_consumption"]].values.astype("int64")
    return dict(an=an, ar=ar, wx=wx, sc=sc, static=static, S=S, T=T, t0=t0, w_mu=mu, w_sd=sd)


def day_samples(td, first_day=8, last_day=None):
    """All (series, day) pairs with a full look-back and at least one observed target hour."""
    T, S = td["T"], td["S"]
    nd = T // 24
    last_day = nd - 1 if last_day is None else last_day
    sidx, didx = np.meshgrid(np.arange(S), np.arange(first_day, last_day + 1), indexing="ij")
    sidx, didx = sidx.ravel(), didx.ravel()
    ds = didx * 24
    obs = np.stack([~np.isnan(td["ar"][sidx, ds + h]) for h in range(24)], 1).any(1)
    return sidx[obs], didx[obs]


def cal_feats(t0, didx):
    base = pd.Timestamp(t0) + pd.to_timedelta(didx * 24, unit="h")
    hrs = np.arange(24)
    hs, hc = np.sin(2 * np.pi * hrs / 24), np.cos(2 * np.pi * hrs / 24)
    dow = base.dayofweek.values
    doy = base.dayofyear.values
    n = len(didx)
    f = np.zeros((n, 24, 6), dtype=np.float32)
    f[:, :, 0], f[:, :, 1] = hs, hc
    f[:, :, 2] = np.sin(2 * np.pi * dow / 7)[:, None]
    f[:, :, 3] = np.cos(2 * np.pi * dow / 7)[:, None]
    f[:, :, 4] = np.sin(2 * np.pi * doy / 365.25)[:, None]
    f[:, :, 5] = np.cos(2 * np.pi * doy / 365.25)[:, None]
    return f


def batch_arrays(td, sidx, didx, mode):
    """mode 'rel' (V3): look-back / target in y/scale units. mode 'raw' (V4): raw units."""
    ds = didx * 24
    ar_ = td["an"] if mode == "rel" else td["ar"]
    lb = np.stack([ar_[s, d - (L + GAP) : d - GAP] for s, d in zip(sidx, ds)])
    tg = np.stack([td["ar"][s, d : d + H] for s, d in zip(sidx, ds)])          # raw target
    sc = np.stack([td["sc"][s, d : d + H] for s, d in zip(sidx, ds)])          # scale per target hour
    wx = np.stack([td["wx"][s, d : d + H] for s, d in zip(sidx, ds)])
    return lb.astype("float32"), tg.astype("float32"), sc.astype("float32"), wx.astype("float32")


class Res(nn.Module):
    def __init__(self, i, h, o, p=0.1, ln=True):
        super().__init__()
        self.f1, self.f2, self.sk, self.dp = nn.Linear(i, h), nn.Linear(h, o), nn.Linear(i, o), nn.Dropout(p)
        # LayerNorm over a single output feature is a constant (x - mean = 0): the output block must not use it
        self.ln = nn.LayerNorm(o) if (ln and o > 1) else nn.Identity()

    def forward(self, x):
        return self.ln(self.dp(self.f2(torch.relu(self.f1(x)))) + self.sk(x))


class TiDE(nn.Module):
    def __init__(self, nw, nser, mode, hid=1024, r=4, p=8, emb=16, n_static=(16, 2, 4, 2), drop=0.5):  # TiDE Table 8 (Electricity)
        super().__init__()
        self.mode = mode
        self.ser_emb = nn.Embedding(nser, emb)
        self.cnt_emb = nn.Embedding(n_static[0], 4)
        self.proj = Res(nw + 6, 64, r, drop)
        extra = 2 if mode == "raw" else 1          # raw: log1p(mean), log1p(std); rel: log scale
        enc_in = L + H * r + emb + 4 + 2 + extra
        self.enc = nn.Sequential(Res(enc_in, hid, hid, drop), Res(hid, hid, hid, drop))
        self.dec = Res(hid, hid, H * p, drop)
        self.tdec = Res(p + r, 64, 1, drop)
        self.skip = nn.Linear(L, H)
        self.p, self.r = p, r

    def forward(self, lb, wx, cal, sid, static, extra):
        b = lb.shape[0]
        x = self.proj(torch.cat([wx, cal], -1))                         # (b,H,r)
        e = torch.cat([lb, x.reshape(b, -1), self.ser_emb(sid), self.cnt_emb(static[:, 0]),
                       static[:, 1:3].float(), extra], 1)
        d = self.dec(self.enc(e)).reshape(b, H, self.p)
        out = self.tdec(torch.cat([d, x], -1)).squeeze(-1)
        return out + self.skip(lb)


class Runner:
    """GPU-resident trainer: every sample tensor is built once, then batches are index-selects on the device
    (no per-step NumPy work, no host sync inside the epoch)."""

    def __init__(self, td, mode, device, seed, out_dir, name, drop=0.5):
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.td, self.mode, self.dev, self.out, self.name = td, mode, device, out_dir, name
        self.model = TiDE(td["wx"].shape[-1], td["S"], mode, drop=drop).to(device)
        self.static = torch.tensor(td["static"], device=device)

    def _prep(self, sidx, didx):
        lb, tg, sc, wx = batch_arrays(self.td, sidx, didx, self.mode)
        cal = cal_feats(self.td["t0"], didx)
        nanm = np.isnan(lb)
        t = lambda a, dt=torch.float32: torch.as_tensor(a, dtype=dt, device=self.dev)
        if self.mode == "raw":
            cnt = (~nanm).sum(1).clip(min=1)
            mean = np.where(nanm, 0, lb).sum(1) / cnt
            var = (np.where(nanm, 0, (lb - mean[:, None]) ** 2)).sum(1) / cnt
            std = np.sqrt(var) + 1.0                                   # +1 kWh floor: keeps flat winter windows finite
            lbn = np.where(nanm, 0, (lb - mean[:, None]) / std[:, None]).astype("float32")
            extra = np.stack([np.log1p(np.abs(mean)), np.log1p(std)], 1).astype("float32")
            aux = (t(mean), t(std))
        else:
            lbn = np.where(nanm, 0, lb).astype("float32")
            extra = np.log(sc[:, :1] + 1.0).astype("float32")
            aux = (torch.zeros(len(lb), device=self.dev), torch.ones(len(lb), device=self.dev))
        return dict(lb=t(lbn), wx=t(wx), cal=t(cal), sid=t(sidx, torch.long), static=self.static[t(sidx, torch.long)],
                    extra=t(extra), tg=t(tg), sc=t(sc), mean=aux[0], std=aux[1])

    def materialize(self, sidx, didx, chunk=8192):
        parts = [self._prep(sidx[i : i + chunk], didx[i : i + chunk]) for i in range(0, len(sidx), chunk)]
        if not parts:
            return None
        return {k: torch.cat([p[k] for p in parts], 0) for k in parts[0]}

    @staticmethod
    def _take(batch, idx):
        return {k: v[idx] for k, v in batch.items()}

    def _predict_raw(self, b):
        out = self.model(b["lb"], b["wx"], b["cal"], b["sid"], b["static"], b["extra"])
        if self.mode == "rel":
            return out * b["sc"]
        return out * b["std"][:, None] + b["mean"][:, None]

    def predict_resident(self, data, bs=8192):
        self.model.eval()
        outs = []
        with torch.no_grad():
            for i in range(0, data["lb"].shape[0], bs):
                outs.append(self._predict_raw(self._take(data, slice(i, i + bs))).clamp(min=0))
        return torch.cat(outs, 0) if outs else None

    def evaluate(self, sidx, didx, bs=8192):
        d = self.materialize(sidx, didx)
        return np.zeros((0, H), dtype="float32") if d is None else self.predict_resident(d, bs).cpu().numpy()

    def _mae_resident(self, data):
        if data is None:
            return None
        p = self.predict_resident(data)
        m = ~torch.isnan(data["tg"])
        return float((torch.abs(p - torch.nan_to_num(data["tg"])) * m).sum() / m.sum().clamp(min=1))

    def train(self, tr, ch, rp, epochs=100, bs=512, lr=1e-3, deadline=None, budget_sec=None, patience=12, min_rel=0.002, overfit_gap=0.05, overfit_run=5):
        dtr, dch, drp = self.materialize(*tr), self.materialize(*ch), self.materialize(*rp)
        n = dtr["lb"].shape[0]
        spe = int(np.ceil(n / bs))
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        const = float(torch.nanmean(torch.abs(dtr["tg"][:, 0]))) + 1.0

        def step(idx):
            b = self._take(dtr, idx)
            pred = self._predict_raw(b)
            m = ~torch.isnan(b["tg"])
            loss = (torch.abs(pred - torch.nan_to_num(b["tg"])) * m).sum() / m.sum().clamp(min=1) / const
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            opt.step()
            return loss.detach()

        # one-unit timing inside the run: 20 warm-up steps decide how many epochs fit the stage budget (state is then reset)
        epochs_eff = epochs
        if budget_sec:
            sd = {k: v.clone() for k, v in self.model.state_dict().items()}
            torch.cuda.synchronize() if self.dev == "cuda" else None
            t0 = time.time()
            for _ in range(20):
                step(torch.randint(0, n, (bs,), device=self.dev))
            torch.cuda.synchronize() if self.dev == "cuda" else None
            per_epoch = (time.time() - t0) / 20 * spe * 1.15 + 1.0
            epochs_eff = int(max(3, min(epochs, budget_sec // per_epoch)))
            self.model.load_state_dict(sd)
            opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs_eff * spe)   # cosine decay as in TiDE Appendix B.3
        hist, best, t0 = [], 1e18, time.time()
        sm, best_sm, since_best, over_run, stop_reason = None, 1e18, 0, 0, "max_epochs"
        for ep in range(epochs_eff):
            if deadline and time.time() > deadline:
                break
            self.model.train()
            perm = torch.randperm(n, device=self.dev)
            tot = torch.zeros((), device=self.dev)
            for i in range(0, n, bs):
                tot += step(perm[i : i + bs]) * min(bs, n - i)
                sch.step()
            row = {"epoch": ep + 1, "epochs_planned": epochs_eff, "sec": round(time.time() - t0, 1),
                   "train_l1_scaled": float(tot) / n, "lr": sch.get_last_lr()[0],
                   "choose_mae": self._mae_resident(dch), "report_mae_monitor_only": self._mae_resident(drp)}
            hist.append(row)
            torch.save(self.model.state_dict(), os.path.join(self.out, f"{self.name}_last.pt"))
            key = row["choose_mae"] if row["choose_mae"] is not None else row["train_l1_scaled"]
            if key < best:
                best = key
                torch.save(self.model.state_dict(), os.path.join(self.out, f"{self.name}_best.pt"))
            # early stopping on a 3-epoch smoothed choose MAE: plateau (no relative gain >= min_rel for `patience` epochs) or
            # overfitting (train loss still falling while smoothed choose MAE sits > overfit_gap above its best for overfit_run epochs)
            sm = key if sm is None else 0.6 * sm + 0.4 * key
            if sm < best_sm * (1 - min_rel):
                best_sm, since_best = sm, 0
            else:
                since_best += 1
            falling = len(hist) > 1 and hist[-1]["train_l1_scaled"] <= hist[-2]["train_l1_scaled"]
            over_run = over_run + 1 if (sm > best_sm * (1 + overfit_gap) and falling) else 0
            row["smoothed_choose"], row["since_best"], row["over_run"] = sm, since_best, over_run
            stop = None
            if over_run >= overfit_run:
                stop = "overfitting"
            elif since_best >= patience and ep + 1 >= 20:
                stop = "plateau"
            _atomic_json(os.path.join(self.out, f"{self.name}_history.json"),
                         {"rows": hist, "best_choose": best, "epochs_planned": epochs_eff, "stop_reason": stop or "running"})
            if stop:
                stop_reason = stop
                break
        _atomic_json(os.path.join(self.out, f"{self.name}_history.json"),
                     {"rows": hist, "best_choose": best, "epochs_planned": epochs_eff, "stop_reason": stop_reason})
        self.model.load_state_dict(torch.load(os.path.join(self.out, f"{self.name}_best.pt"), map_location=self.dev))
        return hist
