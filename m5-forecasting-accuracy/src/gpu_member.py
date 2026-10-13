"""One neural ensemble member trained on the GPU (V5 hierarchical-loss mixer, V6 hurdle mixer). Data and the sparse aggregation matrix live on the GPU;
every step is a full cross-section of all 30,490 series at `n_orig` random origins, so the hierarchical loss over all 12 levels is computed exactly.
Fit uses targets <= d_1885 only; early stopping on the exact WRMSSE of the choose window (origin 1885); the report window is logged for monitoring, never used.
Run: python3 src/gpu_member.py --variant hl --seed 0 --mem-frac 0.13 --budget-min 20"""
import argparse, os, sys, time, json, math
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m5lib import *

L = 112


class Net(nn.Module):
    def __init__(self, d_in, width, depth, drop, variant):
        super().__init__()
        self.variant = variant
        self.emb = nn.ModuleList([nn.Embedding(n, k) for n, k in ((3049, 16), (7, 4), (3, 2), (10, 4), (3, 2))])
        self.edrop = nn.Dropout(0.25)
        dims = [d_in + 28] + [width] * depth
        self.hid = nn.ModuleList([nn.Sequential(nn.Linear(dims[i], dims[i + 1]), nn.LayerNorm(dims[i + 1]), nn.GELU(), nn.Dropout(drop)) for i in range(depth)])
        self.out = nn.Linear(width, HZ * (1 if variant == "hl" else 2))
        nn.init.zeros_(self.out.weight); nn.init.constant_(self.out.bias, 0.0)

    def forward(self, x, codes):
        e = self.edrop(torch.cat([m(codes[:, i]) for i, m in enumerate(self.emb)], 1))
        h = torch.cat([x, e], 1)
        for blk in self.hid: h = blk(h)
        return self.out(h)


class Data:
    def __init__(self, d, dev):
        self.dev = dev; self.n = d.n
        self.D = torch.tensor(d.D, dtype=torch.float16, device=dev)
        self.Pn = torch.tensor(d.Pf / d.Pmax[:, None], dtype=torch.float16, device=dev)
        idx = torch.arange(d.T, device=dev, dtype=torch.int32)[None, :].expand(d.n, -1)
        self.last = torch.cummax(torch.where(self.D > 0, idx, torch.full_like(idx, -1)), 1).values.to(torch.int16); del idx; torch.cuda.empty_cache()
        self.first = torch.where((self.D > 0).any(1), (self.D > 0).int().argmax(1), torch.full((d.n,), d.T, device=dev)).float()
        cal = np.zeros((1969, 14), np.float32)
        cal[np.arange(1969), d.wday - 1] = 1; cal[:, 7] = np.sin(2 * np.pi * d.doy / 366); cal[:, 8] = np.cos(2 * np.pi * d.doy / 366); cal[:, 9:13] = d.ev_t.T; cal[:, 13] = d.dom / 31
        self.C = torch.tensor(cal, device=dev); self.snap = torch.tensor(d.snap, device=dev); self.sst = torch.tensor(d.snap_state, device=dev)
        self.codes = torch.tensor(d.codes[[0, 1, 2, 3, 4]].T.copy(), device=dev)
        S = d.S.tocsr(); self.S = torch.sparse_csr_tensor(torch.tensor(S.indptr, dtype=torch.int64), torch.tensor(S.indices, dtype=torch.int64), torch.tensor(S.data, dtype=torch.float32), size=S.shape, device=dev)
        self.lvl = torch.tensor(d.lvl, device=dev)
        for role in ("choose", "report"):
            sc = Scorer(d, role); setattr(self, "scale_" + role, torch.tensor(np.maximum(sc.scale, 1e-6), dtype=torch.float32, device=dev)); setattr(self, "w_" + role, torch.tensor(sc.w, dtype=torch.float32, device=dev))

    def batch(self, t):
        """Features for all series at origin t (number of known days); only days < t enter the history."""
        n = self.n; D = self.D
        hist = D[:, t - L:t].float(); s = 1.0 + hist[:, -28:].mean(1)
        x_h = (hist / s[:, None]).clamp(max=10.0)
        g = torch.arange(t, t + HZ, device=self.dev)
        cal = self.C[g].reshape(1, -1).expand(n, -1)                     # 28*14, shared across series
        sn = self.snap[self.sst[:, None], g[None, :]]                    # n x 28
        pn = self.Pn[:, g].float(); pr = pn / (self.Pn[:, t - 1].float()[:, None] + 1e-3)
        dsl = ((t - 1 - self.last[:, t - 1].float()).clamp(max=200) / 28.0)[:, None]
        nz = (hist[:, -28:] > 0).float().mean(1, keepdim=True); age = torch.log1p((t - self.first).clamp(min=0))[:, None] / 8.0
        x = torch.cat([x_h, cal, sn, pn, pr.clamp(max=3.0), dsl, nz, age, torch.log1p(s)[:, None]], 1)
        return x, s


def wrmsse_t(data, yhat, y, role):
    """Exact WRMSSE (torch). yhat, y: [n, 28]. Returns scalar tensor."""
    e = torch.sparse.mm(data.S, y - yhat) ** 2
    r = torch.sqrt(e.mean(1) / getattr(data, "scale_" + role))
    per = torch.zeros(12, device=r.device).index_add_(0, data.lvl, getattr(data, "w_" + role) * r)
    return per.mean()


def hier_loss(data, yhat, y):
    """yhat, y: [n, nO, 28] -> weighted RMSSE surrogate over all 12 levels, averaged over origins."""
    n, nO, _ = yhat.shape
    e = torch.sparse.mm(data.S, (y - yhat).reshape(n, nO * HZ)).reshape(-1, nO, HZ) ** 2
    r = torch.sqrt(e.mean(2) / data.scale_choose[:, None] + 1e-6)                  # [42840, nO]
    return (data.w_choose[:, None] * r).sum(0).mean() / 12.0 * 1.0


def predict(net, data, t, variant):
    net.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        x, s = data.batch(t); o = net(x, data.codes).float()
    return head(o, s, variant)[0]


def head(o, s, variant):
    if variant == "hl": return s[:, None] * F.softplus(o + 0.5), None
    p = torch.sigmoid(o[:, :HZ]); mu = s[:, None] * F.softplus(o[:, HZ:] + 0.5) + 1e-3
    return p * mu, (p, mu)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["hl", "hurdle"], required=True); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--mem-frac", type=float, default=0.13)
    ap.add_argument("--budget-min", type=float, default=20); ap.add_argument("--n-orig", type=int, default=2); ap.add_argument("--width", type=int, default=512); ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--drop", type=float, default=0.2); ap.add_argument("--lr", type=float, default=2e-3); ap.add_argument("--wd", type=float, default=0.02)
    ap.add_argument("--eval-every", type=int, default=150); ap.add_argument("--patience", type=int, default=12); ap.add_argument("--max-steps", type=int, default=10**9)
    ap.add_argument("--out", default="results"); a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed); dev = torch.device("cuda"); torch.cuda.set_per_process_memory_fraction(a.mem_frac)
    name = f"{'V5_hl_mixer' if a.variant == 'hl' else 'V6_hurdle_mixer'}_s{a.seed}"; os.makedirs(f"{a.out}/preds", exist_ok=True); os.makedirs("logs", exist_ok=True)
    logf = open(f"logs/{name}.log", "a")
    def log(m):
        line = f"{time.strftime('%H:%M:%S')} {name} {m}"; print(line, flush=True); logf.write(line + "\n"); logf.flush()
    d = load_prep(); data = Data(d, dev); T0 = time.time()
    x0, _ = data.batch(1500); net = Net(x0.shape[1], a.width, a.depth, a.drop, a.variant).to(dev)
    log(f"params {sum(p.numel() for p in net.parameters())/1e6:.2f}M, input {x0.shape[1]}, n_orig {a.n_orig}, mem frac {a.mem_frac}")
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
    y_choose = data.D[:, ORIGIN["choose"]:ORIGIN["choose"] + HZ].float(); y_report = data.D[:, ORIGIN["report"]:ORIGIN["report"] + HZ].float()
    y_trainorig = data.D[:, 1857:1885].float()
    lo, hi = L + 1, FIT_END - HZ                                                      # origins with t + 28 <= 1885
    best, best_state, since, hist, step, run = 9e9, None, 0, [], 0, {"loss": 0.0, "n": 0}
    budget = a.budget_min * 60; gn = 0.0
    while True:
        el = time.time() - T0; frac = min(el / budget, 1.0)
        if frac >= 1.0 or step >= a.max_steps: log("budget reached"); break
        for gr in opt.param_groups: gr["lr"] = a.lr * (0.5 * (1 + math.cos(math.pi * frac))) * min(1.0, (step + 1) / 100)
        net.train(); ts = np.random.randint(lo, hi + 1, size=a.n_orig)
        xs, ss, ys = [], [], []
        for t in ts: x, s = data.batch(int(t)); xs.append(x); ss.append(s); ys.append(data.D[:, t:t + HZ].float())
        x = torch.cat(xs); codes = data.codes.repeat(a.n_orig, 1); s = torch.cat(ss)
        with torch.autocast("cuda", dtype=torch.bfloat16): o = net(x, codes)
        yhat, ph = head(o.float(), s, a.variant); yhat3 = yhat.reshape(a.n_orig, data.n, HZ).permute(1, 0, 2); y3 = torch.stack(ys, 1)
        aux = ((yhat3 - y3) ** 2 / data.scale_choose[:data.n].clamp(min=0.05)[:, None, None]).mean()
        if a.variant == "hl": loss = hier_loss(data, yhat3, y3) + 0.5 * aux; comp = {"hl": float(loss) - 0.5 * float(aux), "aux": float(aux)}
        else:
            p, mu = ph; yy = y3.permute(1, 0, 2).reshape(-1, HZ); pos = (yy > 0).float()
            bce = F.binary_cross_entropy(p.clamp(1e-5, 1 - 1e-5), pos); hub = (F.smooth_l1_loss(mu / s[:, None], yy / s[:, None], reduction="none") * pos).sum() / pos.sum().clamp(min=1)
            loss = bce + hub + 0.5 * aux; comp = {"bce": float(bce), "huber": float(hub), "aux": float(aux)}
        opt.zero_grad(set_to_none=True); loss.backward(); gn = float(torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)); opt.step(); step += 1
        run["loss"] += float(loss); run["n"] += 1
        if step % a.eval_every == 0:
            yc = predict(net, data, ORIGIN["choose"], a.variant); yr = predict(net, data, ORIGIN["report"], a.variant); yt = predict(net, data, 1857, a.variant)
            sc, sr, st = float(wrmsse_t(data, yc, y_choose, "choose")), float(wrmsse_t(data, yr, y_report, "report")), float(wrmsse_t(data, yt, y_trainorig, "choose"))
            rec = dict(step=step, secs=time.time() - T0, lr=opt.param_groups[0]["lr"], train_loss=run["loss"] / run["n"], grad_norm=gn, choose=sc, report_monitor=sr, train_origin=st,
                       gap=sc - st, steps_per_s=step / (time.time() - T0), peak_mem_gb=torch.cuda.max_memory_allocated() / 2**30, **comp)
            hist.append(rec); atomic_json(f"{a.out}/{name}_history.json", hist); run = {"loss": 0.0, "n": 0}
            log(f"step {step} {rec['secs']:.0f}s loss {rec['train_loss']:.4f} choose {sc:.4f} (train-origin {st:.4f}, gap {sc - st:+.4f}) report(monitor) {sr:.4f} lr {rec['lr']:.5f} gn {gn:.2f} {rec['steps_per_s']:.1f} st/s mem {rec['peak_mem_gb']:.2f}G")
            if sc < best - 1e-5: best, since = sc, 0; best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            else:
                since += 1
                if since >= a.patience: log(f"early stop: no choose improvement in {a.patience} evals"); break
    if best_state is not None: net.load_state_dict(best_state)
    for role in ORIGIN: np.save(f"{a.out}/preds/{name}_{role}.npy", predict(net, data, ORIGIN[role], a.variant).cpu().numpy().astype(np.float32))
    log(f"saved predictions; best choose {best:.4f}; steps {step}; {time.time() - T0:.0f}s")


if __name__ == "__main__":
    main()
