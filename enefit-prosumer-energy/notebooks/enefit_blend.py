# Enefit - blend of V1 LightGBM-L1 (capacity-relative, lags>=48h) + V3 TiDE (capacity-relative) + V4 TiDE (window-normalised).
# Self-contained: writes its package, fits on all labelled data, then serves the time-series API one day at a time.
import os, sys, time
PKG = '/kaggle/working/pkg/enefit13'
os.makedirs(PKG, exist_ok=True)
open(os.path.join(PKG, '__init__.py'), 'w').write('')
open(os.path.join(PKG, 'data.py'), 'w').write(r'''import hashlib
import json
import os
import numpy as np
import pandas as pd
HF_REPO = os.environ.get('ENEFIT_HF_REPO', '')
WCOLS = ['temperature', 'dewpoint', 'cloudcover_low', 'cloudcover_mid', 'cloudcover_high', 'cloudcover_total', '10_metre_u_wind_component', '10_metre_v_wind_component', 'direct_solar_radiation', 'surface_solar_radiation_downwards', 'snowfall', 'total_precipitation']
WSHORT = ['temp', 'dew', 'cl_low', 'cl_mid', 'cl_high', 'cl_tot', 'wind_u', 'wind_v', 'dir_rad', 'ssr', 'snow', 'precip']
LAGS = (48, 72, 96, 120, 168, 336)
REPORT_START = pd.Timestamp('2023-04-02')
CHOOSE_START = pd.Timestamp('2023-03-05')

class Store:

    def __init__(self, root=None, repo=HF_REPO):
        self.root, self.repo, self._man = (root, repo, None)

    def manifest(self):
        if self._man is None:
            self._man = json.load(open(self.path('MANIFEST.json', verify=False)))
        return self._man

    def path(self, rel, verify=True):
        if self.root and os.path.exists(os.path.join(self.root, rel)):
            return os.path.join(self.root, rel)
        from huggingface_hub import hf_hub_download
        p = hf_hub_download(self.repo, rel, repo_type='dataset')
        if verify and rel != 'MANIFEST.json':
            want = self.manifest().get(rel, {}).get('sha256')
            h = hashlib.sha256(open(p, 'rb').read()).hexdigest()
            assert want is None or h == want, f'sha256 mismatch for {rel}'
        return p

    def read(self, rel, columns=None):
        return pd.read_parquet(self.path(rel), columns=columns)

def months_between(a, b):
    return [str(p) for p in pd.period_range(a, b, freq='M')]

def county_weather_from_frame(f, st):
    f = f.copy()
    f['forecast_datetime'] = pd.to_datetime(f['forecast_datetime'])
    f['origin_datetime'] = pd.to_datetime(f['origin_datetime'])
    keep = f.origin_datetime.dt.normalize() == f.forecast_datetime.dt.normalize() - pd.Timedelta(days=1)
    f = f[keep].copy()
    f['lat'], f['lon'] = (f.latitude.astype('float32').astype('float64').round(2), f.longitude.astype('float32').astype('float64').round(2))
    f = f.drop_duplicates(['forecast_datetime', 'lat', 'lon'])
    nat = f.groupby('forecast_datetime')[WCOLS].mean()
    nat.columns = ['n_' + c for c in WSHORT]
    f = f.merge(st, on=['lat', 'lon'], how='left')
    g = f.dropna(subset=['county']).groupby(['county', 'forecast_datetime'])[WCOLS].mean()
    g.columns = ['w_' + c for c in WSHORT]
    g = g.reset_index()
    g['county'] = g.county.astype('int16')
    miss = [c for c in range(16) if c not in set(g.county.unique())]
    if miss:
        base = nat.reset_index().rename(columns={'n_' + c: 'w_' + c for c in WSHORT})
        for c in miss:
            extra = base.copy()
            extra['county'] = np.int16(c)
            g = pd.concat([g, extra[g.columns]], ignore_index=True)
    g = g.merge(nat.reset_index(), on='forecast_datetime', how='left').rename(columns={'forecast_datetime': 'datetime'})
    for c in g.columns:
        if c.startswith(('w_', 'n_')):
            g[c] = g[c].astype('float32')
    return g

def station_map(store):
    st = store.read('weather_station_to_county_mapping.parquet')
    return st.assign(lat=st.latitude.round(2), lon=st.longitude.round(2))[['lat', 'lon', 'county']]

def county_weather(store, months):
    st = station_map(store)
    out = []
    for m in months:
        try:
            f = store.read(f'forecast_weather/{m}.parquet')
        except Exception:
            continue
        out.append(county_weather_from_frame(f, st))
    return pd.concat(out, ignore_index=True)

class CsvStore:

    def __init__(self, root):
        self.root, self._fw = (root, None)

    def read(self, rel, columns=None):
        r = self.root
        if rel == 'train.parquet':
            return pd.read_csv(f'{r}/train.csv', parse_dates=['datetime'])
        if rel == 'client.parquet':
            return pd.read_csv(f'{r}/client.csv', parse_dates=['date'])
        if rel == 'electricity_prices.parquet':
            return pd.read_csv(f'{r}/electricity_prices.csv', parse_dates=['forecast_date', 'origin_date'])
        if rel == 'weather_station_to_county_mapping.parquet':
            return pd.read_csv(f'{r}/weather_station_to_county_mapping.csv')
        if rel.startswith('forecast_weather/'):
            if self._fw is None:
                self._fw = pd.read_csv(f'{r}/forecast_weather.csv', parse_dates=['origin_datetime', 'forecast_datetime'])
                for c in WCOLS + ['latitude', 'longitude']:
                    self._fw[c] = self._fw[c].astype('float32')
            m = rel.split('/')[1].split('.')[0]
            return self._fw[self._fw.forecast_datetime.dt.strftime('%Y-%m') == m]
        raise FileNotFoundError(rel)

def load_panel(store, first_month='2021-09', last_month='2023-05'):
    tr = store.read('train.parquet')
    tr = tr.drop(columns=['row_id'], errors='ignore')
    tr['date'] = tr.datetime.dt.normalize()
    cl = store.read('client.parquet')[['county', 'is_business', 'product_type', 'date', 'installed_capacity', 'eic_count']]
    cl = cl.assign(date=cl.date + pd.Timedelta(days=2))
    tr = tr.merge(cl, on=['county', 'is_business', 'product_type', 'date'], how='left')
    tr = tr.sort_values(['prediction_unit_id', 'is_consumption', 'datetime']).reset_index(drop=True)
    for c in ('installed_capacity', 'eic_count'):
        tr[c] = tr.groupby('prediction_unit_id')[c].transform(lambda s: s.ffill().bfill())
    tr['scale'] = np.where(tr.is_consumption == 1, tr.eic_count, tr.installed_capacity).astype('float32')
    tr['scale'] = tr['scale'].clip(lower=1.0)
    ep = store.read('electricity_prices.parquet')[['forecast_date', 'euros_per_mwh']]
    ep = ep.rename(columns={'forecast_date': 'datetime', 'euros_per_mwh': 'price_lag24'})
    ep['datetime'] = ep.datetime + pd.Timedelta(hours=24)
    tr = tr.merge(ep, on='datetime', how='left')
    w = county_weather(store, months_between(first_month, last_month))
    tr = tr.merge(w, on=['county', 'datetime'], how='left')
    dt = tr.datetime
    tr['hour'] = dt.dt.hour.astype('int8')
    tr['dow'] = dt.dt.dayofweek.astype('int8')
    tr['month'] = dt.dt.month.astype('int8')
    doy = dt.dt.dayofyear.values
    tr['doy_s'] = np.sin(2 * np.pi * doy / 365.25).astype('float32')
    tr['doy_c'] = np.cos(2 * np.pi * doy / 365.25).astype('float32')
    tr['y'] = tr.target.astype('float32')
    tr['y_norm'] = (tr.target / tr.scale).astype('float32')
    return tr

def series_grid(panel):
    t0 = panel.datetime.min()
    T = int((panel.datetime.max() - t0) / pd.Timedelta(hours=1)) + 1
    keys = panel[['prediction_unit_id', 'is_consumption']].drop_duplicates().sort_values(['prediction_unit_id', 'is_consumption'])
    sid = {(int(a), int(b)): i for i, (a, b) in enumerate(keys.itertuples(index=False))}
    s = np.fromiter((sid[int(a), int(b)] for a, b in zip(panel.prediction_unit_id.values, panel.is_consumption.values)), dtype=np.int32, count=len(panel))
    ti = ((panel.datetime - t0) / pd.Timedelta(hours=1)).astype(int).values
    an = np.full((len(sid), T), np.nan, dtype=np.float32)
    ar = np.full((len(sid), T), np.nan, dtype=np.float32)
    an[s, ti] = panel.y_norm.values
    ar[s, ti] = panel.y.values
    return (an, ar, s, ti, sid, t0)

def add_lag_features(panel, an, s, ti):
    out = {}
    for k in LAGS:
        j = ti - k
        v = np.where(j >= 0, an[s, np.clip(j, 0, None)], np.nan)
        out[f'lag{k}'] = v.astype('float32')
    stack = np.stack([np.where(ti - 24 * d >= 0, an[s, np.clip(ti - 24 * d, 0, None)], np.nan) for d in range(2, 9)])
    with np.errstate(all='ignore'):
        out['same_hour_mean7'] = np.nanmean(stack, 0).astype('float32')
        out['same_hour_std7'] = np.nanstd(stack, 0).astype('float32')
        out['same_hour_max7'] = np.nanmax(stack, 0).astype('float32')
    day0 = ti // 24 * 24
    lvl = []
    for d in (2, 3, 4, 5, 6, 7, 8):
        cols = np.stack([np.where(day0 - 24 * d + h >= 0, an[s, np.clip(day0 - 24 * d + h, 0, None)], np.nan) for h in range(24)])
        with np.errstate(all='ignore'):
            lvl.append(np.nanmean(cols, 0))
    lvl = np.stack(lvl)
    with np.errstate(all='ignore'):
        out['day_level_d2'] = lvl[0].astype('float32')
        out['week_level'] = np.nanmean(lvl, 0).astype('float32')
    for k, v in out.items():
        panel[k] = v
    return panel
STATIC = ['county', 'is_business', 'product_type', 'is_consumption', 'prediction_unit_id']
CAL = ['hour', 'dow', 'month', 'doy_s', 'doy_c']
WEATHER = ['w_' + c for c in WSHORT] + ['n_temp', 'n_ssr', 'n_cl_tot']
SCALE = ['installed_capacity', 'eic_count']
LAGCOLS = [f'lag{k}' for k in LAGS] + ['same_hour_mean7', 'same_hour_std7', 'same_hour_max7', 'day_level_d2', 'week_level']
''')
open(os.path.join(PKG, 'gbdt.py'), 'w').write(r'''import json
import os
import time
import lightgbm as lgb
import numpy as np
import pandas as pd
from .data import CAL, LAGCOLS, SCALE, STATIC, WEATHER

def _atomic_json(path, obj):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f)
    os.replace(tmp, path)

def v1_features():
    return STATIC + CAL + WEATHER + SCALE + ['price_lag24'] + LAGCOLS
PARAMS_L1 = dict(objective='regression_l1', learning_rate=0.05, num_leaves=127, min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, max_bin=255, verbose=-1, num_threads=0)

def _dataset(df, feats, weight=True):
    X = df[feats]
    return lgb.Dataset(X, label=df.y_norm.values, weight=df.scale.values if weight else None, categorical_feature=['prediction_unit_id'] if 'prediction_unit_id' in feats else 'auto', free_raw_data=False)
''')
open(os.path.join(PKG, 'nn.py'), 'w').write(r'''import json
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from .data import WEATHER
L, H = (168, 24)
GAP = 24

def _atomic_json(path, obj):
    tmp = path + '.tmp'
    json.dump(obj, open(tmp, 'w'))
    os.replace(tmp, path)

def build_tensors(panel, an, ar, s, ti, sid, t0):
    S, T = an.shape
    wx = np.zeros((S, T, len(WEATHER)), dtype=np.float32)
    wvals = panel[WEATHER].values.astype('float32')
    mu = np.nanmean(wvals, 0)
    wvals = np.where(np.isnan(wvals), mu, wvals)
    wx[s, ti] = wvals
    sd = wx.reshape(-1, wx.shape[-1]).std(0) + 1e-06
    wx = (wx - mu) / sd
    sc = np.ones((S, T), dtype=np.float32)
    sc[s, ti] = panel.scale.values
    ser = panel.drop_duplicates(['prediction_unit_id', 'is_consumption']).copy()
    ser['sid'] = [sid[int(a), int(b)] for a, b in zip(ser.prediction_unit_id, ser.is_consumption)]
    ser = ser.sort_values('sid')
    static = ser[['county', 'is_business', 'product_type', 'is_consumption']].values.astype('int64')
    return dict(an=an, ar=ar, wx=wx, sc=sc, static=static, S=S, T=T, t0=t0, w_mu=mu, w_sd=sd)

def day_samples(td, first_day=8, last_day=None):
    T, S = (td['T'], td['S'])
    nd = T // 24
    last_day = nd - 1 if last_day is None else last_day
    sidx, didx = np.meshgrid(np.arange(S), np.arange(first_day, last_day + 1), indexing='ij')
    sidx, didx = (sidx.ravel(), didx.ravel())
    ds = didx * 24
    obs = np.stack([~np.isnan(td['ar'][sidx, ds + h]) for h in range(24)], 1).any(1)
    return (sidx[obs], didx[obs])

def cal_feats(t0, didx):
    base = pd.Timestamp(t0) + pd.to_timedelta(didx * 24, unit='h')
    hrs = np.arange(24)
    hs, hc = (np.sin(2 * np.pi * hrs / 24), np.cos(2 * np.pi * hrs / 24))
    dow = base.dayofweek.values
    doy = base.dayofyear.values
    n = len(didx)
    f = np.zeros((n, 24, 6), dtype=np.float32)
    f[:, :, 0], f[:, :, 1] = (hs, hc)
    f[:, :, 2] = np.sin(2 * np.pi * dow / 7)[:, None]
    f[:, :, 3] = np.cos(2 * np.pi * dow / 7)[:, None]
    f[:, :, 4] = np.sin(2 * np.pi * doy / 365.25)[:, None]
    f[:, :, 5] = np.cos(2 * np.pi * doy / 365.25)[:, None]
    return f

def batch_arrays(td, sidx, didx, mode):
    ds = didx * 24
    ar_ = td['an'] if mode == 'rel' else td['ar']
    lb = np.stack([ar_[s, d - (L + GAP):d - GAP] for s, d in zip(sidx, ds)])
    tg = np.stack([td['ar'][s, d:d + H] for s, d in zip(sidx, ds)])
    sc = np.stack([td['sc'][s, d:d + H] for s, d in zip(sidx, ds)])
    wx = np.stack([td['wx'][s, d:d + H] for s, d in zip(sidx, ds)])
    return (lb.astype('float32'), tg.astype('float32'), sc.astype('float32'), wx.astype('float32'))

class Res(nn.Module):

    def __init__(self, i, h, o, p=0.1, ln=True):
        super().__init__()
        self.f1, self.f2, self.sk, self.dp = (nn.Linear(i, h), nn.Linear(h, o), nn.Linear(i, o), nn.Dropout(p))
        self.ln = nn.LayerNorm(o) if ln and o > 1 else nn.Identity()

    def forward(self, x):
        return self.ln(self.dp(self.f2(torch.relu(self.f1(x)))) + self.sk(x))

class TiDE(nn.Module):

    def __init__(self, nw, nser, mode, hid=1024, r=4, p=8, emb=16, n_static=(16, 2, 4, 2), drop=0.5):
        super().__init__()
        self.mode = mode
        self.ser_emb = nn.Embedding(nser, emb)
        self.cnt_emb = nn.Embedding(n_static[0], 4)
        self.proj = Res(nw + 6, 64, r, drop)
        extra = 2 if mode == 'raw' else 1
        enc_in = L + H * r + emb + 4 + 2 + extra
        self.enc = nn.Sequential(Res(enc_in, hid, hid, drop), Res(hid, hid, hid, drop))
        self.dec = Res(hid, hid, H * p, drop)
        self.tdec = Res(p + r, 64, 1, drop)
        self.skip = nn.Linear(L, H)
        self.p, self.r = (p, r)

    def forward(self, lb, wx, cal, sid, static, extra):
        b = lb.shape[0]
        x = self.proj(torch.cat([wx, cal], -1))
        e = torch.cat([lb, x.reshape(b, -1), self.ser_emb(sid), self.cnt_emb(static[:, 0]), static[:, 1:3].float(), extra], 1)
        d = self.dec(self.enc(e)).reshape(b, H, self.p)
        out = self.tdec(torch.cat([d, x], -1)).squeeze(-1)
        return out + self.skip(lb)

class Runner:

    def __init__(self, td, mode, device, seed, out_dir, name, drop=0.5):
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.td, self.mode, self.dev, self.out, self.name = (td, mode, device, out_dir, name)
        self.model = TiDE(td['wx'].shape[-1], td['S'], mode, drop=drop).to(device)
        self.static = torch.tensor(td['static'], device=device)

    def _prep(self, sidx, didx):
        lb, tg, sc, wx = batch_arrays(self.td, sidx, didx, self.mode)
        cal = cal_feats(self.td['t0'], didx)
        nanm = np.isnan(lb)
        t = lambda a, dt=torch.float32: torch.as_tensor(a, dtype=dt, device=self.dev)
        if self.mode == 'raw':
            cnt = (~nanm).sum(1).clip(min=1)
            mean = np.where(nanm, 0, lb).sum(1) / cnt
            var = np.where(nanm, 0, (lb - mean[:, None]) ** 2).sum(1) / cnt
            std = np.sqrt(var) + 1.0
            lbn = np.where(nanm, 0, (lb - mean[:, None]) / std[:, None]).astype('float32')
            extra = np.stack([np.log1p(np.abs(mean)), np.log1p(std)], 1).astype('float32')
            aux = (t(mean), t(std))
        else:
            lbn = np.where(nanm, 0, lb).astype('float32')
            extra = np.log(sc[:, :1] + 1.0).astype('float32')
            aux = (torch.zeros(len(lb), device=self.dev), torch.ones(len(lb), device=self.dev))
        return dict(lb=t(lbn), wx=t(wx), cal=t(cal), sid=t(sidx, torch.long), static=self.static[t(sidx, torch.long)], extra=t(extra), tg=t(tg), sc=t(sc), mean=aux[0], std=aux[1])

    def materialize(self, sidx, didx, chunk=8192):
        parts = [self._prep(sidx[i:i + chunk], didx[i:i + chunk]) for i in range(0, len(sidx), chunk)]
        if not parts:
            return None
        return {k: torch.cat([p[k] for p in parts], 0) for k in parts[0]}
''')
open(os.path.join(PKG, 'nn_ens.py'), 'w').write(r'''import json
import os
import time
import numpy as np
import torch
import torch.nn as nn
from .nn import H, L, Runner, _atomic_json

class EL(nn.Module):

    def __init__(self, E, i, o):
        super().__init__()
        k = 1.0 / np.sqrt(i)
        self.w = nn.Parameter(torch.empty(E, i, o).uniform_(-k, k))
        self.b = nn.Parameter(torch.empty(E, 1, o).uniform_(-k, k))

    def forward(self, x):
        return torch.baddbmm(self.b, x, self.w)

class ERes(nn.Module):

    def __init__(self, E, i, h, o, keep, ln=True):
        super().__init__()
        self.f1, self.f2, self.sk = (EL(E, i, h), EL(E, h, o), EL(E, i, o))
        self.register_buffer('keep', keep.clone())
        self.ln = ln and o > 1
        if self.ln:
            self.g, self.bb = (nn.Parameter(torch.ones(E, 1, o)), nn.Parameter(torch.zeros(E, 1, o)))

    def forward(self, x):
        h = self.f2(torch.relu(self.f1(x)))
        if self.training:
            h = h * (torch.rand_like(h) < self.keep).to(h.dtype) / self.keep
        y = (h + self.sk(x)).float()
        if self.ln:
            y = (y - y.mean(-1, keepdim=True)) / torch.sqrt(y.var(-1, unbiased=False, keepdim=True) + 1e-05) * self.g + self.bb
        return y

class EnsTiDE(nn.Module):

    def __init__(self, E, nw, nser, mode, drops, hid=1024, r=4, p=8, emb=16):
        super().__init__()
        keep = torch.tensor([1.0 - d for d in drops]).view(E, 1, 1)
        self.register_buffer('keep', keep)
        self.E, self.mode, self.p, self.r = (E, mode, p, r)
        self.ser_emb = nn.Parameter(torch.randn(E, nser, emb) * 0.1)
        self.cnt_emb = nn.Parameter(torch.randn(E, 16, 4) * 0.1)
        extra = 2 if mode == 'raw' else 1
        enc_in = L + H * r + emb + 4 + 2 + extra
        self.proj = ERes(E, nw + 6, 64, r, keep)
        self.enc1, self.enc2 = (ERes(E, enc_in, hid, hid, keep), ERes(E, hid, hid, hid, keep))
        self.dec = ERes(E, hid, hid, H * p, keep)
        self.tdec = ERes(E, p + r, 64, 1, keep, ln=False)
        self.skip = EL(E, L, H)

    def forward(self, lb, wx, cal, sid, static, extra):
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
        return torch.amp.GradScaler('cuda', enabled=enabled)
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=enabled)

def _gather(data, idx):
    return {k: v[idx] for k, v in data.items()}

def train_ensemble(td, mode, device, drops, seeds, tr, ch, rp, out_dir, name, epochs=100, bs=512, lr=0.001, budget_sec=None, patience=12, min_rel=0.002, overfit_gap=0.05, overfit_run=5, min_epochs=20, amp=True, final=False):
    os.makedirs(out_dir, exist_ok=True)
    members = [(d, s) for d in drops for s in seeds]
    E = len(members)
    torch.manual_seed(1234 + sum(seeds))
    helper = Runner(td, mode, device, seeds[0], out_dir, name)
    dtr = helper.materialize(*tr)
    dch = helper.materialize(*ch) if ch is not None and len(ch[0]) else None
    drp = helper.materialize(*rp) if rp is not None and len(rp[0]) else None
    del helper.model
    n = dtr['lb'].shape[0]
    model = EnsTiDE(E, td['wx'].shape[-1], td['S'], mode, [m[0] for m in members]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0001)
    const = float(torch.nanmean(torch.abs(dtr['tg'][:, 0]))) + 1.0
    spe = int(np.ceil(n / bs))
    use_amp = bool(amp and device == 'cuda')
    scaler = _scaler(use_amp)

    def fwd(b):
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
            out = model(b['lb'], b['wx'], b['cal'], b['sid'], b['static'], b['extra'])
        out = out.float()
        return out * b['sc'] if mode == 'rel' else out * b['std'].unsqueeze(-1) + b['mean'].unsqueeze(-1)

    def step(idx=None):
        if idx is None:
            idx = torch.randint(0, n, (E, bs), device=device)
        b = _gather(dtr, idx)
        pred = fwd(b)
        m = ~torch.isnan(b['tg'])
        per = (torch.abs(pred - torch.nan_to_num(b['tg'])) * m).sum((1, 2)) / m.sum((1, 2)).clamp(min=1) / const
        opt.zero_grad(set_to_none=True)
        scaler.scale(per.sum()).backward()
        scaler.unscale_(opt)
        sq = torch.zeros(E, device=device)
        for p_ in model.parameters():
            if p_.grad is not None and p_.shape[0] == E:
                sq += p_.grad.pow(2).flatten(1).sum(1)
        scale = (1.0 / (sq.sqrt() + 1e-06)).clamp(max=1.0)
        for p_ in model.parameters():
            if p_.grad is not None and p_.shape[0] == E:
                p_.grad.mul_(scale.view(E, *[1] * (p_.dim() - 1)))
        scaler.step(opt)
        scaler.update()
        return per.detach()

    def predict(data, bs_=2048):
        model.eval()
        outs = []
        with torch.no_grad():
            for i in range(0, data['lb'].shape[0], bs_):
                sl = {k: v[i:i + bs_].unsqueeze(0).expand(E, *v[i:i + bs_].shape) for k, v in data.items()}
                outs.append(fwd(sl).clamp(min=0))
        return torch.cat(outs, 1)

    def mae(data):
        if data is None:
            return np.full(E, np.nan)
        p = predict(data)
        m = (~torch.isnan(data['tg'])).unsqueeze(0)
        return ((torch.abs(p - torch.nan_to_num(data['tg']).unsqueeze(0)) * m).sum((1, 2)) / m.sum((1, 2))).cpu().numpy()
    epochs_eff = epochs
    if budget_sec:
        sd = {k: v.clone() for k, v in model.state_dict().items()}
        model.train()
        for _ in range(6):
            step()
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(15):
            step()
        torch.cuda.synchronize()
        per_epoch = (time.time() - t0) / 15 * spe * 1.2 + 1.0
        epochs_eff = int(max(5, min(epochs, budget_sec // per_epoch)))
        model.load_state_dict(sd)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0001)
        scaler = _scaler(use_amp)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs_eff * spe)
    best = np.full(E, 1e+18)
    sm = np.full(E, np.nan)
    best_sm = np.full(E, 1e+18)
    since = np.zeros(E, int)
    over = np.zeros(E, int)
    active = np.ones(E, bool)
    reason = ['max_epochs'] * E
    stopped_at = np.zeros(E, int)
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    last_train = np.full(E, np.nan)
    hist, t0 = ([], time.time())
    for ep in range(epochs_eff):
        model.train()
        tot = torch.zeros(E, device=device)
        perms = torch.rand(E, n, device=device).argsort(1)
        for i in range(spe):
            tot += step(perms[:, i * bs:(i + 1) * bs])
            sch.step()
        trn = (tot / spe).cpu().numpy()
        cm, rm = (mae(dch), mae(drp)) if not final else (trn.copy(), np.full(E, np.nan))
        improved = (cm < best) & active if not final else np.zeros(E, bool)
        if improved.any():
            for k, v in model.state_dict().items():
                if v.dim() > 0 and v.shape[0] == E:
                    sel = torch.as_tensor(improved, device=device).view(E, *[1] * (v.dim() - 1))
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
            if active[e] and (not final):
                if over[e] >= overfit_run:
                    active[e], reason[e], stopped_at[e] = (False, 'overfitting', ep + 1)
                elif since[e] >= patience and ep + 1 >= min_epochs:
                    active[e], reason[e], stopped_at[e] = (False, 'plateau', ep + 1)
        hist.append({'epoch': ep + 1, 'sec': round(time.time() - t0, 1), 'train': trn.round(5).tolist(), 'choose_mae': cm.round(2).tolist(), 'report_mae_monitor_only': rm.round(2).tolist(), 'active': active.tolist(), 'lr': sch.get_last_lr()[0]})
        _atomic_json(os.path.join(out_dir, f'{name}_history.json'), {'members': [{'drop': d, 'seed': s} for d, s in members], 'rows': hist, 'best_choose': best.tolist(), 'stop_reason': reason, 'epochs_planned': epochs_eff})
        if improved.any():
            torch.save(best_state, os.path.join(out_dir, f'{name}_best.pt'))
        if (ep + 1) % 5 == 0:
            torch.save(model.state_dict(), os.path.join(out_dir, f'{name}_last.pt'))
        if not active.any():
            break
    stopped_at = np.where(stopped_at == 0, len(hist), stopped_at)
    if final:
        info_final = {'members': [{'drop': d, 'seed': s_, 'stop_reason': 'fixed_epochs', 'stopped_epoch': len(hist)} for d, s_ in members], 'epochs_run': len(hist), 'epochs_planned': epochs_eff, 'wall_sec': round(time.time() - t0, 1), 'peak_mem_mib': 0.0}
        model.eval()
        return (None, info_final, model)
    model.load_state_dict(best_state)
    ev = {k: torch.cat([dch[k], drp[k]], 0) for k in dch}
    preds = predict(ev).cpu().numpy()
    info = {'members': [{'drop': d, 'seed': s, 'best_choose': float(best[i]), 'stop_reason': reason[i], 'stopped_epoch': int(stopped_at[i])} for i, (d, s) in enumerate(members)], 'epochs_run': len(hist), 'epochs_planned': epochs_eff, 'peak_mem_mib': torch.cuda.max_memory_allocated() / 2 ** 20 if device == 'cuda' else 0.0, 'wall_sec': round(time.time() - t0, 1)}
    model.eval()
    return (preds, info, model)

def predict_members(model, data, mode, device, amp=True, bs=2048):
    model.eval()
    E = model.E
    outs = []
    use_amp = bool(amp and device == 'cuda')
    with torch.no_grad():
        for i in range(0, data['lb'].shape[0], bs):
            b = {k: v[i:i + bs].unsqueeze(0).expand(E, *v[i:i + bs].shape) for k, v in data.items()}
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                out = model(b['lb'], b['wx'], b['cal'], b['sid'], b['static'], b['extra'])
            out = out.float()
            out = out * b['sc'] if mode == 'rel' else out * b['std'].unsqueeze(-1) + b['mean'].unsqueeze(-1)
            outs.append(out.clamp(min=0))
    return torch.cat(outs, 1)
''')
open(os.path.join(PKG, 'serve.py'), 'w').write(r'''import time
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from . import data as D
from . import gbdt as G
from . import nn as N
from . import nn_ens as NE
EXTRA_DAYS = 420

def _pad(a, extra, fill):
    shape = list(a.shape)
    shape[1] = extra
    return np.concatenate([a, np.full(shape, fill, dtype=a.dtype)], 1)

def train_final(store, quick=False, v1_rounds=(67, 592), dl_epochs=(46, 40), log=print):
    t0 = time.time()
    panel = D.load_panel(store)
    an, ar, s, ti, sid, t0g = D.series_grid(panel)
    panel = D.add_lag_features(panel, an, s, ti)
    log(f'panel {len(panel)} rows, {len(sid)} series, features built in {time.time() - t0:.0f}s')
    feats = G.v1_features()
    boosters = {}
    valid = panel.y_norm.notna()
    for c, r in zip((0, 1), v1_rounds):
        df = panel[valid & (panel.is_consumption == c)]
        boosters[c] = lgb.train(G.PARAMS_L1, G._dataset(df, feats), num_boost_round=30 if quick else r)
        log(f'V1 booster c{c}: {len(df)} rows, {(30 if quick else r)} rounds, {time.time() - t0:.0f}s')
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    td = N.build_tensors(panel, an, ar, s, ti, sid, t0g)
    sidx, didx = N.day_samples(td)
    models = {}
    drops, seeds = ([0.5], [0]) if quick else ([0.5, 0.2], [0, 1, 2, 3])
    for v, mode, ep in (('v3', 'rel', dl_epochs[0]), ('v4', 'raw', dl_epochs[1])):
        _, info, m = NE.train_ensemble(td, mode, dev, drops, seeds, (sidx, didx), None, None, '/tmp/enefit_final', v, epochs=2 if quick else ep, budget_sec=None, final=True)
        models[v] = (m, mode)
        n_ep, wall_s, n_mem = (info['epochs_run'], info['wall_sec'], len(info['members']))
        log(f'{v}: {n_ep} epochs, {wall_s:.0f}s, members {n_mem}')
    for k, fill in (('an', np.nan), ('ar', np.nan), ('sc', 1.0)):
        td[k] = _pad(td[k], EXTRA_DAYS * 24, fill)
    td['wx'] = _pad(td['wx'], EXTRA_DAYS * 24, 0.0)
    td['T'] = td['an'].shape[1]
    st = D.station_map(store)
    last_client = {}
    cl = store.read('client.parquet')
    for r in cl.sort_values('date').itertuples():
        last_client[int(r.county), int(r.is_business), int(r.product_type)] = (float(r.installed_capacity), float(r.eic_count))
    ep = store.read('electricity_prices.parquet')
    price = {pd.Timestamp(r.forecast_date): float(r.euros_per_mwh) for r in ep.itertuples()}
    return Server(td, sid, boosters, models, dev, st, last_client, price, t0g)

class Server:

    def __init__(self, td, sid, boosters, models, dev, st, last_client, price, t0):
        self.td, self.sid, self.boosters, self.models, self.dev = (td, sid, boosters, models, dev)
        self.st, self.last_client, self.price, self.t0 = (st, last_client, price, pd.Timestamp(t0))
        self.helpers = {}
        self.n_calls = 0

    def _tidx(self, dt):
        return ((pd.to_datetime(dt) - self.t0) / pd.Timedelta(hours=1)).astype(int).values

    def _series(self, df):
        return np.array([self.sid.get((int(u), int(c)), -1) for u, c in zip(df.prediction_unit_id, df.is_consumption)])

    def update_revealed(self, rev):
        if rev is None or len(rev) == 0:
            return
        r = rev.dropna(subset=['target'])
        s, t = (self._series(r), self._tidx(r.datetime))
        ok = (s >= 0) & (t >= 0) & (t < self.td['T'])
        s, t, y = (s[ok], t[ok], r.target.values[ok].astype('float32'))
        self.td['ar'][s, t] = y
        self.td['an'][s, t] = y / self.td['sc'][s, t]

    def update_client(self, client):
        for r in client.itertuples():
            self.last_client[int(r.county), int(r.is_business), int(r.product_type)] = (float(r.installed_capacity), float(r.eic_count))

    def update_price(self, ep):
        for r in ep.itertuples():
            self.price[pd.Timestamp(r.forecast_date)] = float(r.euros_per_mwh)

    def predict_day(self, test, forecast_weather):
        test = test.copy()
        test['datetime'] = pd.to_datetime(test.prediction_datetime)
        w = D.county_weather_from_frame(forecast_weather, self.st)
        test = test.merge(w, on=['county', 'datetime'], how='left')
        test['county'] = test.county.astype('int16')
        keys = list(zip(test.county, test.is_business, test.product_type))
        cap = np.array([self.last_client.get((int(a), int(b), int(c)), (np.nan, np.nan))[0] for a, b, c in keys])
        eic = np.array([self.last_client.get((int(a), int(b), int(c)), (np.nan, np.nan))[1] for a, b, c in keys])
        test['installed_capacity'], test['eic_count'] = (cap, eic)
        scale = np.where(test.is_consumption.values == 1, eic, cap)
        test['scale'] = np.clip(np.nan_to_num(scale, nan=1.0), 1.0, None).astype('float32')
        test['price_lag24'] = [self.price.get(t - pd.Timedelta(hours=24), np.nan) for t in test.datetime]
        dt = test.datetime
        test['hour'], test['dow'], test['month'] = (dt.dt.hour.astype('int8'), dt.dt.dayofweek.astype('int8'), dt.dt.month.astype('int8'))
        doy = dt.dt.dayofyear.values
        test['doy_s'], test['doy_c'] = (np.sin(2 * np.pi * doy / 365.25).astype('float32'), np.cos(2 * np.pi * doy / 365.25).astype('float32'))
        s, t = (self._series(test), self._tidx(test.datetime))
        known = (s >= 0) & (t >= 0) & (t < self.td['T'])
        self.td['sc'][s[known], t[known]] = test.scale.values[known]
        wv = test[D.WEATHER].values.astype('float32')
        wv = np.where(np.isnan(wv), self.td['w_mu'], wv)
        self.td['wx'][s[known], t[known]] = (wv[known] - self.td['w_mu']) / self.td['w_sd']
        ti_known = t.copy()
        test = D.add_lag_features(test, self.td['an'], np.where(known, s, 0), np.where(known, t, 0))
        preds = {}
        p1 = np.full(len(test), np.nan)
        for c, b in self.boosters.items():
            m = (test.is_consumption.values == c) & known
            if m.any():
                p1[m] = b.predict(test.loc[m, G.v1_features()]) * test.scale.values[m]
        preds['v1'] = np.clip(p1, 0, None)
        day = int(np.floor(t[known].min() / 24)) if known.any() else None
        if day is not None:
            sidx = np.unique(s[known])
            didx = np.full(len(sidx), day)
            for v, (model, mode) in self.models.items():
                if v not in self.helpers:
                    self.helpers[v] = N.Runner(self.td, mode, self.dev, 0, '/tmp', f'serve_{v}')
                    del self.helpers[v].model
                data = self.helpers[v].materialize(sidx, didx)
                out = NE.predict_members(model, data, mode, self.dev).mean(0).cpu().numpy()
                pos = {int(k): i for i, k in enumerate(sidx)}
                arr = np.full(len(test), np.nan)
                for j in np.where(known)[0]:
                    arr[j] = out[pos[int(s[j])], int(t[j] - day * 24)]
                preds[v] = arr
        stack = np.stack([preds[k] for k in preds])
        blend = np.nanmean(stack, 0) if np.isfinite(stack).any() else np.zeros(len(test))
        blend = np.nan_to_num(blend, nan=0.0)
        self.n_calls += 1
        self.last_features = test
        return (test.row_id.values, np.clip(blend, 0, None), {k: v for k, v in preds.items()})

def run_api(env, server, log=print, max_days=None):
    n = 0
    for test, revealed_targets, client, hist_w, forecast_weather, el_prices, gas_prices, sample_prediction in env.iter_test():
        try:
            server.update_revealed(revealed_targets)
            server.update_client(client)
            server.update_price(el_prices)
            rid, yhat, _ = server.predict_day(test, forecast_weather)
            out = pd.DataFrame({'row_id': rid, 'target': yhat})
            sample_prediction = sample_prediction.drop(columns=['target']).merge(out, on='row_id', how='left')
            sample_prediction['target'] = sample_prediction.target.fillna(0.0)
        except Exception as e:
            log(f'day {n}: fallback to zeros ({type(e).__name__}: {str(e)[:200]})')
            sample_prediction = sample_prediction.copy()
            sample_prediction['target'] = 0.0
        env.predict(sample_prediction[['row_id', 'target']] if 'data_block_id' not in sample_prediction else sample_prediction)
        n += 1
        if n % 10 == 0:
            log(f'served {n} days')
        if max_days and n >= max_days:
            break
    return n
''')
sys.path.insert(0, '/kaggle/working/pkg')
import torch
print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), flush=True)
from enefit13 import data as D, serve as S
import enefit
ROOT = '/kaggle/input/predict-energy-behavior-of-prosumers'
t0 = time.time()
server = S.train_final(D.CsvStore(ROOT), quick=False, log=lambda m: print(m, flush=True))
print('fit done in', round(time.time() - t0), 's', flush=True)
env = enefit.make_env()
n = S.run_api(env, server, log=lambda m: print(m, flush=True))
print('served days:', n, 'total', round(time.time() - t0), 's', flush=True)
