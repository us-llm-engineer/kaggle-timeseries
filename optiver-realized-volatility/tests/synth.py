"""Synthetic book/trade windows with known efficient-price volatility, microstructure noise and missing seconds."""
import numpy as np, pandas as pd

def make_stock(seed, n_windows=24, p_obs=0.6, sigma=1e-3, noise=2e-4):
    rng = np.random.default_rng(seed)
    rows, trows, truth = [], [], {}
    for tid in range(n_windows):
        sig = sigma * np.exp(rng.normal(0, 0.4))
        x = np.cumsum(rng.normal(0, sig / np.sqrt(600), 600)) + np.log(100.0)
        secs = np.sort(rng.choice(600, size=int(600 * p_obs), replace=False))
        y = x[secs] + rng.normal(0, noise, len(secs))
        px = np.exp(y); half = px * rng.uniform(1e-4, 5e-4, len(secs))
        bs1, as1 = rng.integers(1, 400, len(secs)), rng.integers(1, 400, len(secs))
        bs2, as2 = rng.integers(1, 400, len(secs)), rng.integers(1, 400, len(secs))
        rows.append(pd.DataFrame({"time_id": tid + 5, "seconds_in_bucket": secs, "bid_price1": px - half, "ask_price1": px + half,
                                  "bid_price2": px - 2 * half, "ask_price2": px + 2 * half, "bid_size1": bs1, "ask_size1": as1,
                                  "bid_size2": bs2, "ask_size2": as2}))
        nt = int(rng.integers(5, 80)); ts = np.sort(rng.choice(600, nt))
        trows.append(pd.DataFrame({"time_id": tid + 5, "seconds_in_bucket": ts, "price": np.exp(x[ts] + rng.normal(0, noise, nt)),
                                   "size": rng.integers(1, 300, nt), "order_count": rng.integers(1, 6, nt)}))
        truth[tid + 5] = sig * 1.0          # integrated volatility over the (unit) window: sqrt(IV) = sig
    return pd.concat(rows, ignore_index=True), pd.concat(trows, ignore_index=True), truth
