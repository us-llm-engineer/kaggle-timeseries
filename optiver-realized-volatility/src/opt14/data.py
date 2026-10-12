"""Load the merged feature table, attach targets and roles, prepare NN inputs (robust transform + fit-set standardisation)."""
import numpy as np, pandas as pd
from .common import split_role

ID_COLS = ["stock_id", "time_id"]


def load_table(feature_path, train_csv):
    df = pd.read_parquet(feature_path)
    tr = pd.read_csv(train_csv)
    df = df.merge(tr, on=ID_COLS, how="inner")
    df["role"] = split_role(df["time_id"].to_numpy())
    return df.sort_values(ID_COLS).reset_index(drop=True)


def feature_columns(df, with_cross=True):
    cols = [c for c in df.columns if c not in ID_COLS + ["target", "role"]]
    if not with_cross:
        cols = [c for c in cols if not c.startswith("x_") and c != "stock_logrv_mean"]
    return cols


class Prep:
    """Signed-log compression of heavy tails then z-score, both fitted on the fit role only (no leakage)."""
    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        self.scale = np.maximum(np.nanmedian(np.abs(X), axis=0), 1e-12)
        T = self._t(X)
        self.mu, self.sd = np.nanmean(T, axis=0), np.maximum(np.nanstd(T, axis=0), 1e-6)
        return self

    def _t(self, X):
        return np.sign(X) * np.log1p(np.abs(X) / self.scale)

    def transform(self, X):
        T = (self._t(np.asarray(X, dtype=np.float64)) - self.mu) / self.sd
        return np.clip(np.nan_to_num(T, nan=0.0), -8, 8).astype(np.float32)
