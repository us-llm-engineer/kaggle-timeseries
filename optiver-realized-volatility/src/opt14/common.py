"""Shared helpers: HF access (read key from env/file, never printed), atomic json, split, RMSPE."""
import json, os, time
import numpy as np

# Data location. Either OPT_DATA_DIR points at a directory with the Kaggle layout (train.csv, test.csv, book_train.parquet/stock_id=*/..., trade_train.parquet/...),
# or OPT_HF_REPO names a Hugging Face dataset repo with the same layout and HF_TOKEN (or the file named by OPT_HF_TOKEN_FILE) holds a read token.
REPO = os.environ.get("OPT_HF_REPO", "")
DATA_DIR = os.environ.get("OPT_DATA_DIR", "")
N_BINS, BIN_S = 120, 5          # sequence tensor: 120 bins of 5 seconds
SPLIT_FIT, SPLIT_CHOOSE = 0.70, 0.85   # fit < 0.70 <= choose < 0.85 <= report


def hf_token():
    t = os.environ.get("HF_TOKEN")
    if t: return t.strip()
    p = os.environ.get("OPT_HF_TOKEN_FILE")
    if p and os.path.exists(p): return open(p).read().strip()
    raise RuntimeError("no Hugging Face token: set HF_TOKEN or OPT_HF_TOKEN_FILE (or use OPT_DATA_DIR)")


def hf_get(path, cache_dir):
    """Local path of a data file: from OPT_DATA_DIR if set, else downloaded from the hub repo into cache_dir."""
    if DATA_DIR:
        return os.path.join(DATA_DIR, path)
    from huggingface_hub import hf_hub_download
    return hf_hub_download(REPO, path, repo_type="dataset", token=hf_token(), local_dir=cache_dir)


def list_files():
    if DATA_DIR:
        return [os.path.relpath(os.path.join(dp, f), DATA_DIR) for dp, _, fs in os.walk(DATA_DIR) for f in fs]
    from huggingface_hub import HfApi
    return HfApi(token=hf_token()).list_repo_files(REPO, repo_type="dataset")


def is_cached_copy():
    """True when files come from a download cache (safe to delete after use), False when they are the user's own data."""
    return not DATA_DIR


def atomic_json(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f: json.dump(obj, f)
    os.replace(tmp, path)


def split_role(time_id):
    """Deterministic role per time_id: 0 fit, 1 choose, 2 report (a time_id never straddles roles)."""
    u = ((np.asarray(time_id, dtype=np.uint64) * np.uint64(2654435761)) % np.uint64(2 ** 32)).astype(np.float64) / 2.0 ** 32
    return np.where(u < SPLIT_FIT, 0, np.where(u < SPLIT_CHOOSE, 1, 2)).astype(np.int8)


def rmspe(y, p):
    y = np.asarray(y, dtype=np.float64); p = np.asarray(p, dtype=np.float64)
    return float(np.sqrt(np.mean(((y - p) / y) ** 2)))


def best_scalar(y, p):
    """c minimising RMSPE of c*p: c = sum(p/y) / sum((p/y)^2)."""
    q = np.asarray(p, dtype=np.float64) / np.asarray(y, dtype=np.float64)
    return float(q.sum() / (q * q).sum())


def now():
    return time.time()
