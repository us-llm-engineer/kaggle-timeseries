"""V3 (per-sample: sequence CNN + aggregated features) and V4 (set transformer over the stocks of one time_id).
Both predict log(y) = log(rv_ref) + delta + b, so training starts near the plain-RV baseline and learns a correction."""
import math
import torch, torch.nn as nn

N_STOCK_IDS = 127


class V3(nn.Module):
    def __init__(self, n_feat, n_ch=13, d=128, drop=0.1):
        super().__init__()
        self.emb = nn.Embedding(N_STOCK_IDS, 16)
        self.conv = nn.Sequential(
            nn.Conv1d(n_ch, 64, 5, padding=2), nn.BatchNorm1d(64), nn.GELU(),
            nn.Conv1d(64, 96, 5, stride=2, padding=2), nn.BatchNorm1d(96), nn.GELU(),
            nn.Conv1d(96, 128, 5, stride=2, padding=2), nn.BatchNorm1d(128), nn.GELU())
        self.feat = nn.Sequential(nn.Linear(n_feat, 256), nn.GELU(), nn.Dropout(drop))
        self.head = nn.Sequential(nn.Linear(256 + 256 + 16, 256), nn.GELU(), nn.Dropout(drop), nn.Linear(256, 1))
        self.bias = nn.Parameter(torch.tensor(math.log(0.84)))

    def forward(self, x, seq, stock, log_ref):
        h = self.conv(seq.transpose(1, 2))
        h = torch.cat([h.mean(2), h.amax(2)], 1)
        z = torch.cat([h, self.feat(x), self.emb(stock)], 1)
        return log_ref + self.head(z).squeeze(1) + self.bias


class V4(nn.Module):
    def __init__(self, n_feat, d=128, heads=4, layers=3, drop=0.1):
        super().__init__()
        self.emb = nn.Embedding(N_STOCK_IDS, 16)
        self.inp = nn.Linear(n_feat + 16, d)
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, drop, batch_first=True, norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))
        self.bias = nn.Parameter(torch.tensor(math.log(0.84)))

    def forward(self, x, stock, log_ref, pad):
        """x [B,S,F], stock/log_ref/pad [B,S] (pad True = absent stock). Permutation-equivariant over S (no positional encoding)."""
        h = self.inp(torch.cat([x, self.emb(stock)], 2))
        h = self.enc(h, src_key_padding_mask=pad)
        return log_ref + self.head(h).squeeze(2) + self.bias


def pct_loss(logp, y):
    """Mean squared percentage error on exp(logp); the sqrt is taken at reporting time."""
    r = (torch.exp(logp.float().clamp(-12, 0)) - y) / y
    return (r * r).mean()
