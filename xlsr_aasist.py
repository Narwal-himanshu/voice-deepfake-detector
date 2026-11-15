"""XLS-R 300M front-end + AASIST-style graph-attention back-end (simplified re-implementation)."""
import zlib
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

SR = 16000
MAX_LEN = 4 * SR


# ---------------- data ----------------
def fix_length(x, n=MAX_LEN):
    if len(x) >= n:
        return x[:n]
    return np.tile(x, int(np.ceil(n / len(x))))[:n]


class AudioDataset(Dataset):
    """algo>0 -> RawBoost (use ONLY for training data). prob = fraction of clips augmented.
    style = (floor_lo_db, floor_hi_db, gain_db) -> random loudness + random noise floor (hides recording-style shortcuts).
    deterministic=True -> same style noise for the same file every time (use for dev/test sets)."""
    def __init__(self, csv_path, algo=0, prob=0.5, style=None, deterministic=False):
        self.df = pd.read_csv(csv_path)
        self.algo, self.prob, self.style, self.det = algo, prob, style, deterministic
        if algo > 0:
            from rawboost_aug import rawboost
            self._rb = rawboost
        if style is not None:
            from floor_eq import randomize_style
            self._style = randomize_style

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        wav, sr = sf.read(row["path"], dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        assert sr == SR, f"{row['path']}: sample rate {sr}"
        wav = fix_length(wav)
        if self.style is not None:
            rng = np.random.RandomState(zlib.crc32(row["path"].encode()) % (2 ** 32)) if self.det else np.random
            wav = self._style(wav, rng, *self.style).astype(np.float32)
        if self.algo > 0 and np.random.rand() < self.prob:
            wav = self._rb(wav, self.algo)
        return torch.from_numpy(np.ascontiguousarray(wav)), torch.tensor(row["label"], dtype=torch.float32)


def worker_init(_):
    # every DataLoader worker needs its own numpy seed, otherwise RawBoost noise repeats
    np.random.seed(torch.initial_seed() % (2 ** 32))


# ---------------- back-end ----------------
class ResBlock(nn.Module):
    def __init__(self, cin, cout, pool_t=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(cin)
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.short = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()
        self.pool = nn.MaxPool2d((1, pool_t)) if pool_t > 1 else nn.Identity()

    def forward(self, x):
        y = self.c1(F.selu(self.bn1(x)))
        y = self.c2(F.selu(self.bn2(y)))
        return self.pool(y + self.short(x))


class GraphAttn(nn.Module):
    """Attention over a fully connected graph. nodes (B, N, din) -> (B, N, dout)."""
    def __init__(self, din, dout):
        super().__init__()
        self.q, self.k, self.v = nn.Linear(din, dout), nn.Linear(din, dout), nn.Linear(din, dout)
        self.res = nn.Linear(din, dout)
        self.bn = nn.BatchNorm1d(dout)
        self.scale = dout ** -0.5

    def forward(self, h):
        att = torch.softmax(self.q(h) @ self.k(h).transpose(1, 2) * self.scale, dim=-1)
        out = att @ self.v(h) + self.res(h)
        return F.selu(self.bn(out.transpose(1, 2)).transpose(1, 2))


class Branch(nn.Module):
    """Graph attention on one axis (spectral or temporal) + a learnable master node."""
    def __init__(self, d):
        super().__init__()
        self.g1, self.g2 = GraphAttn(d, d), GraphAttn(d, d)
        self.master = nn.Parameter(torch.randn(1, 1, d) * 0.1)

    def forward(self, nodes):
        h = self.g1(nodes)
        hm = self.g2(torch.cat([self.master.expand(h.size(0), -1, -1), h], dim=1))
        return hm[:, 0], hm[:, 1:]


class XLSR_AASIST(nn.Module):
    def __init__(self, ssl_name="facebook/wav2vec2-xls-r-300m", ssl=None, d=64):
        super().__init__()
        if ssl is None:
            from transformers import Wav2Vec2Model
            ssl = Wav2Vec2Model.from_pretrained(ssl_name, layerdrop=0.0, apply_spec_augment=False)
        self.ssl = ssl
        self.ssl.freeze_feature_encoder()            # keep the small CNN part frozen
        self.proj = nn.Linear(self.ssl.config.hidden_size, 128)
        self.bn0 = nn.BatchNorm2d(1)
        self.enc = nn.Sequential(ResBlock(1, 32, 2), ResBlock(32, 32, 2), ResBlock(32, d), ResBlock(d, d))
        self.spec, self.temp = Branch(d), Branch(d)
        self.drop = nn.Dropout(0.5)
        self.out = nn.Linear(6 * d, 1)

    def forward(self, wav):                          # wav (B, 64000) -> logits (B,)
        wav = wav.float()
        wav = (wav - wav.mean(1, keepdim=True)) / torch.sqrt(wav.var(1, keepdim=True) + 1e-7)
        h = self.ssl(wav).last_hidden_state          # (B, T, 1024)
        x = self.proj(h).transpose(1, 2).unsqueeze(1)  # (B, 1, 128, T)
        x = F.selu(self.bn0(F.max_pool2d(x, (3, 3))))
        x = self.enc(x)                              # (B, d, freq, time)
        s = x.abs().amax(dim=3).transpose(1, 2)      # spectral nodes (B, freq, d)
        t = x.abs().amax(dim=2).transpose(1, 2)      # temporal nodes (B, time, d)
        ms, ns = self.spec(s)
        mt, nt = self.temp(t)
        z = torch.cat([ms, ns.amax(1), ns.mean(1), mt, nt.amax(1), nt.mean(1)], dim=1)
        return self.out(self.drop(z)).squeeze(1)


@torch.no_grad()
def predict(model, loader, device, amp=True):
    model.eval()
    labels, scores = [], []
    for wav, y in loader:
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp and device.type == "cuda"):
            logits = model(wav.to(device))
        scores += torch.sigmoid(logits.float()).cpu().tolist()
        labels += y.tolist()
    return labels, scores
