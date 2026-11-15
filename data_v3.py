"""V3 data pipeline: multi-dataset loading, system-disjoint MLAAD splits,
balanced sampling and shortcut-safe training augmentation.

Key differences from v1/v2:
- a clip is described by (path, label, attack, source) and several datasets can
  be mixed in one CSV
- MLAAD is split by TTS system so the dev/test systems are never seen in training
- a WeightedRandomSampler gives every (source, label) cell the same mass, so
  ASVspoof does not get drowned by MLAAD or vice-versa
- training uses random crops + augment_v3; dev/test are deterministic
"""
import argparse
import os
import re
import zlib
from collections import Counter

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from scipy.signal import resample_poly
from torch.utils.data import Dataset, WeightedRandomSampler

from augment_v3 import augment_train, center_crop, random_crop

SR = 16000
MAX_LEN = 4 * SR


# ------------------------------------------------------------------ paths
def resolve_path(p, roots=()):
    p = str(p)
    if os.path.isabs(p) and os.path.exists(p):
        return p
    for root in roots:
        cand = os.path.join(root, p)
        if os.path.exists(cand):
            return cand
    return p


def _read_csv(csv_path, source, roots=()):
    df = pd.read_csv(csv_path)
    if "label" not in df.columns or "path" not in df.columns:
        raise ValueError(f"{csv_path}: needs 'path' and 'label' columns")
    df = df.copy()
    df["path"] = [resolve_path(p, roots) for p in df["path"]]
    df["source"] = source
    if "attack" not in df.columns:
        df["attack"] = "-"
    if "lang" not in df.columns:
        df["lang"] = "en"
    return df[["path", "utt", "attack", "label", "lang", "source"]] if "utt" in df.columns \
        else df[["path", "attack", "label", "lang", "source"]]


def _source_of(path):
    p = str(path).lower()
    if "mlaad" in p or "/fake/" in p or "/original/" in p:
        return "mlaad"
    if "la_" in p or "asvspoof" in p or "/la/" in p:
        return "asv"
    return "other"


# ------------------------------------------------------------------ split builder
def build_v3_splits(out_dir="data_v3",
                    asv_train="colab_pack/data/train.csv",
                    asv_dev="colab_pack/data/dev.csv",
                    asv_eval="colab_pack/data/eval.csv",
                    mlaad_train="colab_pack_v2/data/mlaad_train2.csv",
                    mlaad_test="colab_pack_v2/data/mlaad_test_unseen.csv",
                    mlaad_all="colab_pack/data/mlaad_all.csv",
                    mlaad_matched="colab_pack/data/mlaad_matched.csv",
                    dev_system_share=0.12, seed=42, roots=()):
    """Create v3 train/dev/test CSVs. Returns dict name -> csv path."""
    os.makedirs(out_dir, exist_ok=True)
    roots = tuple(roots) if roots else (".",)

    def rd(p, src):
        if p and os.path.exists(p):
            return _read_csv(p, src, roots)
        print(f"  [warn] missing {p}, skipping")
        return pd.DataFrame(columns=["path", "utt", "attack", "label", "lang", "source"])

    asv_tr = rd(asv_train, "asv")
    asv_dv = rd(asv_dev, "asv")
    asv_te = rd(asv_eval, "asv")
    ml_tr_full = rd(mlaad_train, "mlaad")
    ml_te = rd(mlaad_test, "mlaad")
    ml_all = rd(mlaad_all, "mlaad")
    ml_match = rd(mlaad_matched, "mlaad")

    # split MLAAD training systems into train / dev by TTS system
    ml_dev = ml_tr_full.iloc[0:0].copy()
    ml_tr = ml_tr_full
    if len(ml_tr_full):
        fake = ml_tr_full[ml_tr_full.label == 1]
        real = ml_tr_full[ml_tr_full.label == 0]
        systems = sorted(fake.attack.unique().tolist())
        rng = np.random.RandomState(seed)
        rng.shuffle(systems)
        n_dev_sys = max(1, int(round(dev_system_share * len(systems))))
        dev_systems = set(systems[:n_dev_sys])
        fake_dev = fake[fake.attack.isin(dev_systems)]
        fake_tr = fake[~fake.attack.isin(dev_systems)]
        n_real_dev = min(len(fake_dev), len(real))
        real_dev = real.sample(n_real_dev, random_state=seed)
        real_tr = real.drop(real_dev.index)
        ml_dev = pd.concat([real_dev, fake_dev]).sample(frac=1, random_state=seed)
        ml_tr = pd.concat([real_tr, fake_tr]).sample(frac=1, random_state=seed)
        print(f"  MLAAD dev systems ({n_dev_sys}): {', '.join(sorted(dev_systems))}")

    # never let a test system leak into train/dev
    if len(ml_te):
        test_systems = set(ml_te[ml_te.label == 1].attack.unique().tolist())
        ml_tr = ml_tr[~ml_tr.attack.isin(test_systems)]
        ml_dev = ml_dev[~ml_dev.attack.isin(test_systems)]

    train = pd.concat([asv_tr, ml_tr]).sample(frac=1, random_state=seed).reset_index(drop=True)
    dev = pd.concat([asv_dv, ml_dev]).sample(frac=1, random_state=seed).reset_index(drop=True)

    out = {
        "train": os.path.join(out_dir, "v3_train.csv"),
        "dev": os.path.join(out_dir, "v3_dev.csv"),
        "dev_asv": os.path.join(out_dir, "v3_dev_asv.csv"),
        "dev_mlaad": os.path.join(out_dir, "v3_dev_mlaad.csv"),
        "test_asv": os.path.join(out_dir, "v3_test_asv.csv"),
        "test_mlaad_all": os.path.join(out_dir, "v3_test_mlaad_all.csv"),
        "test_mlaad_matched": os.path.join(out_dir, "v3_test_mlaad_matched.csv"),
        "test_mlaad_unseen": os.path.join(out_dir, "v3_test_mlaad_unseen.csv"),
    }
    train.to_csv(out["train"], index=False)
    dev.to_csv(out["dev"], index=False)
    asv_dv.to_csv(out["dev_asv"], index=False)
    ml_dev.to_csv(out["dev_mlaad"], index=False)
    if len(asv_te):
        asv_te.to_csv(out["test_asv"], index=False)
    if len(ml_all):
        ml_all.to_csv(out["test_mlaad_all"], index=False)
    if len(ml_match):
        ml_match.to_csv(out["test_mlaad_matched"], index=False)
    if len(ml_te):
        ml_te.to_csv(out["test_mlaad_unseen"], index=False)

    print("\n=== v3 splits ===")
    for name, df in [("train", train), ("dev", dev)]:
        c = df.groupby(["source", "label"]).size().unstack(fill_value=0)
        print(f"{name}: {len(df)} clips")
        print(c.to_string())
    print("test_asv:", len(asv_te), "| test_mlaad_all:", len(ml_all),
          "| test_mlaad_matched:", len(ml_match), "| test_mlaad_unseen:", len(ml_te))
    return out


# ------------------------------------------------------------------ dataset
class AudioDatasetV3(Dataset):
    def __init__(self, csv_path, train=False, n=MAX_LEN, aug=None, seed=0,
                 sources=None, epoch=0):
        self.df = pd.read_csv(csv_path)
        if sources is not None:
            self.df = self.df[self.df["source"].isin(sources)].reset_index(drop=True)
        if "source" not in self.df.columns:
            self.df["source"] = [_source_of(p) for p in self.df["path"]]
        self.train, self.n, self.aug, self.seed, self.epoch = train, n, aug, seed, epoch
        self.src_to_idx = {s: i for i, s in enumerate(sorted(self.df["source"].unique()))}

    def __len__(self):
        return len(self.df)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def _rng(self, i):
        info = torch.utils.data.get_worker_info()
        wid = 0 if info is None else info.id
        key = (self.seed * 1_000_003 + self.epoch * 7919 + wid * 104729 + int(i)) % (2 ** 63)
        return np.random.default_rng(key)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        wav, sr = sf.read(row["path"], dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        sr = int(sr)
        if sr != SR:
            g = np.gcd(sr, SR)
            wav = resample_poly(wav, SR // g, sr // g).astype(np.float32)
        wav = np.nan_to_num(wav, nan=0.0, posinf=0.0, neginf=0.0)

        rng = self._rng(i)
        if self.train:
            wav = random_crop(wav, rng, self.n)
            wav = augment_train(wav, rng, self.aug)
        else:
            wav = center_crop(wav, self.n)
            wav = np.nan_to_num(wav, nan=0.0, posinf=0.0, neginf=0.0)

        return (torch.from_numpy(np.ascontiguousarray(wav, dtype=np.float32)),
                torch.tensor(float(row["label"]), dtype=torch.float32),
                torch.tensor(self.src_to_idx[row["source"]], dtype=torch.long))


def worker_init_fn(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))


def make_weighted_sampler(df, class_balance=True, source_weights=None):
    """Each (source, label) cell gets equal mass; optional per-source multipliers."""
    if "source" not in df.columns:
        df = df.copy()
        df["source"] = [_source_of(p) for p in df["path"]]
    keys = list(zip(df["source"], df["label"])) if class_balance else list(df["source"])
    counts = Counter(keys)
    w = []
    for key, src in zip(keys, df["source"]):
        v = 1.0 / counts[key]
        if source_weights:
            v *= float(source_weights.get(src, 1.0))
        w.append(v)
    w = np.asarray(w, dtype=np.float64)
    w /= w.mean()
    return WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double),
                                 num_samples=len(w), replacement=True)


def describe(csv_path):
    df = pd.read_csv(csv_path)
    if "source" not in df.columns:
        df["source"] = [_source_of(p) for p in df["path"]]
    print(f"{csv_path}: {len(df)} clips")
    print(df.groupby(["source", "label"]).size().unstack(fill_value=0).to_string())
    if "attack" in df.columns:
        print("fake systems:", df[df.label == 1].attack.nunique())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data_v3")
    ap.add_argument("--asv-train", default="colab_pack/data/train.csv")
    ap.add_argument("--asv-dev", default="colab_pack/data/dev.csv")
    ap.add_argument("--asv-eval", default="colab_pack/data/eval.csv")
    ap.add_argument("--mlaad-train", default="colab_pack_v2/data/mlaad_train2.csv")
    ap.add_argument("--mlaad-test", default="colab_pack_v2/data/mlaad_test_unseen.csv")
    ap.add_argument("--mlaad-all", default="colab_pack/data/mlaad_all.csv")
    ap.add_argument("--mlaad-matched", default="colab_pack/data/mlaad_matched.csv")
    ap.add_argument("--dev-system-share", type=float, default=0.12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--root", action="append", default=[], help="extra root for relative paths")
    a = ap.parse_args()
    build_v3_splits(out_dir=a.out, asv_train=a.asv_train, asv_dev=a.asv_dev, asv_eval=a.asv_eval,
                    mlaad_train=a.mlaad_train, mlaad_test=a.mlaad_test, mlaad_all=a.mlaad_all,
                    mlaad_matched=a.mlaad_matched, dev_system_share=a.dev_system_share,
                    seed=a.seed, roots=a.root or (".",))
