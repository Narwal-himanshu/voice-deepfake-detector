"""
Build a leak-controlled MLAAD test set: each real clip is paired with a fake clip
that has a similar noise floor (and, if possible, similar silence ratio).
  python3 make_matched_mlaad.py --csv data/mlaad_all.csv --tol 3
Then:
  python3 evaluate.py --csv data/mlaad_matched.csv --ckpt checkpoints/baseline_rb4.pt --name rb4_mlaad_matched
Note: this is for TESTING only. Never train on it.
"""
import argparse
import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

SR = 16000
MAX_LEN = 4 * SR

def feats(path):
    x, sr = sf.read(path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    assert sr == SR, f"expected 16k, got {sr}"
    x = x[:MAX_LEN]
    n = 400
    fr = x[: len(x) // n * n].reshape(-1, n)
    fe = 10 * np.log10(np.mean(fr ** 2, axis=1) + 1e-10)
    return np.percentile(fe, 10), float(np.mean(fe < fe.max() - 40))

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="data/mlaad_all.csv")
ap.add_argument("--out", default="data/mlaad_matched.csv")
ap.add_argument("--tol", type=float, default=3.0, help="max noise floor gap (dB) in a pair")
ap.add_argument("--sil_tol", type=float, default=0.15, help="max silence-ratio gap in a pair")
a = ap.parse_args()

df = pd.read_csv(a.csv)
nf, sil = [], []
for p in tqdm(df.path, desc="measuring"):
    f, s = feats(p)
    nf.append(f); sil.append(s)
df["noise_floor"], df["silence"] = nf, sil

rng = np.random.RandomState(42)
real = df[df.label == 0].sample(frac=1, random_state=42)
fake = df[df.label == 1].copy()
used, rows = set(), []
for _, r in real.iterrows():
    c = fake[~fake.index.isin(used)]
    c = c[(abs(c.noise_floor - r.noise_floor) <= a.tol) & (abs(c.silence - r.silence) <= a.sil_tol)]
    if len(c) == 0:
        continue
    j = (abs(c.noise_floor - r.noise_floor)).idxmin()
    used.add(j)
    rows.append(r)
    rows.append(fake.loc[j])

out = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
print(f"\nPairs kept: {len(out)//2} of {len(real)} real clips (tol {a.tol} dB)")
if len(out) < 100:
    raise SystemExit("Too few pairs. Increase --tol (e.g. 6) or --sil_tol.")
y = out.label.values
print(f"Check on matched set (want ~0.5 = no leak left):")
print(f"  noise_floor AUC = {roc_auc_score(y, out.noise_floor):.3f}")
print(f"  silence AUC     = {roc_auc_score(y, out.silence):.3f}")
print("Fake systems in matched set:", out[out.label == 1].attack.nunique())
out.drop(columns=["noise_floor", "silence"]).to_csv(a.out, index=False)
print("Saved", a.out)
