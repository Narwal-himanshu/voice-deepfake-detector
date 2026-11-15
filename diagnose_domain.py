"""
Domain diagnosis: ASVspoof vs MLAAD, real vs fake. No torch needed.
  python3 diagnose_domain.py --asv data/train.csv --mlaad data/mlaad_all.csv --n 200
Prints per-group averages of simple audio features. If ASVspoof real and
MLAAD real look very different, the model can learn "recording style".
"""
import argparse
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import welch, resample_poly

SR = 16000

def load(path, max_sec=4):
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    orig_sr = sr
    if sr != SR:
        from math import gcd
        g = gcd(sr, SR)
        x = resample_poly(x, SR // g, sr // g).astype("float32")
    return x[: max_sec * SR], orig_sr

def feats(x, orig_sr):
    eps = 1e-10
    rms_db = 20 * np.log10(np.sqrt(np.mean(x ** 2)) + eps)
    peak_db = 20 * np.log10(np.max(np.abs(x)) + eps)
    # frame energies (25 ms)
    n = 400
    fr = x[: len(x) // n * n].reshape(-1, n)
    fe = 10 * np.log10(np.mean(fr ** 2, axis=1) + eps)
    noise_floor = np.percentile(fe, 10)
    silence_ratio = np.mean(fe < (fe.max() - 40))
    f, p = welch(x, fs=SR, nperseg=512)
    tot = p.sum() + eps
    hf_4_8 = p[f >= 4000].sum() / tot
    hf_7_8 = p[f >= 7000].sum() / tot
    cum = np.cumsum(p) / tot
    rolloff95 = f[np.searchsorted(cum, 0.95)]
    return dict(orig_sr=orig_sr, rms_db=rms_db, peak_db=peak_db, noise_floor_db=noise_floor,
                silence_ratio=silence_ratio, hf_4_8k_ratio=hf_4_8, hf_7_8k_ratio=hf_7_8,
                rolloff95_hz=rolloff95)

def run(csv, tag, n):
    df = pd.read_csv(csv)
    rows = []
    for lab, g in df.groupby("label"):
        g = g.sample(min(n, len(g)), random_state=1)
        for p in g.path:
            try:
                x, sr = load(p)
                r = feats(x, sr)
            except Exception as e:
                print("skip", p, e)
                continue
            r["group"] = f"{tag}_{'real' if lab == 0 else 'fake'}"
            rows.append(r)
    return pd.DataFrame(rows)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--asv", default="data/train.csv")
    ap.add_argument("--mlaad", default="data/mlaad_all.csv")
    ap.add_argument("--n", type=int, default=200, help="files per class per dataset")
    a = ap.parse_args()
    df = pd.concat([run(a.asv, "ASV", a.n), run(a.mlaad, "MLAAD", a.n)])
    pd.set_option("display.width", 200)
    out = df.groupby("group").mean(numeric_only=True).round(3)
    print("\n=== Average features per group ===")
    print(out.T.to_string())
    print("\nHow to read:")
    print("- noise_floor_db: higher = noisier background. ASV is clean studio-like.")
    print("- rolloff95_hz / hf ratios: how much high-frequency content exists.")
    print("- silence_ratio: share of near-silent frames.")
    print("- If ASV_real and MLAAD_real differ a lot, model can learn style, not fakeness.")
    df.to_csv("results/_domain_diag.csv", index=False) if __import__("os").path.isdir("results") else None
