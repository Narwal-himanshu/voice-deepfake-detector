"""
Why did the model fail on a test set?  python analyze.py --csv data/mlaad_all.csv
Part 1: audio facts (sample rate, length) for real vs fake  (no model needed)
Part 2: what the model scores for real vs fake, and for each TTS system
"""
import os, argparse
import numpy as np
import pandas as pd
import soundfile as sf

ap = argparse.ArgumentParser()
ap.add_argument("--csv", required=True)
ap.add_argument("--ckpt", default="checkpoints/baseline.pt")
ap.add_argument("--n", type=int, default=300, help="files per class to check (keeps it fast)")
ap.add_argument("--meta_only", action="store_true")
a = ap.parse_args()

df = pd.read_csv(a.csv)
df = pd.concat([g.sample(min(a.n, len(g)), random_state=1) for _, g in df.groupby("label")]).reset_index(drop=True)

# ---------- Part 1: audio facts ----------
sr, dur, ch = [], [], []
for p in df.path:
    i = sf.info(p)
    sr.append(i.samplerate); dur.append(i.duration); ch.append(i.channels)
df["sr"], df["dur"], df["ch"] = sr, dur, ch
print("\n=== PART 1: audio facts (label 0 = real, 1 = fake) ===")
for lab, g in df.groupby("label"):
    print(f"label {lab}: n={len(g)} | sample rates={sorted(g.sr.unique().tolist())} | "
          f"duration mean={g.dur.mean():.1f}s median={g.dur.median():.1f}s | "
          f"shorter than 4s: {(g.dur < 4).mean()*100:.0f}% | channels={sorted(g.ch.unique().tolist())}")
if a.meta_only:
    raise SystemExit

# ---------- Part 2: model scores ----------
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import torch
from torch.utils.data import DataLoader
from dataset import AudioDataset
from model import LogMel, BaselineCNN, get_device, predict

os.makedirs("results", exist_ok=True)
df.to_csv("results/_diag.csv", index=False)
device = get_device()
model = BaselineCNN().to(device)
model.load_state_dict(torch.load(a.ckpt, map_location=device))
labels, scores = predict(model, LogMel(), DataLoader(AudioDataset("results/_diag.csv"), batch_size=32), device)
df["score"] = scores
print("\n=== PART 2: model scores (score near 1 = model says FAKE) ===")
for lab, g in df.groupby("label"):
    print(f"label {lab} ({'real' if lab == 0 else 'fake'}): mean score={g.score.mean():.3f} | "
          f"predicted FAKE: {(g.score >= 0.5).mean()*100:.0f}%")
if "attack" in df.columns:
    print("\nPer source (real / each TTS system): mean score | % predicted fake")
    t = df.groupby("attack").score.agg(["mean", lambda s: (s >= 0.5).mean() * 100, "count"])
    t.columns = ["mean_score", "pct_pred_fake", "n"]
    print(t.sort_values("mean_score").round(2).to_string())