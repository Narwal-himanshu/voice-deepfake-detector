"""
Test a trained model on any CSV.
  python evaluate.py --csv data/eval.csv --name baseline_in_domain
Results are also appended to results/results.csv (used later for the bar chart).
"""
import os, argparse
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import torch
import pandas as pd
from torch.utils.data import DataLoader
from dataset import AudioDataset
from model import LogMel, BaselineCNN, get_device, predict
from metrics import compute_metrics

ap = argparse.ArgumentParser()
ap.add_argument("--csv", required=True)
ap.add_argument("--ckpt", default="checkpoints/baseline.pt")
ap.add_argument("--name", required=True)
ap.add_argument("--batch", type=int, default=32)
a = ap.parse_args()

device = get_device()
model = BaselineCNN().to(device)
model.load_state_dict(torch.load(a.ckpt, map_location=device))
labels, scores = predict(model, LogMel(), DataLoader(AudioDataset(a.csv), batch_size=a.batch), device)
m = compute_metrics(labels, scores)
print(f"[{a.name}] EER={m['EER']:.2f}%  AUC={m['AUC']:.3f}  F1={m['F1']:.3f}")

os.makedirs("results", exist_ok=True)
row = pd.DataFrame([{"name": a.name, **m}])
row.to_csv("results/results.csv", mode="a", header=not os.path.exists("results/results.csv"), index=False)
