"""
Train the baseline.   python train.py --epochs 10
Saves the best model (lowest dev EER) to checkpoints/baseline.pt
"""
import os, argparse
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import AudioDataset
from model import LogMel, BaselineCNN, get_device, predict
from metrics import compute_metrics

ap = argparse.ArgumentParser()
ap.add_argument("--epochs", type=int, default=10)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--data", default="data")
ap.add_argument("--algo", type=int, default=0, help="RawBoost algo 0-8 (0 = off)")
ap.add_argument("--prob", type=float, default=0.5, help="fraction of train clips augmented")
ap.add_argument("--name", default="baseline", help="checkpoint name -> checkpoints/<name>.pt")
a = ap.parse_args()

device = get_device()
print("Device:", device, "| RawBoost algo:", a.algo, "| prob:", a.prob, "| name:", a.name)
train_dl = DataLoader(AudioDataset(f"{a.data}/train.csv", algo=a.algo, prob=a.prob), batch_size=a.batch, shuffle=True, num_workers=0)
dev_dl = DataLoader(AudioDataset(f"{a.data}/dev.csv"), batch_size=a.batch, num_workers=0)

feat = LogMel()                      # stays on CPU
model = BaselineCNN().to(device)
opt = torch.optim.Adam(model.parameters(), lr=a.lr)
loss_fn = nn.BCEWithLogitsLoss()     # BCE loss (real=0, fake=1)

os.makedirs("checkpoints", exist_ok=True)
best = 100.0
for ep in range(1, a.epochs + 1):
    model.train()
    total = 0.0
    for wav, y in tqdm(train_dl, desc=f"Epoch {ep}/{a.epochs}"):
        x, y = feat(wav).to(device), y.to(device)
        loss = loss_fn(model(x), y)
        opt.zero_grad(); loss.backward(); opt.step()
        total += loss.item()
    labels, scores = predict(model, feat, dev_dl, device)
    m = compute_metrics(labels, scores)
    print(f"Epoch {ep}: train_loss={total/len(train_dl):.4f} | dev EER={m['EER']:.2f}% AUC={m['AUC']:.3f} F1={m['F1']:.3f}")
    if m["EER"] < best:
        best = m["EER"]
        torch.save(model.state_dict(), f"checkpoints/{a.name}.pt")
        print(f"  saved best model (dev EER {best:.2f}%)")
print("Done. Best dev EER:", f"{best:.2f}%")
