"""Evaluate a v3 checkpoint on in-domain and cross-domain test sets.

Example:
  python evaluate_v3.py --ckpt runs_v3/xlsr_v3.pt \
      --calibrate-csv data_v3/v3_dev.csv \
      --csv asv_eval=data_v3/v3_test_asv.csv \
      --csv mlaad_unseen=data_v3/v3_test_mlaad_unseen.csv \
      --csv mlaad_all=data_v3/v3_test_mlaad_all.csv \
      --shortcut-test data_v3/v3_test_mlaad_unseen.csv
"""
import argparse
import os

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from scipy.signal import resample_poly
from torch.utils.data import DataLoader

from calibration_v3 import fit_calibration, save_calibration, sigmoid, summarize
from data_v3 import AudioDatasetV3, worker_init_fn
from floor_eq import feats
from model_v3 import load_model_v3, predict_logits
from train_v3 import get_device

SR = 16000


def make_loader(csv, args):
    ds = AudioDatasetV3(csv, train=False, seed=args.seed)
    return DataLoader(ds, batch_size=args.batch, shuffle=False,
                      num_workers=args.workers, worker_init_fn=worker_init_fn), ds


def save_scores(csv, labels, logits, out_path, temperature=1.0, fake_thr=0.5,
                real_thr=0.0):
    df = pd.read_csv(csv)
    n = min(len(df), len(logits))
    df = df.iloc[:n].copy()
    df["logit"] = logits[:n]
    df["score"] = sigmoid(logits[:n], temperature)
    df["pred"] = np.where(df.score >= fake_thr, 1,
                          np.where(df.score < real_thr, 0, -1))
    df["label"] = np.asarray(labels[:n]).astype(int)
    df.to_csv(out_path, index=False)
    return df


def per_attack(df, fake_thr):
    rows = []
    for atk, g in df.groupby("attack"):
        rows.append({"attack": atk, "n": len(g), "mean_score": float(g.score.mean()),
                     "pct_pred_fake": float((g.score >= fake_thr).mean() * 100)})
    return pd.DataFrame(rows).sort_values("mean_score").set_index("attack")


# ---------------------------------------------------------------- shortcut
def _load4(path):
    x, sr = sf.read(path, dtype="float32", frames=4 * SR)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if int(sr) != SR:
        g = np.gcd(int(sr), SR)
        x = resample_poly(x, SR // g, int(sr) // g).astype(np.float32)
    return x


def shortcut_frame(csv, n_per_class=400, seed=1):
    df = pd.read_csv(csv)
    parts = []
    for lab, g in df.groupby("label"):
        parts.append(g.sample(min(n_per_class, len(g)), random_state=seed))
    df = pd.concat(parts).reset_index(drop=True)
    rows = []
    for p, y in zip(df.path, df.label):
        try:
            rows.append({**feats(_load4(p)), "label": int(y)})
        except Exception as e:
            print("  shortcut skip", p, e)
    return pd.DataFrame(rows)


def shortcut_report(train_csv=None, test_csv=None, n_per_class=400):
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    if not test_csv:
        return
    te = shortcut_frame(test_csv, n_per_class)
    if len(te) < 10 or te.label.nunique() < 2:
        print("shortcut: not enough data")
        return
    feats_cols = [c for c in te.columns if c != "label"]
    print("\n=== shortcut control (higher = more separable by recording style) ===")
    for c in feats_cols:
        auc = roc_auc_score(te.label, te[c])
        print(f"  single feature {c:14s} AUC = {max(auc, 1 - auc):.3f}")
    if train_csv and os.path.exists(train_csv):
        tr = shortcut_frame(train_csv, n_per_class)
        clf = GradientBoostingClassifier(random_state=0).fit(tr[feats_cols], tr.label)
        auc = roc_auc_score(te.label, clf.predict_proba(te[feats_cols])[:, 1])
        print(f"  tiny gradient-boosting classifier AUC = {max(auc, 1 - auc):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--csv", action="append", default=[], help="name=path, repeatable")
    ap.add_argument("--calibrate-csv", default=None)
    ap.add_argument("--out", default="results_v3")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--shortcut-train", default=None)
    ap.add_argument("--shortcut-test", default=None)
    ap.add_argument("--shortcut-n", type=int, default=400)
    ap.add_argument("--shortcut-only", action="store_true",
                    help="run the shortcut control and exit (no checkpoint needed)")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.shortcut_only:
        shortcut_report(args.shortcut_train, args.shortcut_test, args.shortcut_n)
        return
    if not args.ckpt:
        ap.error("--ckpt is required unless --shortcut-only is used")

    device = get_device()
    model = load_model_v3(args.ckpt, device=device)
    model.eval()
    name = args.name or os.path.splitext(os.path.basename(args.ckpt))[0]
    print(f"loaded {args.ckpt} on {device}")

    cal = None
    if args.calibrate_csv:
        loader, _ = make_loader(args.calibrate_csv, args)
        labels, logits = predict_logits(model, loader, device)
        cal = fit_calibration(logits, labels)
        save_calibration(os.path.join(args.out, f"calibration_{name}.json"), cal)
        print(f"calibration: T={cal['temperature']:.3f} real<{cal['real_thr']:.4f} "
              f"fake>={cal['fake_thr']:.4f}")

    rows = []
    specs = [s if "=" in s else f"set{i}={s}" for i, s in enumerate(args.csv)]
    for spec in specs:
        set_name, csv = spec.split("=", 1)
        if not os.path.exists(csv):
            print("skip missing", csv)
            continue
        loader, _ = make_loader(csv, args)
        labels, logits = predict_logits(model, loader, device)
        if cal:
            m = summarize(labels, logits, temperature=cal["temperature"],
                          threshold=cal["fake_thr"], real_thr=cal["real_thr"])
        else:
            m = summarize(labels, logits)
        df = save_scores(csv, labels, logits,
                         os.path.join(args.out, f"scores_{name}_{set_name}.csv"),
                         temperature=cal["temperature"] if cal else 1.0,
                         fake_thr=m["fake_thr"], real_thr=m["real_thr"])
        print(f"[{name} | {set_name:16s}] n={m['n']:5d} EER={m['EER']:6.2f}% "
              f"AUC={m['AUC']:.3f} bacc={m['balanced_acc']:.3f} "
              f"real={m['real_acc']} fake={m['fake_acc']} unc={m['uncertain_rate']:.3f}")
        rows.append({"name": name, "set": set_name, **m})

        if "attack" in df.columns and (df.label == 1).any():
            print(per_attack(df[df.label == 1], m["fake_thr"]).round(3).to_string())

    if rows:
        out = os.path.join(args.out, f"eval_{name}.csv")
        pd.DataFrame(rows).to_csv(out, index=False)
        print("saved ->", out)

    if args.shortcut_test:
        shortcut_report(args.shortcut_train, args.shortcut_test, args.shortcut_n)


if __name__ == "__main__":
    main()
