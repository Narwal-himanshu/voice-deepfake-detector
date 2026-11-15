"""Score calibration + robust metrics for v3.

v1 picked LOW=1.2e-4 from LA-eval, which made every real-world clip "uncertain".
Here we fit a temperature on a held-out calibration set (mixed ASVspoof + MLAAD)
and derive two thresholds:
  score < real_thr   -> real
  score >= fake_thr  -> fake
  otherwise          -> uncertain
so the demo's confidence band actually means something.
"""
import json

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (balanced_accuracy_score, f1_score, roc_auc_score,
                             roc_curve)

from metrics import compute_metrics


def sigmoid(x, temperature=1.0):
    x = np.asarray(x, dtype=np.float64) / max(float(temperature), 1e-6)
    return 1.0 / (1.0 + np.exp(-x))


def fit_temperature(logits, labels, iterations=100):
    z = torch.tensor(np.asarray(logits, dtype=np.float64))
    y = torch.tensor(np.asarray(labels, dtype=np.float64))
    log_t = torch.zeros((), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=int(iterations),
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        t = torch.exp(log_t).clamp(0.05, 20.0)
        loss = F.binary_cross_entropy_with_logits(z / t, y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(torch.exp(log_t).clamp(0.05, 20.0).item())


def threshold_at_eer(scores, labels):
    fpr, tpr, thr = roc_curve(np.asarray(labels).astype(int), np.asarray(scores))
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    return float(thr[min(i, len(thr) - 1)])


def threshold_at_fpr(scores, labels, max_fpr=0.01):
    fpr, tpr, thr = roc_curve(np.asarray(labels).astype(int), np.asarray(scores))
    ok = np.where(fpr <= max_fpr)[0]
    if len(ok) == 0:
        return float(np.max(scores) + 1e-6)
    return float(thr[ok[-1]])


def summarize(labels, logits, temperature=1.0, threshold=None, fpr_target=0.01,
              real_thr=None):
    """EER/AUC are threshold-free. F1/balanced-acc use threshold (or EER threshold).

    Pass both threshold and real_thr from a separate calibration set to evaluate
    without tuning on the test labels.
    """
    labels = np.asarray(labels).astype(int)
    scores = sigmoid(logits, temperature)
    m = compute_metrics(labels, scores)
    thr_fake = threshold_at_eer(scores, labels) if threshold is None else float(threshold)
    thr_real = threshold_at_fpr(scores, labels, fpr_target) if real_thr is None else float(real_thr)
    pred = (scores >= thr_fake).astype(int)
    real = labels == 0
    fake = labels == 1
    return {
        **m,
        "EER_thr": thr_fake,
        "real_thr": thr_real,
        "fake_thr": thr_fake,
        "balanced_acc": float(balanced_accuracy_score(labels, pred)),
        "F1_at_thr": float(f1_score(labels, pred)),
        "real_acc": float((pred[real] == 0).mean()) if real.any() else None,
        "fake_acc": float((pred[fake] == 1).mean()) if fake.any() else None,
        "uncertain_rate": float(((scores >= thr_real) & (scores < thr_fake)).mean()),
        "temperature": float(temperature),
        "n": int(len(labels)),
    }


def fit_calibration(logits, labels, fpr_target=0.01):
    t = fit_temperature(logits, labels)
    scores = sigmoid(logits, t)
    return {
        "temperature": t,
        "fake_thr": threshold_at_eer(scores, labels),
        "real_thr": threshold_at_fpr(scores, labels, fpr_target),
        "fpr_target": fpr_target,
    }


def save_calibration(path, cal):
    with open(path, "w") as f:
        json.dump(cal, f, indent=2)


def load_calibration(path):
    with open(path) as f:
        return json.load(f)
