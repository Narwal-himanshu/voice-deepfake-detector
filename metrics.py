"""EER, AUC, F1. label 1 = fake (positive class), score = probability of fake."""
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, f1_score

def compute_metrics(labels, scores, threshold=0.5):
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=float)
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    eer = (fpr[i] + fnr[i]) / 2
    return {
        "EER": float(eer * 100),                     # in %, lower is better
        "AUC": float(roc_auc_score(labels, scores)), # closer to 1 is better
        "F1": float(f1_score(labels, (scores >= threshold).astype(int))),
    }
