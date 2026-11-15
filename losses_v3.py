"""Losses for v3 training.

The literature (e.g. Scalable-AASIST 2025, Beyond Silence 2024) finds that once a
model has learnt the easy cases it should focus on hard examples. Focal loss does
that. We also support a BCE -> focal hybrid schedule.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, reduction="mean"):
        super().__init__()
        self.gamma, self.alpha, self.reduction = gamma, alpha, reduction

    def forward(self, logits, targets):
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = bce * (1 - p_t).pow(self.gamma)
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = loss * alpha_t
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class HybridLoss(nn.Module):
    """loss = (1 - focal_weight) * BCE + focal_weight * focal."""

    def __init__(self, focal_weight=0.5, gamma=2.0, alpha=0.25):
        super().__init__()
        self.w = float(focal_weight)
        self.bce = nn.BCEWithLogitsLoss()
        self.focal = BinaryFocalLoss(gamma=gamma, alpha=alpha)

    def forward(self, logits, targets):
        return (1 - self.w) * self.bce(logits, targets) + self.w * self.focal(logits, targets)


def build_loss(name="focal", gamma=2.0, alpha=0.25, focal_weight=0.5):
    name = (name or "focal").lower()
    if name == "bce":
        return nn.BCEWithLogitsLoss()
    if name == "focal":
        return BinaryFocalLoss(gamma=gamma, alpha=alpha)
    if name == "hybrid":
        return HybridLoss(focal_weight=focal_weight, gamma=gamma, alpha=alpha)
    raise ValueError(f"unknown loss: {name}")
