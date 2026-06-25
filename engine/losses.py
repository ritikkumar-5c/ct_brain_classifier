"""
engine/losses.py
Loss functions for class imbalance (paper used a weighted class function).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.weight = weight
        self.gamma = gamma
        self.ls = label_smoothing

    def forward(self, logits, target):
        ce = F.cross_entropy(
            logits, target, weight=self.weight,
            reduction="none", label_smoothing=self.ls,
        )
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


class CostSensitiveLoss(nn.Module):
    """Cost-sensitive loss for asymmetric clinical error costs.

    Minimizes the expected misclassification cost under the predicted distribution,
    E_j[ p_j * C[true, j] ], where C is a cost matrix (0 on the diagonal). Under-
    calling pathology to 'normal' is made expensive via C, so the model is pushed
    to not miss near_normal / abnormal. A small CE term (ce_lambda) is blended in
    for gradient stability / calibration.
    """
    def __init__(self, cost_matrix, weight=None, ce_lambda=0.3, label_smoothing=0.0):
        super().__init__()
        self.register_buffer("cost", cost_matrix.float())
        self.weight = weight
        self.ce_lambda = ce_lambda
        self.ls = label_smoothing

    def forward(self, logits, target):
        p = torch.softmax(logits, dim=1)
        cost_rows = self.cost.to(logits.device)[target]          # (B, C)
        expected_cost = (p * cost_rows).sum(dim=1).mean()
        loss = expected_cost
        if self.ce_lambda > 0:
            loss = loss + self.ce_lambda * F.cross_entropy(
                logits, target, weight=self.weight, label_smoothing=self.ls)
        return loss


def build_cost_matrix(cfg):
    """Cost matrix C[true, pred]: 0 on diagonal, 1 for generic errors, and a
    higher cost for under-calling pathology to 'normal' (index 0)."""
    n = cfg.num_classes
    C = torch.ones(n, n) - torch.eye(n)
    if n >= 3:                       # 0=normal, 1=near_normal, 2=abnormal
        C[2, 0] = float(cfg.cost_miss_abnormal)      # abnormal -> normal (worst miss)
        C[1, 0] = float(cfg.cost_miss_near_normal)   # near_normal -> normal
    elif n == 2:
        C[1, 0] = float(cfg.cost_miss_abnormal)      # abnormal -> normal
    return C


def build_loss(cfg, class_weight=None):
    if cfg.loss == "focal":
        return FocalLoss(weight=class_weight, gamma=cfg.focal_gamma,
                         label_smoothing=cfg.label_smoothing)
    if cfg.loss == "cost_sensitive":
        return CostSensitiveLoss(build_cost_matrix(cfg), weight=class_weight,
                                 ce_lambda=getattr(cfg, "cost_ce_lambda", 0.3),
                                 label_smoothing=cfg.label_smoothing)
    return nn.CrossEntropyLoss(weight=class_weight,
                               label_smoothing=cfg.label_smoothing)
