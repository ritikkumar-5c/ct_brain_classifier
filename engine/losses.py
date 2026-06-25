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


def build_loss(cfg, class_weight=None):
    if cfg.loss == "focal":
        return FocalLoss(weight=class_weight, gamma=cfg.focal_gamma,
                         label_smoothing=cfg.label_smoothing)
    return nn.CrossEntropyLoss(weight=class_weight,
                               label_smoothing=cfg.label_smoothing)
