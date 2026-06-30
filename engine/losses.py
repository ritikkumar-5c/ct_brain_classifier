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


class RuleOutLoss(nn.Module):
    """Auxiliary loss on the binary rule-out head (not-normal=1 vs normal=0).

    Combines two terms:
      * BCE-with-logits (optional pos_weight) — calibrates the bag not-normal score.
      * a PARTIAL-AUC pairwise surrogate that concentrates gradient on the high-
        sensitivity / low-false-clear operating region used for auto-report:
        within the batch, take the HARDEST positives (the lowest-scoring not-normal
        cases — these set where a high-sensitivity threshold must sit) and the
        HARDEST negatives (the highest-scoring normals — the would-be false clears),
        and apply a squared-hinge ranking penalty max(0, margin-(s_pos-s_neg))^2 over
        those pairs. Pushing exactly those pairs apart grows the high-p_normal tail
        (raises specificity at the target sensitivity) far more directly than a plain
        BCE/CE, which spreads effort across the whole score range.

    pos_frac/neg_frac in (0,1] select the hard subsets; with small batches keep
    neg_frac=1.0 (few normals per batch) and pos_frac<1 to focus on hard pathology.
    The pAUC term is skipped for any batch missing either class.
    """
    def __init__(self, pauc_lambda=1.0, pos_frac=0.5, neg_frac=1.0,
                 margin=1.0, bce_pos_weight=1.0):
        super().__init__()
        self.pauc_lambda = float(pauc_lambda)
        self.pos_frac = float(pos_frac)
        self.neg_frac = float(neg_frac)
        self.margin = float(margin)
        self.register_buffer("pos_weight", torch.tensor(float(bce_pos_weight)))

    def forward(self, logit, target):
        logit = logit.flatten()
        target = target.flatten().float()                       # 1 = not-normal, 0 = normal
        loss = F.binary_cross_entropy_with_logits(
            logit, target, pos_weight=self.pos_weight.to(logit.device))
        pos = logit[target > 0.5]
        neg = logit[target <= 0.5]
        if self.pauc_lambda > 0 and pos.numel() > 0 and neg.numel() > 0:
            kp = max(1, int(round(self.pos_frac * pos.numel())))
            kn = max(1, int(round(self.neg_frac * neg.numel())))
            hard_pos = pos.topk(min(kp, pos.numel()), largest=False).values   # low-scoring pathology
            hard_neg = neg.topk(min(kn, neg.numel()), largest=True).values    # high-scoring normals
            diff = self.margin - (hard_pos.unsqueeze(1) - hard_neg.unsqueeze(0))   # kp x kn
            loss = loss + self.pauc_lambda * torch.clamp(diff, min=0).pow(2).mean()
        return loss


def build_ruleout_loss(cfg):
    return RuleOutLoss(
        pauc_lambda=getattr(cfg, "ruleout_pauc_lambda", 1.0),
        pos_frac=getattr(cfg, "ruleout_pos_frac", 0.5),
        neg_frac=getattr(cfg, "ruleout_neg_frac", 1.0),
        margin=getattr(cfg, "ruleout_margin", 1.0),
        bce_pos_weight=getattr(cfg, "ruleout_bce_pos_weight", 1.0),
    )


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
