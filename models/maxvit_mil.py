"""
models/maxvit_mil.py
Study-level classifier = paper backbone (MaxViT) + gated-attention MIL pooling.

Why MIL: the paper classified one label per *image*. Here the label is one per
*study* (a bag of slices). Gated attention pooling (Ilse et al., 2018) lets the
model attend to the diagnostically relevant slices and produce a single study
decision, while exposing per-slice attention weights for interpretability
(complementing the Grad-CAM++ spatial maps from xai/).

Forward:
    bag:  B x K x 3 x H x W   (K slices, padded)
    mask: B x K               (True = real slice)
  ->
    logits:    B x num_classes
    attn:      B x K          (per-slice attention, softmax over real slices)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

from .build import build_backbone


class GatedAttentionPool(nn.Module):
    """Gated attention MIL pooling (Ilse et al., 2018)."""
    def __init__(self, in_dim, attn_dim, dropout=0.0):
        super().__init__()
        self.V = nn.Linear(in_dim, attn_dim)
        self.U = nn.Linear(in_dim, attn_dim)
        self.w = nn.Linear(attn_dim, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, H, mask):
        # H: B x K x D, mask: B x K
        a = torch.tanh(self.V(H)) * torch.sigmoid(self.U(H))   # B x K x attn_dim
        a = self.w(self.drop(a)).squeeze(-1)                   # B x K (raw scores)
        a = a.masked_fill(~mask, float("-inf"))                # ignore padded slices
        attn = torch.softmax(a, dim=1)                         # B x K
        attn = torch.nan_to_num(attn)                          # guard all-padded edge case
        Z = torch.bmm(attn.unsqueeze(1), H).squeeze(1)         # B x D  (weighted sum)
        return Z, attn


class TopKPool(nn.Module):
    """Per-slice scorer + top-k mean pooling for the binary rule-out head.

    Each slice gets a scalar 'not-normal' score; the bag score is the MEAN of the k
    highest slice scores (over real, non-padded slices). Unlike the attention-weighted
    *mean* of GatedAttentionPool, top-k pooling does NOT dilute sparse evidence — a
    finding present on only 1-2 slices still drives the bag score — which is exactly
    what is needed so subtle pathology cannot masquerade as a confident normal and
    contaminate the auto-report bucket. Returns (bag_logit B, slice_scores B x K).
    """
    def __init__(self, in_dim, k=8, hidden=256, dropout=0.0):
        super().__init__()
        self.k = int(k)
        self.scorer = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, H, mask):
        s = self.scorer(H).squeeze(-1)                         # B x K  per-slice not-normal score
        s = s.masked_fill(~mask, float("-inf"))                # ignore padded slices
        k = min(self.k, s.size(1))
        topv, _ = s.topk(k, dim=1)                             # B x k (highest-scoring slices)
        valid = torch.isfinite(topv)                           # guard bags with < k real slices
        bag = topv.masked_fill(~valid, 0.0).sum(1) / valid.sum(1).clamp(min=1)   # B
        return bag, torch.nan_to_num(s, neginf=0.0)


class MaxViTMIL(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone, feat_dim = build_backbone(cfg)          # per-slice encoder
        self.pool = GatedAttentionPool(feat_dim, cfg.mil_attn_dim, dropout=cfg.dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Dropout(cfg.dropout),
            nn.Linear(feat_dim, cfg.num_classes),
        )
        # v5: optional binary rule-out head (top-k pooling) sharing the backbone (multi-task).
        self.multitask = bool(getattr(cfg, "multitask_ruleout", False))
        if self.multitask:
            self.ruleout_pool = TopKPool(feat_dim, k=getattr(cfg, "ruleout_topk", 8),
                                         hidden=cfg.mil_attn_dim, dropout=cfg.dropout)

    def encode_slices(self, bag, mask):
        """bag B x K x 3 x H x W -> H_feats B x K x D (padding zeroed).

        Slices are encoded in chunks of cfg.slice_chunk (0 = all at once) so that
        activation memory does not grow with the bag size K. This is what makes
        feeding ALL slices of a set feasible:
          - freeze_backbone -> encode under no_grad (cheapest; head-only training)
          - grad_checkpoint -> recompute activations in backward (full fine-tune,
            bounded memory, ~20-30% slower)
        """
        B, K = bag.shape[:2]
        flat = bag.view(B * K, *bag.shape[2:])                 # (B*K) x 3 x H x W
        n = flat.size(0)
        chunk = self.cfg.slice_chunk or n                      # 0 -> single forward
        use_ckpt = self.cfg.grad_checkpoint and self.training and not self.cfg.freeze_backbone
        outs = []
        for i in range(0, n, chunk):
            sub = flat[i:i + chunk]
            if self.cfg.freeze_backbone:
                with torch.no_grad():
                    f = self.backbone(sub)
            elif use_ckpt:
                f = cp.checkpoint(self.backbone, sub, use_reentrant=False)
            else:
                f = self.backbone(sub)
            outs.append(f)
        feats = torch.cat(outs, 0).view(B, K, -1)              # (B*K) x D -> B x K x D
        feats = feats * mask.unsqueeze(-1)                     # zero padded slices
        return feats

    def forward(self, bag, mask, return_attn=False, return_ruleout=False):
        H = self.encode_slices(bag, mask)                      # B x K x D
        Z, attn = self.pool(H, mask)                           # B x D , B x K
        logits = self.head(Z)                                  # B x num_classes
        if not (return_attn or return_ruleout):
            return logits                                      # backward-compatible default
        out = [logits]
        if return_attn:
            out.append(attn)
        if return_ruleout:
            # B not-normal logit (None when the rule-out head is disabled)
            out.append(self.ruleout_pool(H, mask)[0] if self.multitask else None)
        return tuple(out)


def build_model(cfg):
    return MaxViTMIL(cfg)
