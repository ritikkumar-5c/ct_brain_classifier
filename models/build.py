"""
models/build.py
Backbone factory. MaxViT is the primary model from the paper; ViT/TNT/ConvNeXt
are the comparison architectures (paper Sec 4.5). All are loaded from `timm`
as feature extractors (num_classes=0 -> pooled embedding per image/slice).
"""
import timm
import torch.nn as nn

# Maps the paper's backbone names to concrete timm model ids.
TIMM_IDS = {
    "maxvit":   "maxvit_tiny_tf_224.in1k",     # primary (CNN + multi-axis attention)
    "vit":      "vit_base_patch16_224.augreg2_in21k_ft_in1k",  # baseline
    "tnt":      "tnt_s_patch16_224",           # transformer-in-transformer
    "convnext": "convnext_base.fb_in1k",       # hybrid CNN
}


def build_backbone(cfg):
    """Returns (backbone_module, feature_dim). Backbone maps 3xHxW -> feature_dim vector."""
    timm_id = cfg.timm_name if cfg.timm_name else TIMM_IDS[cfg.backbone]
    backbone = timm.create_model(
        timm_id,
        pretrained=cfg.pretrained,
        num_classes=0,        # remove classifier -> pooled features
        drop_rate=cfg.dropout,
    )
    feat_dim = backbone.num_features
    if cfg.freeze_backbone:
        # paper froze everything but the final classifier; we expose this as a flag
        for p in backbone.parameters():
            p.requires_grad = False
    return backbone, feat_dim
