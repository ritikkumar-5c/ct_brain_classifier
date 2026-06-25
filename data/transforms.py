"""
data/transforms.py
DICOM -> tensor pipeline.

Paper steps reproduced:
 - 4.4 Preprocessing: resize to 224x224; grayscale single channel replicated to 3 channels;
       ImageNet normalization (required by the timm pretrained backbones).
 - 4.5.1 Classical augmentation: random crop, horizontal flip, rotation, color jitter.

DICOM-specific step (the paper used pre-exported PNGs): Hounsfield-Unit windowing.
Raw DICOM pixels are converted to HU via RescaleSlope/RescaleIntercept, then a brain
window (center=40, width=80 by default) is applied and rescaled to [0,255].
"""
import numpy as np
import torch
import torchvision.transforms.v2 as T


def dicom_to_hu(ds) -> np.ndarray:
    """pydicom Dataset -> float32 HU array."""
    arr = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    return arr * slope + intercept


def apply_window(hu: np.ndarray, center: float, width: float) -> np.ndarray:
    """HU -> uint8 [0,255] under a (center,width) window."""
    lo, hi = center - width / 2.0, center + width / 2.0
    out = np.clip(hu, lo, hi)
    out = (out - lo) / max(hi - lo, 1e-6)
    return (out * 255.0).astype(np.uint8)


def dicom_to_uint8(ds, center=None, width=None) -> np.ndarray:
    """Full DICOM->displayable uint8 single-channel image with windowing."""
    hu = dicom_to_hu(ds)
    if center is None or width is None:
        # fall back to the window stored in the DICOM tags if present
        wc = getattr(ds, "WindowCenter", 40.0)
        ww = getattr(ds, "WindowWidth", 80.0)
        center = float(wc[0] if isinstance(wc, (list, tuple)) or hasattr(wc, "__len__") and not isinstance(wc, str) else wc)
        width = float(ww[0] if isinstance(ww, (list, tuple)) or hasattr(ww, "__len__") and not isinstance(ww, str) else ww)
    return apply_window(hu, center, width)


def dicom_to_multiwindow(ds, windows, jitter: float = 0.0) -> np.ndarray:
    """DICOM -> HxWx3 uint8, one clinical window per channel.

    `windows` is a sequence of (center, width) pairs. The HU array is computed
    once, then each window is applied to its own channel:
        ch0 = brain (W80/L40), ch1 = subdural (W200/L80), ch2 = bone (W2800/L600).

    `jitter` > 0 enables HU-window augmentation: each channel's center and width
    are independently scaled by U(1-jitter, 1+jitter) before windowing. This is a
    CT-specific intensity augmentation (use only at training time).
    """
    hu = dicom_to_hu(ds)
    chans = []
    for (c, w) in windows:
        c, w = float(c), float(w)
        if jitter and jitter > 0.0:
            c *= float(np.random.uniform(1.0 - jitter, 1.0 + jitter))
            w *= float(np.random.uniform(1.0 - jitter, 1.0 + jitter))
        chans.append(apply_window(hu, c, w))
    return np.stack(chans, axis=-1)                    # HxWx3 uint8


def build_transforms(cfg, train: bool):
    """Returns a callable: uint8 HxWx3 numpy (multi-window) -> 3xHxW float tensor."""
    size = cfg.image_size
    base = [
        T.ToImage(),                                   # numpy HxWx3 -> tv tensor (keeps 3 channels)
    ]
    if train:
        # geometric/photometric augs, each toggled by its config knob.
        # (HU-window jitter is applied earlier, in dicom_to_multiwindow.)
        aug = []
        if cfg.aug_crop_scale_min < 1.0:
            aug.append(T.RandomResizedCrop(size, scale=(cfg.aug_crop_scale_min, 1.0), antialias=True))
        else:
            aug.append(T.Resize((size, size), antialias=True))
        if cfg.aug_hflip:
            aug.append(T.RandomHorizontalFlip(p=0.5))
        if cfg.aug_rotation_deg and cfg.aug_rotation_deg > 0:
            aug.append(T.RandomRotation(degrees=cfg.aug_rotation_deg))
        if (cfg.aug_brightness or cfg.aug_contrast):
            aug.append(T.ColorJitter(brightness=cfg.aug_brightness, contrast=cfg.aug_contrast))
    else:
        aug = [T.Resize((size, size), antialias=True)]
    tail = [
        T.ToDtype(torch.float32, scale=True),          # -> [0,1]
        T.Normalize(mean=cfg.norm_mean, std=cfg.norm_std),
    ]
    return T.Compose(base + aug + tail)
