"""
infer.py
Run a trained checkpoint on a single study folder of DICOM slices.
Outputs: predicted label + probability, per-slice attention, and Grad-CAM++
overlays saved as PNGs for the top-attended slices.

  python infer.py --ckpt runs/maxvit_mil/best.pt --study_dir /data/ct_studies/study_0007 \
      --out_dir explanations/study_0007 --topk 3
"""
import os
import glob
import argparse
import numpy as np
import torch
from PIL import Image

from config import get_config
from data.dicom_dataset import _instance_number
from data.transforms import build_transforms, dicom_to_multiwindow
from models.maxvit_mil import build_model
from xai.gradcampp import GradCAMpp, denormalize, overlay
import pydicom


def load_study(study_dir, cfg):
    paths = sorted(glob.glob(os.path.join(study_dir, "**", "*.dcm"), recursive=True))
    if not paths:
        paths = [p for p in glob.glob(os.path.join(study_dir, "**", "*"), recursive=True) if os.path.isfile(p)]
    paths = sorted(paths, key=_instance_number)
    paths = paths[: cfg.max_slices_per_study]
    tf = build_transforms(cfg, train=False)
    slices = []
    for p in paths:
        ds = pydicom.dcmread(p)
        img = dicom_to_multiwindow(ds, cfg.windows)     # HxWx3 (brain/subdural/bone)
        slices.append(tf(img))
    bag = torch.stack(slices)                     # K x 3 x H x W
    return bag, paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--study_dir", required=True)
    ap.add_argument("--out_dir", default="explanations")
    ap.add_argument("--topk", type=int, default=3)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    state = torch.load(args.ckpt, map_location=device)
    cfg = get_config(**{k: v for k, v in state["cfg"].items() if hasattr(get_config(), k)})
    model = build_model(cfg).to(device)
    model.load_state_dict(state["model"]); model.eval()

    bag, paths = load_study(args.study_dir, cfg)
    bag = bag.to(device)
    mask = torch.ones(1, bag.size(0), dtype=torch.bool, device=device)

    with torch.no_grad():
        logits, attn = model(bag.unsqueeze(0), mask, return_attn=True)
        prob = torch.softmax(logits, 1)[0].cpu().numpy()
        attn = attn[0].cpu().numpy()
    pred = int(prob.argmax())
    label_name = cfg.class_names[pred]
    probs_str = ", ".join(f"{n}={p:.3f}" for n, p in zip(cfg.class_names, prob))
    print(f"Study prediction: {label_name}  ({probs_str})")

    # Grad-CAM++ on the top-attended slices
    cam = GradCAMpp(model, cfg.gradcam_layer)
    try:
        top_idx = np.argsort(attn)[::-1][: args.topk]
        for rank, i in enumerate(top_idx):
            slice_t = bag[i].clone().requires_grad_(True)
            heat = cam(slice_t, target_class=cfg.num_classes - 1)
            img = denormalize(bag[i], cfg.norm_mean, cfg.norm_std)
            blended = overlay(img, heat)
            fn = os.path.join(args.out_dir, f"rank{rank}_slice{i}_attn{attn[i]:.3f}.png")
            Image.fromarray(blended).save(fn)
            print(f"  saved {fn}")
    finally:
        cam.remove()


if __name__ == "__main__":
    main()
