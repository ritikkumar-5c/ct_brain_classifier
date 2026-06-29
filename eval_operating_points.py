"""
eval_operating_points.py
Load a trained best.pt, collect val + test not-normal probabilities ONCE, then
report the normal-vs-not-normal screening operating point at several target
sensitivities (threshold chosen on val, applied to test — no leakage).
Exact precision/FAR computed from test confusion counts.

Usage: python eval_operating_points.py --run runs/maxvit384_3class_clinical_v3
"""
import argparse, glob, os
import numpy as np
import torch
from torch.utils.data import DataLoader

from config import get_config, Config
from data.dicom_dataset import index_studies_from_csv, StudyMILDataset, mil_collate
from models.maxvit_mil import build_model
from engine.metrics import pathology_operating_point, apply_operating_point, compute_metrics


@torch.no_grad()
def collect(model, loader, device, use_amp):
    model.eval(); ys, ps = [], []
    for batch in loader:
        bag = batch["bag"].to(device); mask = batch["mask"].to(device)
        with torch.autocast(device_type="cuda" if use_amp else "cpu",
                            dtype=torch.float16, enabled=use_amp):
            logits = model(bag, mask)
        ys.extend(batch["label"].tolist())
        ps.extend(torch.softmax(logits.float(), 1).cpu().tolist())
    return np.array(ys), np.array(ps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--targets", default="0.95,0.96,0.97,0.98,0.99")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(os.path.join(args.run, args.ckpt), map_location=device, weights_only=False)
    saved = {k: v for k, v in ckpt.get("cfg", {}).items() if hasattr(Config, k)}
    cfg = get_config(**saved)
    print(f"[cfg] image_size={cfg.image_size} max_slices={cfg.max_slices_per_study} "
          f"num_classes={cfg.num_classes} ckpt_epoch={ckpt.get('epoch')}")

    model = build_model(cfg); model.load_state_dict(ckpt["model"]); model.to(device)

    def loader(csv):
        items = index_studies_from_csv(csv, cfg.class_names)
        ds = StudyMILDataset(items, cfg, train=False)   # train=False -> K96 eval sampling
        return DataLoader(ds, batch_size=cfg.batch_size, shuffle=False,
                          num_workers=cfg.num_workers, collate_fn=mil_collate, pin_memory=True)

    print("[collect] val ..."); vy, vp = collect(model, loader(cfg.val_csv), device, cfg.use_amp)
    print("[collect] test ..."); ty, tp = collect(model, loader(cfg.test_csv), device, cfg.use_amp)

    # full 3-class held-out metrics (val + test)
    for name, (yy, pp) in [("VAL", (vy, vp)), ("TEST", (ty, tp))]:
        m = compute_metrics(yy.tolist(), pp.tolist(), cfg.num_classes, class_names=cfg.class_names)
        print(f"\n=== {name} 3-class metrics (argmax) ===")
        for k in ("accuracy", "balanced_acc", "auc", "f1", "precision",
                  "recall_normal", "recall_near_normal", "recall_abnormal",
                  "normal_specificity", "not_normal_sensitivity"):
            if k in m:
                print(f"  {k:28s} = {m[k]:.4f}")

    npos = int((ty != 0).sum()); nneg = int((ty == 0).sum())
    print(f"\nTEST set: not_normal(pos)={npos}  normal(neg)={nneg}  prevalence_pos={npos/(npos+nneg):.3f}")
    print(f"\n{'target':>6} {'thr':>6} | {'valSens':>7} {'valSpec':>7} | "
          f"{'tstSens':>7} {'tstSpec':>7} {'tstPrec':>7} {'FAR':>6} | {'TP':>4} {'FP':>4} {'FN':>4} {'TN':>4}")
    s_test = 1.0 - tp[:, 0]
    is_norm = (ty == 0)
    for tgt in [float(x) for x in args.targets.split(",")]:
        op = pathology_operating_point(vy, vp, target_sensitivity=tgt)
        thr = op["threshold"]
        flag = s_test >= thr
        TP = int((flag & ~is_norm).sum()); FP = int((flag & is_norm).sum())
        FN = int((~flag & ~is_norm).sum()); TN = int((~flag & is_norm).sum())
        sens = TP / max(TP + FN, 1); spec = TN / max(TN + FP, 1)
        prec = TP / max(TP + FP, 1); far = 1 - spec
        print(f"{tgt:>6.2f} {thr:>6.3f} | {op['sensitivity']:>7.3f} {op['specificity']:>7.3f} | "
              f"{sens:>7.3f} {spec:>7.3f} {prec:>7.3f} {far:>6.3f} | {TP:>4} {FP:>4} {FN:>4} {TN:>4}")


if __name__ == "__main__":
    main()
