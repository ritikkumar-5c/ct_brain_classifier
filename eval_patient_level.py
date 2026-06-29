"""
eval_patient_level.py
Series-level vs patient-level evaluation. Collects per-series probabilities once
(val + test), then aggregates each patient's series two ways:
  - mean : average softmax across the patient's series
  - max  : take the most-pathological series (lowest P(normal)) = "flag if any series suspicious"
Reports 3-class metrics + normal-vs-not-normal operating-point sweep
(threshold chosen on val, applied to test) at series and patient level.
Also dumps per-series (patient_id,label,p0,p1,p2) CSVs for free re-aggregation.

Usage: python eval_patient_level.py --run runs/maxvit384_3class_clinical_v3
"""
import argparse, os
import numpy as np
import torch
from collections import defaultdict
from torch.utils.data import DataLoader

from config import get_config, Config
from data.dicom_dataset import index_studies_from_csv, StudyMILDataset, mil_collate, _study_of
from models.maxvit_mil import build_model
from engine.metrics import pathology_operating_point, compute_metrics


@torch.no_grad()
def collect(model, loader, device, use_amp):
    model.eval(); sids, ys, ps = [], [], []
    for batch in loader:
        bag = batch["bag"].to(device); mask = batch["mask"].to(device)
        with torch.autocast(device_type="cuda" if use_amp else "cpu",
                            dtype=torch.float16, enabled=use_amp):
            logits = model(bag, mask)
        sids.extend(batch["study_id"])
        ys.extend(batch["label"].tolist())
        ps.extend(torch.softmax(logits.float(), 1).cpu().tolist())
    return sids, np.array(ys), np.array(ps)


def aggregate(sids, ys, ps, method):
    byp = defaultdict(list)
    for i, s in enumerate(sids):
        byp[_study_of(s)].append(i)
    PY, PP = [], []
    for pid, idx in byp.items():
        probs = ps[idx]
        y = int(ys[idx][0])                       # labels consistent within patient
        v = probs.mean(0) if method == "mean" else probs[int(np.argmin(probs[:, 0]))]
        PY.append(y); PP.append(v)
    return np.array(PY), np.array(PP)


def report(name, vy, vp, ty, tp, class_names, targets):
    m = compute_metrics(ty.tolist(), tp.tolist(), len(class_names), class_names=class_names)
    npos = int((ty != 0).sum()); nneg = int((ty == 0).sum())
    print(f"\n===== {name}  (TEST units: {len(ty)}; not_normal={npos} normal={nneg}, prev={npos/len(ty):.3f}) =====")
    print(f"  3-class: acc={m['accuracy']:.4f} balAcc={m['balanced_acc']:.4f} auc={m['auc']:.4f} "
          f"rN/rNN/rA={m['recall_normal']:.3f}/{m['recall_near_normal']:.3f}/{m['recall_abnormal']:.3f}")
    print(f"  {'tgt':>4} {'thr':>6} | {'tSens':>6} {'tSpec':>6} {'tPrec':>6} {'FAR':>6} | {'TP':>4} {'FP':>4} {'FN':>4} {'TN':>4}")
    s = 1.0 - tp[:, 0]; isn = (ty == 0)
    for tgt in targets:
        thr = pathology_operating_point(vy, vp, target_sensitivity=tgt)["threshold"]
        flag = s >= thr
        TP = int((flag & ~isn).sum()); FP = int((flag & isn).sum())
        FN = int((~flag & ~isn).sum()); TN = int((~flag & isn).sum())
        sens = TP / max(TP+FN, 1); spec = TN / max(TN+FP, 1); prec = TP / max(TP+FP, 1)
        print(f"  {tgt:>4.2f} {thr:>6.3f} | {sens:>6.3f} {spec:>6.3f} {prec:>6.3f} {1-spec:>6.3f} | {TP:>4} {FP:>4} {FN:>4} {TN:>4}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--targets", default="0.95,0.98")
    args = ap.parse_args()
    targets = [float(x) for x in args.targets.split(",")]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(os.path.join(args.run, args.ckpt), map_location=device, weights_only=False)
    cfg = get_config(**{k: v for k, v in ck.get("cfg", {}).items() if hasattr(Config, k)})
    print(f"[cfg] {args.run} ckpt_epoch={ck.get('epoch')} image_size={cfg.image_size} max_slices={cfg.max_slices_per_study}")
    model = build_model(cfg); model.load_state_dict(ck["model"]); model.to(device)

    def loader(csv):
        ds = StudyMILDataset(index_studies_from_csv(csv, cfg.class_names), cfg, train=False)
        return DataLoader(ds, batch_size=cfg.batch_size, shuffle=False,
                          num_workers=cfg.num_workers, collate_fn=mil_collate, pin_memory=True)

    print("[collect] val ..."); vs, vy, vp = collect(model, loader(cfg.val_csv), device, cfg.use_amp)
    print("[collect] test ..."); ts, ty, tp = collect(model, loader(cfg.test_csv), device, cfg.use_amp)

    # dump per-series probs for free re-aggregation
    for split, sids, ys, ps in [("val", vs, vy, vp), ("test", ts, ty, tp)]:
        path = os.path.join(args.run, f"series_probs_{split}.csv")
        with open(path, "w") as f:
            f.write("patient_id,label,p_normal,p_near,p_abn\n")
            for sid, y, p in zip(sids, ys, ps):
                f.write(f"{_study_of(sid)},{int(y)},{p[0]:.6f},{p[1]:.6f},{p[2]:.6f}\n")
        print(f"[dump] {path}")

    report("SERIES-level", vy, vp, ty, tp, cfg.class_names, targets)
    for meth in ("mean", "max"):
        vyy, vpp = aggregate(vs, vy, vp, meth)
        tyy, tpp = aggregate(ts, ty, tp, meth)
        report(f"PATIENT-level [{meth}]", vyy, vpp, tyy, tpp, cfg.class_names, targets)


if __name__ == "__main__":
    main()
