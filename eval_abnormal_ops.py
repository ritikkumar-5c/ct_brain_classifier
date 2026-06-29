"""
eval_abnormal_ops.py
Abnormal-vs-rest (OVR) operating points for v3, patient-level mean aggregation.
Positive class = ABNORMAL; score = P(abnormal). Threshold set on val to hit a
target ABNORMAL sensitivity, applied to test. Use case: auto-escalate / prioritize
likely-abnormal studies. Reads dumped series_probs_{val,test}.csv (no GPU).

Usage: python eval_abnormal_ops.py --run runs/maxvit384_3class_clinical_v3
"""
import argparse, os
import numpy as np
from collections import defaultdict
from sklearn.metrics import roc_curve


def load(run, split):
    rows = []
    with open(os.path.join(run, f"series_probs_{split}.csv")) as f:
        next(f)
        for line in f:
            pid, lab, p0, p1, p2 = line.strip().split(",")
            rows.append((pid, int(lab), float(p2)))   # p2 = P(abnormal)
    return rows


def patient_mean(rows):
    byp = defaultdict(list)
    for pid, lab, p2 in rows:
        byp[pid].append((lab, p2))
    y, s = [], []
    for pid, rs in byp.items():
        y.append(1 if rs[0][0] == 2 else 0)            # 1 = abnormal
        s.append(np.mean([r[1] for r in rs]))          # mean P(abnormal)
    return np.array(y), np.array(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--targets", default="0.95,0.96,0.97,0.98,0.99")
    args = ap.parse_args()
    targets = [float(x) for x in args.targets.split(",")]
    vy, vs = patient_mean(load(args.run, "val"))
    ty, ts = patient_mean(load(args.run, "test"))
    P = int((ty == 1).sum()); Ngeg = int((ty == 0).sum())
    print(f"[patient-level mean]  test={len(ty)}  abnormal(pos)={P}  non-abnormal(neg)={Ngeg}  "
          f"prev_abn={P/len(ty):.3f}\n")
    fpr, tpr, thr = roc_curve(vy, vs)
    print(f"{'tgtSens':>7} {'thr':>6} | {'Sens':>6} {'Spec':>6} {'Prec':>6} {'NPV':>6} {'FAR':>6} | {'TP':>4} {'FP':>4} {'FN':>4} {'TN':>4}")
    for tgt in targets:
        ok = tpr >= tgt
        idx = int(np.where(ok)[0][np.argmin(fpr[ok])]) if ok.any() else int(np.argmax(tpr))
        th = thr[idx]
        flag = ts >= th
        TP = int((flag & (ty == 1)).sum()); FP = int((flag & (ty == 0)).sum())
        FN = int((~flag & (ty == 1)).sum()); TN = int((~flag & (ty == 0)).sum())
        Se = TP/max(TP+FN, 1); Sp = TN/max(TN+FP, 1)
        prec = TP/max(TP+FP, 1); npv = TN/max(TN+FN, 1)
        print(f"{tgt:>7.3f} {th:>6.3f} | {Se*100:>5.1f}% {Sp*100:>5.1f}% {prec*100:>5.1f}% {npv*100:>5.1f}% {(1-Sp)*100:>5.1f}% | {TP:>4} {FP:>4} {FN:>4} {TN:>4}")


if __name__ == "__main__":
    main()
