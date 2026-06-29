"""
eval_normal_ops.py
Re-score the dumped series probabilities (series_probs_{val,test}.csv) treating
NORMAL as the positive class ("confirm-normal / rule-out" operating point):
threshold chosen on val to hit a target NORMAL sensitivity, applied to test.
Reports series + patient-level (mean / max) at no GPU cost.

Usage: python eval_normal_ops.py --run runs/maxvit384_3class_clinical_v3
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
            rows.append((pid, int(lab), float(p0), float(p1), float(p2)))
    return rows


def aggregate(rows, level):
    if level == "series":
        y = np.array([r[1] for r in rows])
        P = np.array([[r[2], r[3], r[4]] for r in rows])
        return y, P
    byp = defaultdict(list)
    for r in rows:
        byp[r[0]].append(r)
    Y, P = [], []
    for pid, rs in byp.items():
        probs = np.array([[r[2], r[3], r[4]] for r in rs])
        y = rs[0][1]
        v = probs.mean(0) if level == "mean" else probs[int(np.argmin(probs[:, 0]))]
        Y.append(y); P.append(v)
    return np.array(Y), np.array(P)


def normal_op(vy, vp, ty, tp, targets):
    """Positive class = NORMAL (index 0). score = P(normal). Threshold on val -> test."""
    out = []
    v_pos = (vy == 0).astype(int); v_s = vp[:, 0]
    fpr, tpr, thr = roc_curve(v_pos, v_s)
    t_pos = (ty == 0); t_s = tp[:, 0]
    for tgt in targets:
        ok = tpr >= tgt
        idx = int(np.where(ok)[0][np.argmin(fpr[ok])]) if ok.any() else int(np.argmax(tpr))
        th = thr[idx]
        flag = t_s >= th                      # predict NORMAL
        TP = int((flag & t_pos).sum()); FP = int((flag & ~t_pos).sum())
        FN = int((~flag & t_pos).sum()); TN = int((~flag & ~t_pos).sum())
        sens = TP / max(TP+FN, 1); spec = TN / max(TN+FP, 1); prec = TP / max(TP+FP, 1)
        out.append((tgt, th, sens, spec, prec, 1-spec, TP, FP, FN, TN))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--targets", default="0.95,0.96,0.97,0.98,0.99")
    args = ap.parse_args()
    targets = [float(x) for x in args.targets.split(",")]
    vrows, trows = load(args.run, "val"), load(args.run, "test")

    for level, tag in [("series", "SERIES"), ("mean", "PATIENT [mean]"), ("max", "PATIENT [max]")]:
        vy, vp = aggregate(vrows, level); ty, tp = aggregate(trows, level)
        npos = int((ty == 0).sum())
        print(f"\n===== NORMAL operating points — {tag} (units {len(ty)}; normal(pos)={npos} not_normal={len(ty)-npos}) =====")
        print(f"  {'tgtSens':>7} {'thr':>6} | {'NSens':>6} {'NSpec':>6} {'NPrec':>6} {'NFAR':>6} | {'TP':>4} {'FP':>4} {'FN':>4} {'TN':>4}")
        for tgt, th, sens, spec, prec, far, TP, FP, FN, TN in normal_op(vy, vp, ty, tp, targets):
            print(f"  {tgt:>7.2f} {th:>6.3f} | {sens:>6.3f} {spec:>6.3f} {prec:>6.3f} {far:>6.3f} | {TP:>4} {FP:>4} {FN:>4} {TN:>4}")


if __name__ == "__main__":
    main()
