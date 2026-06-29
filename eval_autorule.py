"""
eval_autorule.py
Auto-rule-out deployment analysis for v3 (patient-level, mean aggregation).
The AI auto-clears confidently-normal studies (s = 1-P(normal) < threshold);
everything else goes to a doctor. Threshold set on val to hit a target
NOT-NORMAL sensitivity (safety: don't auto-clear pathology), applied to test.

Reports, per safety operating point:
  - threshold, achieved test sensitivity / specificity
  - % of NORMAL studies automated  (= specificity)
  - missed pathology (FN) and miss rate (1 - sensitivity)
Then projects workload reduction + NPV at several deployment prevalences.

Reads series_probs_{val,test}.csv (no GPU). Usage: --run runs/maxvit384_3class_clinical_v3
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
            rows.append((pid, int(lab), float(p0)))
    return rows


def patient_mean(rows):
    byp = defaultdict(list)
    for pid, lab, p0 in rows:
        byp[pid].append((lab, p0))
    y, pnorm = [], []
    for pid, rs in byp.items():
        y.append(rs[0][0])
        pnorm.append(np.mean([r[1] for r in rs]))
    y = np.array(y); s = 1.0 - np.array(pnorm)        # not-normal score
    return y, s


def threshold_for(vy, vs, target):
    pos = (vy != 0).astype(int)
    fpr, tpr, thr = roc_curve(pos, vs)
    ok = tpr >= target
    idx = int(np.where(ok)[0][np.argmin(fpr[ok])]) if ok.any() else int(np.argmax(tpr))
    return thr[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--targets", default="0.95,0.98,0.99,0.995,0.999,1.0")
    ap.add_argument("--prevalences", default="0.792,0.50,0.20,0.10",
                    help="fraction NOT-NORMAL in deployment population")
    args = ap.parse_args()
    targets = [float(x) for x in args.targets.split(",")]
    prevs = [float(x) for x in args.prevalences.split(",")]

    vy, vs = patient_mean(load(args.run, "val"))
    ty, ts = patient_mean(load(args.run, "test"))
    n_norm = int((ty == 0).sum()); n_path = int((ty != 0).sum())
    print(f"[patient-level, mean]  val={len(vy)}  test={len(ty)} (normal={n_norm}, not_normal={n_path}, "
          f"test prev_not_normal={n_path/len(ty):.3f})\n")

    print("=== Safety operating points (threshold set on val, measured on test) ===")
    print(f"{'tgtSens':>7} {'thr':>6} | {'tSens':>6} {'tSpec':>6} | {'%NORMAL_auto':>12} {'missedPath':>10} {'missRate':>8}")
    ops = []
    for tgt in targets:
        th = threshold_for(vy, vs, tgt)
        flag = ts >= th                       # flagged -> doctor; else auto-cleared normal
        TP = int((flag & (ty != 0)).sum()); FP = int((flag & (ty == 0)).sum())
        FN = int((~flag & (ty != 0)).sum()); TN = int((~flag & (ty == 0)).sum())
        Se = TP/max(TP+FN, 1); Sp = TN/max(TN+FP, 1)
        ops.append((tgt, th, Se, Sp, FN))
        print(f"{tgt:>7.3f} {th:>6.3f} | {Se:>6.3f} {Sp:>6.3f} | {Sp*100:>11.1f}% {FN:>10} {1-Se:>8.4f}")

    print("\n=== Deployment projection at different NOT-NORMAL prevalences ===")
    print("(workload_auto = % of ALL studies auto-cleared; NPV = of auto-cleared, %truly normal;")
    print(" missed_per_1000 = pathology auto-cleared per 1000 studies)")
    for tgt, th, Se, Sp, FN in ops:
        print(f"\n-- operating point: target not-normal sens={tgt:.3f}  (test Se={Se:.3f}, Sp={Sp:.3f}) --")
        print(f"   {'prev_notNorm':>12} {'%normal':>8} | {'workload_auto':>13} {'NPV':>7} {'missed/1000':>11}")
        for p in prevs:                       # p = fraction NOT-normal
            auto = (1-p)*Sp + p*(1-Se)        # fraction of all studies auto-cleared
            npv = ((1-p)*Sp) / max((1-p)*Sp + p*(1-Se), 1e-9)
            missed = p*(1-Se)*1000
            print(f"   {p:>12.3f} {(1-p)*100:>7.0f}% | {auto*100:>12.1f}% {npv:>7.3f} {missed:>11.1f}")


if __name__ == "__main__":
    main()
