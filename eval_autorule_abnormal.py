"""
eval_autorule_abnormal.py
Auto-rule-out deployment analysis for v3 — ABNORMAL objective (patient-level, mean agg).

Companion to eval_autorule.py. Here the positive (dangerous) class is ABNORMAL only;
the negative pool is normal + near_normal. The AI auto-clears studies that are
confidently NOT-abnormal (s = P(abnormal) < threshold) — this auto-reports BOTH
normal AND near_normal studies; everything that looks abnormal goes to a doctor.
Threshold set on val to hit a target ABNORMAL sensitivity (safety: don't auto-clear
abnormal pathology), applied to test. No leakage.

Reports, per safety operating point:
  - threshold, achieved test abnormal sensitivity / specificity
  - % of NON-abnormal studies automated (= specificity), split normal vs near_normal
  - missed ABNORMAL (FN) and miss rate (1 - sensitivity)
  - NPV (of auto-cleared, % truly non-abnormal)
Then projects workload reduction + NPV at several deployment prevalences.

Reads series_probs_{val,test}.csv (no GPU). Usage: --run runs/maxvit384_3class_clinical_v3
"""
import argparse, os
import numpy as np
from collections import defaultdict
from sklearn.metrics import roc_curve, roc_auc_score


def load(run, split):
    rows = []
    with open(os.path.join(run, f"series_probs_{split}.csv")) as f:
        next(f)
        for line in f:
            pid, lab, p0, p1, p2 = line.strip().split(",")
            rows.append((pid, int(lab), float(p0), float(p1), float(p2)))
    return rows


def patient_mean(rows):
    byp = defaultdict(list)
    for pid, lab, p0, p1, p2 in rows:
        byp[pid].append((lab, p2))            # p2 = P(abnormal) = the abnormal score
    y, pabn = [], []
    for pid, rs in byp.items():
        y.append(rs[0][0])
        pabn.append(np.mean([r[1] for r in rs]))
    return np.array(y), np.array(pabn)         # s = P(abnormal)


def threshold_for(vy, vs, target):
    pos = (vy == 2).astype(int)                # positive = abnormal only
    fpr, tpr, thr = roc_curve(pos, vs)
    ok = tpr >= target
    idx = int(np.where(ok)[0][np.argmin(fpr[ok])]) if ok.any() else int(np.argmax(tpr))
    return thr[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--targets", default="0.95,0.98,0.99,0.995,0.999,1.0")
    ap.add_argument("--prevalences", default="0.435,0.30,0.10,0.05",
                    help="fraction ABNORMAL in deployment population")
    args = ap.parse_args()
    targets = [float(x) for x in args.targets.split(",")]
    prevs = [float(x) for x in args.prevalences.split(",")]

    vy, vs = patient_mean(load(args.run, "val"))
    ty, ts = patient_mean(load(args.run, "test"))
    n_abn = int((ty == 2).sum()); n_neg = int((ty != 2).sum())
    n_norm = int((ty == 0).sum()); n_near = int((ty == 1).sum())
    auc = roc_auc_score((ty == 2).astype(int), ts)
    print(f"[patient-level, mean]  val={len(vy)}  test={len(ty)} "
          f"(abnormal={n_abn}, non_abnormal={n_neg} [normal={n_norm}, near={n_near}], "
          f"test prev_abnormal={n_abn/len(ty):.3f}, abn-vs-rest AUC={auc:.3f})\n")

    print("=== Safety operating points (threshold set on val, measured on test) ===")
    print(f"{'tgtSens':>7} {'thr':>6} | {'Sens(TP/Pos)':>16} {'Spec=%cleared(TN/Neg)':>22} "
          f"{'missedAbn(FN/Pos)':>18} {'missRate':>8} {'NPV':>7} | {'TN: norm+near':>14} {'FP: norm+near':>14}")
    ops = []
    for tgt in targets:
        th = threshold_for(vy, vs, tgt)
        flag = ts >= th                       # flagged -> doctor (looks abnormal); else auto-cleared
        TP = int((flag & (ty == 2)).sum()); FP = int((flag & (ty != 2)).sum())
        FN = int((~flag & (ty == 2)).sum()); TN = int((~flag & (ty != 2)).sum())
        Se = TP/max(TP+FN, 1); Sp = TN/max(TN+FP, 1); npv = TN/max(TN+FN, 1)
        tn_norm = int((~flag & (ty == 0)).sum()); tn_near = int((~flag & (ty == 1)).sum())
        fp_norm = int((flag & (ty == 0)).sum()); fp_near = int((flag & (ty == 1)).sum())
        ops.append((tgt, th, Se, Sp, FN, npv))
        print(f"{tgt:>7.3f} {th:>6.3f} | {Se*100:>6.1f}% {TP:>4}/{TP+FN:<4} "
              f"{Sp*100:>6.1f}% {TN:>3}/{TN+FP:<4} {FN:>5}/{FN+TP:<4} {1-Se:>8.4f} {npv*100:>6.1f}% | "
              f"{tn_norm:>5}+{tn_near:<5} {fp_norm:>5}+{fp_near:<5}")

    print("\n=== Deployment projection at different ABNORMAL prevalences ===")
    print("(workload_auto = % of ALL studies auto-cleared; NPV = of auto-cleared, %truly non-abnormal;")
    print(" missed_per_1000 = abnormal auto-cleared per 1000 studies)")
    for tgt, th, Se, Sp, FN, _ in ops:
        print(f"\n-- operating point: target abnormal sens={tgt:.3f}  (test Se={Se:.3f}, Sp={Sp:.3f}) --")
        print(f"   {'prev_abn':>12} {'%nonAbn':>8} | {'workload_auto':>13} {'NPV':>7} {'missed/1000':>11}")
        for p in prevs:                       # p = fraction ABNORMAL
            auto = (1-p)*Sp + p*(1-Se)
            npv = ((1-p)*Sp) / max((1-p)*Sp + p*(1-Se), 1e-9)
            missed = p*(1-Se)*1000
            print(f"   {p:>12.3f} {(1-p)*100:>7.0f}% | {auto*100:>12.1f}% {npv:>7.3f} {missed:>11.1f}")


if __name__ == "__main__":
    main()
