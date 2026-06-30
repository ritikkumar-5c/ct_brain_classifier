"""
eval_cascade.py
Hierarchical / cascaded inference on the SINGLE v3 3-class softmax (no retraining).

Two stages on the per-patient mean probabilities (p_normal, p_near, p_abn):
  Stage 1 (isolate abnormal):  score = P(abn).  Threshold T1 from val ROC of
           abnormal-vs-(normal+near) at a target abnormal sensitivity.
           P(abn) >= T1  -> ABNORMAL (escalate); else -> Stage 2.
  Stage 2 (re-normalize, then split normal vs near on the pass-through):
           P_adj(near) = P(near) / (P(near) + P(normal)).   [re-normalization —
           Stage 1 consumed P(abn) mass, so raw P(near) is deflated.]
           Threshold T2 from val ROC of near-vs-normal (POSITIVE = near_normal,
           NEGATIVE = normal) on the val pass-through subset, at a target near
           sensitivity.  P_adj(near) >= T2 -> NEAR_NORMAL; else -> NORMAL.

All thresholds set on VAL, applied to TEST (no leakage). Patient-mean aggregation.
Reads series_probs_{val,test}.csv. Usage: --run runs/maxvit384_3class_clinical_v3
"""
import argparse, os
import numpy as np
from collections import defaultdict
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix


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
        byp[pid].append((lab, p0, p1, p2))
    y, P = [], []
    for pid, rs in byp.items():
        y.append(rs[0][0])
        P.append([np.mean([r[i] for r in rs]) for i in (1, 2, 3)])  # normal, near, abn
    return np.array(y), np.array(P)


def thr_at(y, score, poslab, target):
    pos = (y == poslab).astype(int)
    fpr, tpr, th = roc_curve(pos, score)
    ok = tpr >= target
    return th[int(np.where(ok)[0][np.argmin(fpr[ok])])] if ok.any() else th[int(np.argmax(tpr))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--s1-targets", default="0.95,0.98,0.99")
    ap.add_argument("--s2-target", type=float, default=0.95)
    args = ap.parse_args()

    vy, vP = patient_mean(load(args.run, "val"))
    ty, tP = patient_mean(load(args.run, "test"))
    N = len(ty)
    print(f"[patient-mean] test N={N} (normal {int((ty==0).sum())} / near {int((ty==1).sum())} / abn {int((ty==2).sum())})")
    print(f"AUC abn-vs-rest={roc_auc_score((ty==2).astype(int),tP[:,2]):.3f}  "
          f"near-vs-normal={roc_auc_score(ty[ty!=2]==1,(tP[ty!=2,1]/(tP[ty!=2,1]+tP[ty!=2,0]+1e-9))):.3f}\n")

    print("=== STAGE 1: abnormal vs (normal+near), score=P(abn) ===")
    for s1 in [float(x) for x in args.s1_targets.split(",")]:
        T1 = thr_at(vy, vP[:, 2], 2, s1)
        esc = tP[:, 2] >= T1
        TP = int((esc & (ty == 2)).sum()); FP = int((esc & (ty != 2)).sum())
        FN = int((~esc & (ty == 2)).sum())
        print(f"  target {s1:.2f}: T1={T1:.3f}  abn_sens={TP/(TP+FN)*100:.1f}% ({TP}/{TP+FN})  "
              f"escalated={int(esc.sum())} ({int(esc.sum())/N*100:.0f}% of worklist; abn {TP} + FP {FP})  "
              f"passed_to_S2={int((~esc).sum())}  abn_LOST_to_S2={FN}")

    s2 = args.s2_target
    print(f"\n=== END-TO-END TRIAGE (Stage 2 target near-sens={s2:.2f}) ===")
    print(f"{'S1tgt':>6} {'T1':>6} {'T2':>6} | ESCALATE(n/ne/ab)   LIGHT(n/ne/ab)   AUTOCLEAR(n/ne/ab=DANGER)")
    for s1 in [float(x) for x in args.s1_targets.split(",")]:
        T1 = thr_at(vy, vP[:, 2], 2, s1)
        vpass = vP[:, 2] < T1
        vadj = vP[vpass, 1] / (vP[vpass, 1] + vP[vpass, 0] + 1e-9)
        m = vy[vpass] != 2
        T2 = thr_at(vy[vpass][m], vadj[m], 1, s2)
        padj = tP[:, 1] / (tP[:, 1] + tP[:, 0] + 1e-9)
        pred = np.where(tP[:, 2] >= T1, 2, np.where(padj >= T2, 1, 0))

        def brk(c):
            s = pred == c
            return int((s & (ty == 0)).sum()), int((s & (ty == 1)).sum()), int((s & (ty == 2)).sum())
        e, l, a = brk(2), brk(1), brk(0)
        print(f"{s1:>6.2f} {T1:>6.3f} {T2:>6.3f} | {sum(e):>4} {e[0]}/{e[1]}/{e[2]}   "
              f"{sum(l):>3} {l[0]}/{l[1]}/{l[2]}   {sum(a):>3} {a[0]}/{a[1]}/{a[2]}")
        print(f"        escalate {sum(e)/N*100:.0f}%;  auto-clear-normal={sum(a)} ({a[0]} true-normal, {a[2]} abnormal-as-NORMAL);  abnormals not escalated={l[2]+a[2]}")

    # hard 3-class labeler comparison at (s1=0.98, s2 target)
    T1 = thr_at(vy, vP[:, 2], 2, 0.98)
    vpass = vP[:, 2] < T1; vadj = vP[vpass, 1] / (vP[vpass, 1] + vP[vpass, 0] + 1e-9); m = vy[vpass] != 2
    T2 = thr_at(vy[vpass][m], vadj[m], 1, s2)
    padj = tP[:, 1] / (tP[:, 1] + tP[:, 0] + 1e-9)
    casc = np.where(tP[:, 2] >= T1, 2, np.where(padj >= T2, 1, 0))
    flat = np.argmax(tP, axis=1)
    print(f"\n=== AS A HARD 3-CLASS LABELER (cascade S1=0.98/S2={s2:.2f}) vs FLAT argmax ===")
    for tag, pred in [("CASCADE", casc), ("FLAT  ", flat)]:
        cm = confusion_matrix(ty, pred, labels=[0, 1, 2])
        rec = [cm[i, i] / cm[i].sum() * 100 for i in range(3)]
        print(f"  {tag}: recall normal={rec[0]:.1f} near={rec[1]:.1f} abn={rec[2]:.1f}  balanced_acc={np.mean(rec):.1f}")


if __name__ == "__main__":
    main()
