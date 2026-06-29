# v3 Production Deployment — Abnormal Detection (OVR: abnormal vs normal+near_normal)

> **Model:** ct_brain v3 (`runs/maxvit384_3class_clinical_v3`, best.pt = ep3), MaxViT-MIL, 3-class.
> **Date:** 2026-06-29
> **Positive class:** **abnormal**; negative = normal + near_normal. Score = `P(abnormal)`.
> **Scoring unit:** per-patient, **mean** aggregation. **Evaluated on:** held-out test set (1,583 patients). ⚠️ enriched prevalence — see §5.
> Companion to `v3_production_deployment.md` (not-normal rule-out). This one targets the **most severe** class.

---

## 1. Use case — escalation, not rule-out

For abnormal, the clinically useful action is the *opposite* of the not-normal report: **flag likely-abnormal studies for urgent / senior-radiologist attention** (prioritize the worklist, speed up time-to-diagnosis). The safety-critical error is an **abnormal that is NOT escalated** — acceptable only if non-flagged studies are still read (just not prioritized). The AI **never down-ranks an abnormal out of human review**; it only re-orders the queue.

Test composition: **abnormal (positive) = 689 (43.5%)**, non-abnormal (normal+near) = 894 (56.5%).

---

## 2. Operating points at target ABNORMAL sensitivity (threshold on VAL → TEST)

Score = `P(abnormal)`; threshold maximizes specificity subject to the sensitivity floor. Pos = 689 abnormal, Neg = 894.

| target sens | threshold | Abnormal Sensitivity (TP/Pos) | Specificity (TN/Neg) | Precision (PPV) | NPV | FAR (1−spec) | flag rate | TP / FP / FN / TN |
|--:|--:|--:|--:|--:|--:|--:|--:|--|
| 0.95 | 0.025 | 94.9% (654/689) | 43.1% (385/894) | 56.2% | 91.7% | 56.9% | 73.5% | 654 / 509 / 35 / 385 |
| 0.96 | 0.020 | 95.9% (661/689) | 37.9% (339/894) | 54.4% | 92.4% | 62.1% | 76.8% | 661 / 555 / 28 / 339 |
| 0.97 | 0.015 | 97.1% (669/689) | 32.1% (287/894) | 52.4% | 93.5% | 67.9% | 80.6% | 669 / 607 / 20 / 287 |
| 0.98 | 0.012 | 98.0% (675/689) | 27.7% (248/894) | 51.1% | 94.7% | 72.3% | 83.4% | 675 / 646 / 14 / 248 |
| 0.99 | 0.004 | 99.1% (683/689) | 14.1% (126/894) | 47.1% | 95.5% | 85.9% | 91.7% | 683 / 768 / 6 / 126 |

**Reading:** high abnormal sensitivity is achievable but **useless for prioritization** — to catch 95% of abnormals you flag **73.5% of all studies** (precision 56%), and at 99% you flag **92%**. You can't "prioritize" three-quarters of the queue. The high-sensitivity regime only makes sense for the *inverse* read (the non-flagged bucket as a routine lane, NPV ≈ 92%).

---

## 3. The useful regime — high-precision escalation (default / argmax threshold)

Operating at the model's natural decision point (argmax — predict abnormal when it's the top class) gives a **much better escalation signal**:

| metric | value |
|---|--:|
| flag volume | 584 / 1,583 (**36.9%**) |
| Abnormal Sensitivity | 70.4% (485/689) |
| Specificity | 88.9% (795/894) |
| **Precision (PPV)** | **83.0% (485/584)** |
| abnormals NOT flagged (→ routine read) | 204 (29.6% of abnormals) |

At this point the escalated lane is **83% truly abnormal** at a manageable **37% flag volume** — a usable "read these first / send to senior" queue. The cost is sensitivity: ~30% of abnormals aren't escalated, so **they must still be read in the routine lane** (this is prioritization, not triage-out).

---

## 4. Two deployment patterns & their value

| pattern | operating point | what it does | benefit | risk |
|---|---|---|---|---|
| **Escalate (prioritize)** | argmax / high-precision | top 37% by P(abnormal) → urgent/senior queue | faster turnaround on true abnormals (83% of flags real); no studies removed | 30% of abnormals read at normal priority (not missed, just not expedited) |
| **Routine rule-out** | 95% abnormal-sens | non-flagged 27% → routine/low-priority lane | defers ~26% of worklist as low-urgency | 35 abnormals (5%) land in routine lane — slower read |

**Savings character:** this OVR does **not reduce headcount** (everything is still read). Its value is **turnaround-time**: surfacing severe cases earlier (escalation) and/or deferring confidently-non-abnormal studies to a low-priority lane (rule-out). Neither lane is fully autonomous, so safety risk is *delay*, not *missed-with-no-review* — far softer than the not-normal auto-report scenario.

---

## 5. Caveats

- **Prevalence (critical).** Test is **43.5% abnormal** (enriched). In a real population the abnormal base rate differs — a recent 1-week sample was ~45% abnormal (similar here), but precision/NPV must be **re-measured on production data**. Higher abnormal prevalence raises precision but lowers NPV (the routine lane gets riskier).
- **Calibrate first.** v3's `P(abnormal)` is overconfident; calibrate before fixing thresholds.
- **AUC ceiling (~0.85)** caps the precision/specificity trade-off — this is why high sensitivity floods the queue. A higher-AUC model (2-class or abnormal-focused head) would tighten it.
- **abnormal↔near_normal confusion** dominates the errors (92 near_normal flagged abnormal at argmax; 173 abnormal predicted near_normal in the full 3-class matrix) — the same noisy boundary that caps the 3-class model.

**Bottom line:** v3-abnormal-OVR is best deployed as a **high-precision escalation/prioritization signal** (argmax: 37% flagged, 83% precision) to speed up severe-case turnaround — *not* as a high-sensitivity gate (which flags most of the queue). It improves turnaround, not headcount, and every study still reaches a human.

---
*Operating points from `eval_abnormal_ops.py` over `series_probs_{val,test}.csv` (patient-level mean). Argmax point from the 3-class confusion matrix (`v3_triage_3lane_deployment.md` §2).*
