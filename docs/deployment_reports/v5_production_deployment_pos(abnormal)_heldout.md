# v5 Production Deployment — AI Auto-Rule-Out (ABNORMAL objective) (TEST: held-out split)

> **Model:** ct_brain v5 (`runs/maxvit384_3class_clinical_v5`, best snapshot = ep8; **training still ongoing**), MaxViT-MIL, 3-class (+ rule-out multi-task head).
> **Date:** 2026-07-01
> **Test set:** the original patient-grouped **held-out** test split (`train_data/csvs/splits/test.csv`) — enriched, not a deployment population.
> **Objective:** positive = **abnormal**, negative = **non-abnormal (normal + near_normal)**. Decision score `s = P(abnormal)` (per study, mean over series). Thresholds set on the original validation set, measured on this test set.
> **Companion:** `v5_production_deployment_pos(near-normal_abnormal)_heldout.md`

---

## 1. Core principle — rule-out only, never auto-diagnose

The AI does **one** autonomous thing: **auto-clear a confidently-negative study** off the worklist. It never diagnoses pathology.

| | |
|---|---|
| **Positive (Pos)** | abnormal |
| **Negative (Neg)** | non-abnormal (normal + near_normal) |
| **Decision score** | `s = P(abnormal)` (per study) |
| **Auto-clear if** | `s < threshold` → skip doctor |
| **Threshold set by** | max tolerable **miss rate** (safety budget) |
| **Only dangerous error** | an **abnormal study auto-cleared** (a cleared near_normal is an accepted outcome) |

---

## 2. Real-time production flow

```
   CT scanner / PACS ──DICOM──> 1.Ingest ─> 2.Preprocess (HU->3 windows, 96 slices@384px)
   ─> 3.QC fail-safe (unreadable/atypical -> doctor) ─> 4.Inference v5 (P(normal,near,abn))
   ─> 5.Study aggregate (mean over series) -> score ─> 6.Decision: score < threshold ?
        YES -> 7a.AUTO-CLEAR (skip doctor, auto-draft, X% audit)
        NO  -> 7b.FLAG -> radiologist (prioritized by score)
   ─> 8.Monitoring (auto-clear rate, audited miss rate, drift, prevalence; auto-revert)
```

**Safety nets:** QC fail-safe → doctor; random audit of auto-cleared studies to measure the *real* miss rate; prevalence/drift monitoring with auto-revert; human override always wins. DICOM decoding covers JPEG-Lossless / JPEG-2000 / JPEG-LS (`pylibjpeg-*` + `python-gdcm`).

---

## 2b. Test-set composition — held-out split

Scored **per study** (mean over series) → denominators are studies.

| class | series | studies |
|---|--:|--:|
| normal | 546 | 336 |
| near_normal | 942 | 558 |
| abnormal | 1,135 | 689 |
| **total** | **2,623** | **1,583** |

Prevalence: normal **21.2%**, not-normal **78.8%**, abnormal **43.5%**. Pos = 689 abnormal, Neg = 894.

Discrimination on this set (per-study, mean aggregation):

| boundary | AUC |
|---|--:|
| not-normal vs normal | 0.901 |
| abnormal vs normal only | 0.932 |
| abnormal vs (normal + near_normal) | 0.853 |
| near_normal vs normal | 0.761 |
| 3-class macro (OvR) | 0.834 |

---

## 3. Choosing the threshold = choosing a safety budget

Thresholds set on the original validation set, applied to this test set. **Auto-Cleared (PN)** = predicted-negative (leaves the worklist); it contains the correct clears **and** the misses.

| Target Sens (val) | Threshold | Sensitivity achieved (TP/Pos) | Auto-Cleared (PN) | ↳ normal (TN ✓) | ↳ near_normal (TN ✓) | ↳ abnormal (FN ✗) | Miss rate (FN/Pos) | Neg Precision |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 95% | 0.070 | 95.5% (658/689) | 24.2% (383/1,583) | 226 | 126 | **31** | 4.50% (31/689) | 91.9% |
| 98% | 0.048 | 98.0% (675/689) | 16.0% (253/1,583) | 174 | 65 | **14** | 2.03% (14/689) | 94.5% |
| 99% | 0.032 | 98.8% (681/689) | 9.0% (142/1,583) | 99 | 35 | **8** | 1.16% (8/689) | 94.4% |
| 99.5% | 0.014 | 99.9% (688/689) | 1.5% (23/1,583) | 17 | 5 | **1** | 0.15% (1/689) | 95.7% |
| 99.9% | 0.011 | 100.0% (689/689) | 0.7% (11/1,583) | 9 | 2 | **0** | 0.00% (0/689) | 100.0% |

*NPV = (normal+near TN)/Auto-Cleared; Miss rate = abnormal(FN)/Pos; specificity (TN/Neg) = 39.4% / 26.7% / 15.0% / 2.5% / 1.2%.*

**Reading:** at the **99.5% target** point the AI auto-clears **1.5%** of the worklist (23/1,583) at **95.7%** negative precision and a **0.15%** miss rate (achieved sensitivity 99.9%). Operate high; recalibrate on recent production data before quoting any number.

---

## 4. Caveats

| Caveat | Note |
|---|---|
| **Snapshot** | v5 is still training; this is the best checkpoint at ep8. Re-run when training finishes. |
| **Threshold transfer** | Val thresholds may undershoot their sensitivity target under distribution shift — recalibrate on recent production data. |
| **Prevalence** | NPV depends on the test prevalence (normal 21.2% here); re-measure on the live mix. |
| **Label quality** | `classification` labels auto-derived from reports, not adjudicated. |
| **Regulatory** | Autonomous rule-out is a clinical decision — needs prospective validation + sign-off. |

---
*Operating points from `eval_autorule_abnormal.py` over the dumped `series_probs_{val,test}.csv` (per-study, mean aggregation), v5 ep8 snapshot. Companion: `v5_production_deployment_pos(near-normal_abnormal)_heldout.md`.*