# v5 Production Deployment — AI Auto-Rule-Out (ABNORMAL objective) (TEST: June 21–27 production week)

> **Model:** ct_brain v5 (`runs/maxvit384_3class_clinical_v5`, best snapshot = ep8; **training still ongoing**), MaxViT-MIL, 3-class (+ rule-out multi-task head).
> **Date:** 2026-07-01
> **Test set:** a real **consecutive-week production sample** reported 21–27 June 2026 (`disk_vdc/test_data/csvs/test_june_21_27.csv`), pruned to the primary axial series; reflects the live case mix.
> **Objective:** positive = **abnormal**, negative = **non-abnormal (normal + near_normal)**. Decision score `s = P(abnormal)` (per study, mean over series). Thresholds set on the original validation set, measured on this test set.
> **Companion:** `v5_production_deployment_pos(near-normal_abnormal)_test_21_27.md`

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

## 2b. Test-set composition — June 21–27 production week

Scored **per study** (mean over series) → denominators are studies.

| class | series | studies |
|---|--:|--:|
| normal | 508 | 327 |
| near_normal | 5,791 | 3,614 |
| abnormal | 4,346 | 2,741 |
| **total** | **10,645** | **6,682** |

Prevalence: normal **4.9%**, not-normal **95.1%**, abnormal **41.0%**. Pos = 2,741 abnormal, Neg = 3,941.

Discrimination on this set (per-study, mean aggregation):

| boundary | AUC |
|---|--:|
| not-normal vs normal | 0.881 |
| abnormal vs normal only | 0.926 |
| abnormal vs (normal + near_normal) | 0.813 |
| near_normal vs normal | 0.760 |
| 3-class macro (OvR) | 0.809 |

---

## 3. Choosing the threshold = choosing a safety budget

Thresholds set on the original validation set, applied to this test set. **Auto-Cleared (PN)** = predicted-negative (leaves the worklist); it contains the correct clears **and** the misses.

| Target Sens (val) | Threshold | Sensitivity achieved (TP/Pos) | Auto-Cleared (PN) | ↳ normal (TN ✓) | ↳ near_normal (TN ✓) | ↳ abnormal (FN ✗) | Miss rate (FN/Pos) | Neg Precision |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 95% | 0.070 | 93.4% (2,561/2,741) | 21.5% (1,438/6,682) | 233 | 1,025 | **180** | 6.57% (180/2,741) | 87.5% |
| 98% | 0.048 | 96.7% (2,650/2,741) | 13.3% (888/6,682) | 179 | 618 | **91** | 3.32% (91/2,741) | 89.8% |
| 99% | 0.032 | 98.6% (2,702/2,741) | 7.5% (501/6,682) | 116 | 346 | **39** | 1.42% (39/2,741) | 92.2% |
| 99.5% | 0.014 | 99.9% (2,737/2,741) | 1.1% (72/6,682) | 25 | 43 | **4** | 0.15% (4/2,741) | 94.4% |
| 99.9% | 0.011 | 100.0% (2,740/2,741) | 0.2% (14/6,682) | 4 | 9 | **1** | 0.04% (1/2,741) | 92.9% |

*NPV = (normal+near TN)/Auto-Cleared; Miss rate = abnormal(FN)/Pos; specificity (TN/Neg) = 31.9% / 20.2% / 11.7% / 1.7% / 0.3%.*

**Reading:** at the **99.5% target** point the AI auto-clears **1.1%** of the worklist (72/6,682) at **94.4%** negative precision and a **0.15%** miss rate (achieved sensitivity 99.9%). Operate high; recalibrate on recent production data before quoting any number.

---

## 4. Caveats

| Caveat | Note |
|---|---|
| **Snapshot** | v5 is still training; this is the best checkpoint at ep8. Re-run when training finishes. |
| **Threshold transfer** | Val thresholds may undershoot their sensitivity target under distribution shift — recalibrate on recent production data. |
| **Prevalence** | NPV depends on the test prevalence (normal 4.9% here); re-measure on the live mix. |
| **Label quality** | `classification` labels auto-derived from reports, not adjudicated. |
| **Regulatory** | Autonomous rule-out is a clinical decision — needs prospective validation + sign-off. |

---
*Operating points from `eval_autorule_abnormal.py` over the dumped `series_probs_{val,test}.csv` (per-study, mean aggregation), v5 ep8 snapshot. Companion: `v5_production_deployment_pos(near-normal_abnormal)_test_21_27.md`.*