# v5 Production Deployment — AI Auto-Rule-Out of Normal CT-Brain Studies (TEST: held-out split)

> **Model:** ct_brain v5 (`runs/maxvit384_3class_clinical_v5`, best snapshot = ep8; **training still ongoing**), MaxViT-MIL, 3-class (+ rule-out multi-task head).
> **Date:** 2026-07-01
> **Test set:** the original patient-grouped **held-out** test split (`train_data/csvs/splits/test.csv`) — enriched, not a deployment population.
> **Objective:** positive = **not-normal (near_normal + abnormal)**, negative = **normal**. Decision score `s = 1 − P(normal)` (per study, mean over series). Thresholds set on the original validation set, measured on this test set.
> **Companion:** `v5_production_deployment_pos(abnormal)_heldout.md`

---

## 1. Core principle — rule-out only, never auto-diagnose

The AI does **one** autonomous thing: **auto-clear a confidently-negative study** off the worklist. It never diagnoses pathology.

| | |
|---|---|
| **Positive (Pos)** | not-normal (near_normal + abnormal) |
| **Negative (Neg)** | normal |
| **Decision score** | `s = 1 − P(normal)` (per study) |
| **Auto-clear if** | `s < threshold` → skip doctor |
| **Threshold set by** | max tolerable **miss rate** (safety budget) |
| **Only dangerous error** | a **pathology auto-cleared as normal** |

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

Prevalence: normal **21.2%**, not-normal **78.8%**, abnormal **43.5%**. Pos = 1,247 not-normal, Neg = 336.

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

| Target Sens (val) | Threshold | Sensitivity achieved (TP/Pos) | Auto-Cleared (PN) | ↳ normal (TN ✓) | ↳ near_normal (FN ✗) | ↳ abnormal (FN ✗) | Miss rate (FN/Pos) | Neg Precision |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 95% | 0.361 | 93.7% (1,168/1,247) | 16.2% (256/1,583) | 177 | **59** | **20** | 6.34% (79/1,247) | 69.1% |
| 98% | 0.174 | 97.8% (1,220/1,247) | 6.6% (105/1,583) | 78 | **19** | **8** | 2.17% (27/1,247) | 74.3% |
| 99% | 0.121 | 99.0% (1,235/1,247) | 3.8% (60/1,583) | 48 | **7** | **5** | 0.96% (12/1,247) | 80.0% |
| 99.5% | 0.075 | 99.7% (1,243/1,247) | 1.2% (19/1,583) | 15 | **3** | **1** | 0.32% (4/1,247) | 78.9% |
| 99.9% | 0.059 | 99.9% (1,246/1,247) | 0.4% (7/1,583) | 6 | **1** | **0** | 0.08% (1/1,247) | 85.7% |
| 100% | 0.059 | 99.9% (1,246/1,247) | 0.4% (7/1,583) | 6 | **1** | **0** | 0.08% (1/1,247) | 85.7% |

*NPV = normal(TN)/Auto-Cleared; Miss rate = (near+abn FN)/Pos.*

**Reading:** at the **99.5% target** point the AI auto-clears **1.2%** of the worklist (19/1,583) at **78.9%** negative precision and a **0.32%** miss rate (achieved sensitivity 99.7%). Operate high; recalibrate on recent production data before quoting any number.

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
*Operating points from `eval_autorule.py` over the dumped `series_probs_{val,test}.csv` (per-study, mean aggregation), v5 ep8 snapshot. Companion: `v5_production_deployment_pos(abnormal)_heldout.md`.*