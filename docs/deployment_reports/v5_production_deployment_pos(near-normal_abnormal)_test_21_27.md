# v5 Production Deployment — AI Auto-Rule-Out of Normal CT-Brain Studies (TEST: June 21–27 production week)

> **Model:** ct_brain v5 (`runs/maxvit384_3class_clinical_v5`, best snapshot = ep8; **training still ongoing**), MaxViT-MIL, 3-class (+ rule-out multi-task head).
> **Date:** 2026-07-01
> **Test set:** a real **consecutive-week production sample** reported 21–27 June 2026 (`disk_vdc/test_data/csvs/test_june_21_27.csv`), pruned to the primary axial series; reflects the live case mix.
> **Objective:** positive = **not-normal (near_normal + abnormal)**, negative = **normal**. Decision score `s = 1 − P(normal)` (per study, mean over series). Thresholds set on the original validation set, measured on this test set.
> **Companion:** `v5_production_deployment_pos(abnormal)_test_21_27.md`

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

## 2b. Test-set composition — June 21–27 production week

Scored **per study** (mean over series) → denominators are studies.

| class | series | studies |
|---|--:|--:|
| normal | 508 | 327 |
| near_normal | 5,791 | 3,614 |
| abnormal | 4,346 | 2,741 |
| **total** | **10,645** | **6,682** |

Prevalence: normal **4.9%**, not-normal **95.1%**, abnormal **41.0%**. Pos = 6,355 not-normal, Neg = 327.

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

| Target Sens (val) | Threshold | Sensitivity achieved (TP/Pos) | Auto-Cleared (PN) | ↳ normal (TN ✓) | ↳ near_normal (FN ✗) | ↳ abnormal (FN ✗) | Miss rate (FN/Pos) | Neg Precision |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 95% | 0.361 | 89.9% (5,712/6,355) | 12.6% (842/6,682) | 199 | **546** | **97** | 10.12% (643/6,355) | 23.6% |
| 98% | 0.174 | 95.6% (6,077/6,355) | 6.0% (400/6,682) | 122 | **231** | **47** | 4.37% (278/6,355) | 30.5% |
| 99% | 0.121 | 97.8% (6,218/6,355) | 3.2% (212/6,682) | 75 | **110** | **27** | 2.16% (137/6,355) | 35.4% |
| 99.5% | 0.075 | 99.5% (6,321/6,355) | 1.0% (65/6,682) | 31 | **26** | **8** | 0.54% (34/6,355) | 47.7% |
| 99.9% | 0.059 | 99.8% (6,345/6,355) | 0.4% (24/6,682) | 14 | **8** | **2** | 0.16% (10/6,355) | 58.3% |
| 100% | 0.059 | 99.8% (6,345/6,355) | 0.4% (24/6,682) | 14 | **8** | **2** | 0.16% (10/6,355) | 58.3% |

*NPV = normal(TN)/Auto-Cleared; Miss rate = (near+abn FN)/Pos.*

**Reading:** at the **99.5% target** point the AI auto-clears **1.0%** of the worklist (65/6,682) at **47.7%** negative precision and a **0.54%** miss rate (achieved sensitivity 99.5%). Operate high; recalibrate on recent production data before quoting any number.

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
*Operating points from `eval_autorule.py` over the dumped `series_probs_{val,test}.csv` (per-study, mean aggregation), v5 ep8 snapshot. Companion: `v5_production_deployment_pos(abnormal)_test_21_27.md`.*