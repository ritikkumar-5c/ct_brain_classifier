# v3 Production Deployment — AI Auto-Rule-Out of Normal CT-Brain Studies (TEST: curated GOLDEN SET)

> **Model:** ct_brain v3 (`runs/maxvit384_3class_clinical_v3`, best.pt = ep3), MaxViT-MIL, 3-class.
> **Date:** 2026-07-02
> **Test set:** the **curated golden set** (`disk_vdc/ct_brain_orig_csv/ct_brain_golden_set_storage.csv`), DICOM under `disk_vdc/ct_brain_dicom` (folder = `study_path` with `/`→`_`). **945 of 989** golden studies were locally available and scored. Case mix: **normal prevalence only 5.0%**, abnormal 60.6%.
> **Goal:** use the AI to **safely auto-clear confidently-normal studies** so radiologists only read the rest — without missing pathology.
> **Scoring unit:** per-study, **mean** aggregation over the study's series (all series ≥10 slices, *not* pruned to a primary axial series). Thresholds **transferred from the original val** (no recalibration). Companion: `v3_production_deployment_pos(abnormal)_golden_set.md`.

---

## 1. Core principle — rule-out only, never auto-diagnose

The AI does **one** autonomous thing: **auto-clear a confidently-normal study** off the worklist. It never diagnoses pathology.

| | |
|---|---|
| **Positive (Pos)** | not-normal = near_normal + abnormal |
| **Negative (Neg)** | normal |
| **Decision score** | `s = 1 − P(normal)` (per study) |
| **Auto-clear if** | `s < threshold` → skip doctor |
| **Otherwise** | → radiologist, exactly as today |
| **Threshold set by** | max tolerable **miss rate** (safety budget), *not* workload saved |
| **Only dangerous error** | a **pathology auto-cleared as normal** (false-negative leaving the queue) |

---

## 2. Real-time production flow

```
   CT scanner / PACS ──DICOM──> 1.Ingest ─> 2.Preprocess (HU->3 windows, 96 slices@384px)
   ─> 3.QC fail-safe (unreadable/atypical -> doctor) ─> 4.Inference v3 (P(normal,near,abn))
   ─> 5.Study aggregate (mean over series) -> s=1-P(normal) ─> 6.Decision: s < threshold ?
        YES -> 7a.AUTO-CLEAR normal (skip doctor, auto-draft normal report, X% audit)
        NO  -> 7b.FLAG -> radiologist (prioritized by score)
   ─> 8.Monitoring (auto-clear rate, audited miss rate, drift, prevalence; auto-revert)
```

**Safety nets:** QC fail-safe → doctor; random audit of auto-cleared studies to measure the *real* miss rate; prevalence/drift monitoring with auto-revert; human override always wins.

---

## 2b. Test-set composition — golden set

Scored **per study** → denominators are studies. **Normal prevalence is only 5.0%.**

| class | series | studies |
|---|--:|--:|
| normal | 69 | 47 |
| near_normal | 541 | 325 |
| abnormal | 907 | 573 |
| **not-normal (near_normal + abnormal)** | **1,448** | **898** |
| **total** | **1,517** | **945** |

Discrimination on this set (per-study, mean aggregation):

| boundary | AUC (golden set) | (orig. held-out split) |
|---|--:|--:|
| not-normal vs normal — *this objective* | **0.861** | 0.90 |
| abnormal vs normal only | 0.926 | 0.95 |
| abnormal vs (normal + near_normal) | 0.773 | 0.88 |
| near_normal vs normal (weak boundary) | 0.699 | 0.75 |
| 3-class macro (OvR) | 0.773 | 0.85 |

---

## 3. Choosing the threshold = choosing a safety budget

Higher threshold = safer (fewer misses) but less automation. **Thresholds set on the original val, applied to the golden set.** **Auto-Cleared (PN)** = predicted-negative = every study that leaves the worklist; it contains the correct clears (normal TN) **and** the misses (near_normal + abnormal FN).

| Target Sens (val) | Threshold | Sensitivity achieved (TP/Pos) | Auto-Cleared (PN) | ↳ normal (TN ✓) | ↳ near_normal (FN ✗) | ↳ abnormal (FN ✗) | Miss rate (FN/Pos) | Negative Precision |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 95% | 0.208 | 92.1% (827/898) | 10.2% (96/945) | 25 | **43** | **28** | 7.91% (71/898) | 26.0% |
| 98% | 0.066 | 96.0% (862/898) | 5.8% (55/945) | 19 | **22** | **14** | 4.01% (36/898) | 34.5% |
| 99% | 0.038 | 97.3% (874/898) | 4.1% (39/945) | 15 | **14** | **10** | 2.67% (24/898) | 38.5% |
| 99.5% | 0.023 | 98.8% (887/898) | 2.0% (19/945) | 8 | **7** | **4** | 1.22% (11/898) | 42.1% |
| 99.9% | 0.018 | 99.1% (890/898) | 1.6% (15/945) | 7 | **6** | **2** | 0.89% (8/898) | 46.7% |
| 100% | 0.016 | 99.2% (891/898) | 1.4% (13/945) | 6 | **5** | **2** | 0.78% (7/898) | 46.2% |

*Columns: **Auto-Cleared (PN)** = (TN + FN)/945 = normal(TN) + near_normal(FN) + abnormal(FN); only the **normal(TN)** sub-column is a correct clear. **Negative Precision** (NPV) = normal(TN) / Auto-Cleared. **Miss rate** = (near + abn FN)/898. Pos = 898 not-normal, Neg = 47 normal.*

**Two production-critical findings on the golden set:**

1. **Threshold transfer undershoots.** Val thresholds that targeted 95%/99% deliver **92.1%/97.3%** here — the score distribution shifted. **Recalibrate on recent production data.**
2. **NPV collapses at low normal prevalence.** Only 5.0% of studies are truly normal, so of everything auto-cleared just **26–47%** is genuinely normal (vs 70–86% on the enriched split). Aggressive rule-out is **not safe** here — even operating high (99.9–100%), the AI clears only ~1.4–1.6% of the worklist and is still auto-clearing more pathology than normals (NPV 46%).

**Reading:** on this abnormal-heavy set the not-normal rule-out headroom is very small — a direct consequence of (a) 5% normal prevalence and (b) v3's modest not-normal-vs-normal AUC (0.86). Treat as a calibration/monitoring baseline, not a green light. (The **abnormal** objective is materially more usable on this set — see companion report.)

---

## 4. Caveats & how to automate *more* safely

| Caveat | What it means | Action |
|---|---|---|
| ⚠️ **Threshold transfer** | Val thresholds miss their sensitivity target here (92% vs 95%). | **Re-fit thresholds on recent production data**; re-check continuously. |
| ⚠️ **Prevalence** | NPV is at the golden set's 5.0%-normal mix and is low (26–47%). | Re-measure NPV on the live distribution before quoting any number. |
| **No primary-series pruning** | All series ≥10 slices scored and mean-aggregated (unlike the pruned June-week set). | A primary-axial series selector may lift accuracy; re-score if available. |
| **Coverage / decode** | 945/989 golden studies scored; 1 zero-placeholdered (23 unreadable slices). | Keep JPEG decoders in the image; QC fail-safe routes undecodable studies to a doctor. |
| **Label quality** | `classification` labels auto-derived from reports; near_normal is noisy. | Spot-adjudicate a sample; treat near_normal miss counts as soft. |
| **v3 ceiling** | macro AUC 0.77; not-normal-vs-normal 0.86 on this set. | Lift AUC: 2-class reframe, near_normal label cleaning, ensemble + TTA, conformal selective prediction. |
| **Regulatory** | Autonomous rule-out is a clinical decision. | Scoping analysis only — needs prospective validation + sign-off. |

---
*Operating points from cascade-free rule-out over `runs/maxvit384_3class_clinical_v3/eval_golden_set/series_probs_test.csv` (per-study, mean). Thresholds set on the original val, measured on the golden set. Companion: `v3_production_deployment_pos(abnormal)_golden_set.md` (abnormal objective).*
