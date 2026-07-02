# v3 Production Deployment — AI Auto-Rule-Out (ABNORMAL objective) (TEST: curated GOLDEN SET)

> **Model:** ct_brain v3 (`runs/maxvit384_3class_clinical_v3`, best.pt = ep3), MaxViT-MIL, 3-class.
> **Date:** 2026-07-02
> **Test set:** the **curated golden set** (`disk_vdc/ct_brain_orig_csv/ct_brain_golden_set_storage.csv`), DICOM under `disk_vdc/ct_brain_dicom` (folder = `study_path` with `/`→`_`). **945 of 989** golden studies were locally available and scored (44 missing folders / no series ≥10 slices). Case mix is **abnormal-heavy**: abnormal **60.6%**, normal only **5.0%**.
> **Objective (this report):** **positive = abnormal**, **negative = normal + near_normal**. The AI escalates only studies that look **abnormal**; everything confidently non-abnormal is auto-cleared. Companion: `v3_production_deployment_pos(near-normal_abnormal)_golden_set.md`.
> **Scoring unit:** per-study, **mean** aggregation over the study's series (all series ≥10 slices, *not* pruned to a primary axial series). Thresholds **transferred from the original val** (no recalibration).

---

## 1. Core principle — escalate abnormal, auto-clear the rest

The AI does **one** autonomous thing: **auto-clear a confidently non-abnormal study** (normal *or* near_normal) off the worklist. It never diagnoses pathology.

| | |
|---|---|
| **Positive (Pos)** | abnormal |
| **Negative (Neg)** | non-abnormal = normal + near_normal |
| **Decision score** | `s = P(abnormal)` (per study) |
| **Auto-clear if** | `s < threshold` → skip doctor (normal or near_normal) |
| **Otherwise** | → radiologist, exactly as today |
| **Threshold set by** | max tolerable **abnormal-miss rate** (safety budget), *not* workload saved |
| **Only dangerous error** | an **abnormal study auto-cleared** (FN). A cleared near_normal is *not* a miss here — it's an accepted outcome. |

> ⚠️ **More aggressive promise than the not-normal report.** It auto-reports near_normal studies (which *do* have minor findings). Whether that's acceptable is a clinical/regulatory call — see §4.

---

## 2. Real-time production flow

```
   CT scanner / PACS ──DICOM──> 1.Ingest ─> 2.Preprocess (HU->3 windows, 96 slices@384px)
   ─> 3.QC fail-safe (unreadable/atypical -> doctor) ─> 4.Inference v3 (P(normal,near,abn))
   ─> 5.Study aggregate (mean over series) -> s=P(abnormal) ─> 6.Decision: s < threshold ?
        YES -> 7a.AUTO-CLEAR non-abnormal (skip doctor, auto-draft, X% audit)
        NO  -> 7b.FLAG -> radiologist (prioritized by P(abnormal))
   ─> 8.Monitoring (auto-clear rate, audited abnormal-miss rate, drift, prevalence; auto-revert)
```

**Safety nets:** QC fail-safe → doctor; random audit of auto-cleared studies to measure the *real* abnormal-miss rate; prevalence/drift monitoring with auto-revert; human override always wins.

---

## 2b. Test-set composition — golden set

Scored **per study** → denominators are studies. Abnormal prevalence = **60.6%**, normal only **5.0%**.

| class | series | studies | role here |
|---|--:|--:|---|
| normal | 69 | 47 | negative (auto-clearable) |
| near_normal | 541 | 325 | negative (auto-clearable) |
| abnormal | 907 | 573 | **positive — must escalate** |
| **non-abnormal (normal + near_normal)** | **610** | **372** | **Negative (Neg)** |
| **total** | **1,517** | **945** | |

**Why this framing** — the must-not-miss class (abnormal) sits on the model's *cleanest* boundary:

| boundary | AUC (golden set) | (orig. held-out split) |
|---|--:|--:|
| abnormal vs normal only — cleanest, what this objective leans on | **0.926** | 0.95 |
| not-normal vs normal — the other report's objective | 0.861 | 0.90 |
| **abnormal vs (normal + near_normal)** — *this objective* | 0.773 | 0.88 |
| near_normal vs normal — weak boundary, dropped from the safety target | 0.699 | 0.75 |
| 3-class macro (OvR) | 0.773 | 0.85 |

AUCs are lower than the original held-out split — expected for a fresh, harder curated set with auto-derived labels — but the must-not-miss abnormal-vs-normal boundary still holds at **0.93**.

---

## 3. Choosing the threshold = choosing a safety budget

Higher threshold = safer (fewer missed abnormals) but less automation. **Thresholds set on the original val, applied to the golden set.** **Auto-Cleared (PN)** = predicted-negative = every study that leaves the worklist; it contains the correct clears (normal + near_normal TN) **and** the missed abnormals (FN).

| Target Sens (val) | Threshold | Sensitivity achieved (TP/Pos) | Auto-Cleared (PN) | ↳ normal (TN ✓) | ↳ near_normal (TN ✓) | ↳ abnormal (FN ✗) | Miss rate (FN/Pos) | Negative Precision |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 95% | 0.025 | 90.6% (519/573) | 18.3% (173/945) | 33 | 86 | **54** | 9.42% (54/573) | 68.8% |
| 98% | 0.012 | 95.3% (546/573) | 10.8% (102/945) | 26 | 49 | **27** | 4.71% (27/573) | 73.5% |
| 99% | 0.004 | 97.0% (556/573) | 6.2% (59/945) | 19 | 23 | **17** | 2.97% (17/573) | 71.2% |
| 99.5% | 0.003 | 98.4% (564/573) | 4.4% (42/945) | 16 | 17 | **9** | 1.57% (9/573) | 78.6% |
| 99.9% | 0.003 | 98.4% (564/573) | 3.9% (37/945) | 13 | 15 | **9** | 1.57% (9/573) | 75.7% |

*Columns: **Auto-Cleared (PN)** = (TN + FN)/945 = normal(TN) + near_normal(TN) + abnormal(FN); only the **abnormal(FN)** sub-column is a miss. **Negative Precision** (NPV) = (normal + near_normal TN) / Auto-Cleared. **Miss rate** = abnormal(FN)/573. **specificity** = TN/372 = % of clearable studies automated (32.0% / 20.2% / 11.3% / 8.9% / 7.5%). Pos = 573 abnormal, Neg = 372 non-abnormal.*

**Findings on the golden set:**

- **Threshold transfer undershoots** (as on the production week): the val threshold for 95% abnormal-sensitivity delivers **90.6%** here; the 99% target delivers 97.0%. **Recalibrate on recent production data.**
- **NPV holds at 69–79%** (non-abnormal) despite the hard set — because abnormal prevalence is high (61%), of everything cleared ~69–79% is genuinely non-abnormal. But a large share of clears are **near_normal** (auto-reported despite minor findings): 86 of 119 correct clears (≈72%) at the 95% point.
- **Miss floor.** The lowest usable threshold still auto-clears **9 abnormal studies** (mean `P(abnormal)` below any threshold) → miss floors at ~1.57% (9/573); hard, confidently-mis-scored abnormals (label-review / hard-negative candidates).

**Reading:** at the **99.5% / 1.57%-miss** point the AI clears ~4.4% of the worklist (42/945) at 79% non-abnormal precision (specificity 8.9%); the **95%** point clears 18.3% but misses 9.4% of abnormals — too high for autonomous rule-out. On this abnormal-heavy set the automation headroom is small.

---

## 4. Caveats & how to automate *more* safely

| Caveat | What it means | Action |
|---|---|---|
| ⚠️ **Threshold transfer** | Val thresholds undershoot their sensitivity target here (90.6% vs 95%). | **Re-fit thresholds on recent production data**; re-check continuously. |
| ⚠️ **Prevalence** | NPV is on the abnormal-heavy golden set (61% abnormal, 5% normal) — not a live population. | Re-measure NPV on the true production distribution before quoting any number. |
| **near_normal auto-reported** | This objective clears near_normal (minor findings) — ≈72% of correct clears at the 95% point. | Get clinical sign-off on auto-reporting near_normal. |
| **~9-case miss floor** | Miss can't go below ~1.57% — 9 abnormals are confidently mis-scored. | Pull those for label review + hard-negative mining. |
| **No primary-series pruning** | All series ≥10 slices were scored and mean-aggregated (unlike the pruned June-week set). | A series-selector (primary axial only) may lift accuracy; re-score if available. |
| **Coverage / decode** | 945/989 golden studies scored; 1 study zero-placeholdered (23 unreadable slices). | Keep JPEG decoders in the image; QC fail-safe routes undecodable studies to a doctor. |
| **Label quality** | `classification` labels auto-derived from reports. | Spot-adjudicate; treat near_normal counts as soft. |
| **v3 ceiling** | abn-vs-rest AUC 0.77; abn-vs-normal 0.93 on this set. | Lift AUC: rule-out multi-task head (v5), near_normal label cleaning, conformal selective prediction. |
| **Regulatory** | Autonomous rule-out is a clinical decision. | Scoping analysis only — needs prospective validation + sign-off. |

---
*Operating points from cascade-free rule-out over `runs/maxvit384_3class_clinical_v3/eval_golden_set/series_probs_test.csv` (per-study, mean). Thresholds set on the original val, measured on the golden set. Companion: `v3_production_deployment_pos(near-normal_abnormal)_golden_set.md`.*
