# v3 Production Deployment — AI Auto-Rule-Out (ABNORMAL objective)

> **Model:** ct_brain v3 (`runs/maxvit384_3class_clinical_v3`, best.pt = ep3), MaxViT-MIL, 3-class.
> **Date:** 2026-06-30
> **Objective (this report):** **positive = abnormal**, **negative = normal + near_normal**. The AI escalates only studies that look **abnormal**; everything confidently non-abnormal (normal *and* near_normal) is auto-cleared. This is the companion to `v3_production_deployment_pos(near-normal_abnormal).md`, which uses **positive = not-normal (near_normal + abnormal)**; the two are compared in `v3_objective_comparison.md`.
> **Why this framing:** v3's dangerous-miss class (abnormal) sits on the model's *cleanest* decision boundary (abnormal-vs-normal AUC **0.95**), while the normal↔near_normal boundary is the weak one (AUC 0.75). Defining the safety target as "never auto-clear an **abnormal** study" — and treating near_normal as clearable — aligns the deployment with what the model is actually good at.
> **Scoring unit:** per-patient, **mean** aggregation over the patient's series.

---

## 1. Core principle — escalate abnormal, auto-clear the rest

The AI does **one** autonomous thing: **auto-clear a confidently non-abnormal study** (normal *or* near_normal) off the worklist. It never diagnoses pathology.

| | |
|---|---|
| **Positive (Pos)** | abnormal |
| **Negative (Neg)** | non-abnormal = normal + near_normal |
| **Decision score** | `s = P(abnormal)` (per patient) |
| **Auto-clear if** | `s < threshold` → skip doctor (normal or near_normal) |
| **Otherwise** | → radiologist, exactly as today |
| **Threshold set by** | max tolerable **abnormal-miss rate** (safety budget), *not* workload saved |
| **Only dangerous error** | an **abnormal study auto-cleared** (FN). A cleared near_normal is *not* a miss here — it's an accepted outcome. |

> ⚠️ **More aggressive promise than the not-normal report.** It auto-reports near_normal studies (which *do* have minor findings). Whether that's acceptable is a clinical/regulatory call — see §4 (Caveats) and `v3_objective_comparison.md`. The numbers look better than the not-normal report partly because near_normal misses no longer count, partly because more volume is cleared.

---

## 2. Real-time production flow

```
            ┌─────────────────────────────────────────────────────────────┐
            │                     CT scanner / PACS                         │
            └───────────────────────────┬─────────────────────────────────┘
                                         │ DICOM series (study)
                                         ▼
                              ┌──────────────────────┐
                              │ 1. Ingest / router    │  (DICOM C-STORE → AI service queue)
                              └──────────┬───────────┘
                                         ▼
                              ┌──────────────────────┐
                              │ 2. Preprocess         │  HU → 3 windows (brain/subdural/bone),
                              │                       │  96 slices/series @384px
                              └──────────┬───────────┘
                                         ▼
                              ┌──────────────────────┐
                              │ 3. QC / fail-safe     │  unreadable / atypical / too-few-slices
                              │                       │  ───────────► send to doctor (no AI call)
                              └──────────┬───────────┘
                                         ▼
                              ┌──────────────────────┐
                              │ 4. Inference (v3)     │  per-series P(normal,near,abn)
                              └──────────┬───────────┘
                                         ▼
                              ┌──────────────────────┐
                              │ 5. Patient aggregate  │  mean over series → P(abnormal)
                              │                       │  s = P(abnormal)
                              └──────────┬───────────┘
                                         ▼
                        ┌────────────────┴─────────────────┐
                        │  6. Decision:  s < threshold ?    │
                        └───────┬───────────────────┬───────┘
                       YES ◄────┘                   └────► NO
                        │                                  │
                        ▼                                  ▼
          ┌──────────────────────────┐        ┌──────────────────────────┐
          │ 7a. AUTO-CLEAR non-abn    │        │ 7b. FLAG → radiologist     │
          │  • skip doctor read       │        │  • worklist (prioritize by │
          │  • auto-draft report      │        │    P(abnormal))            │
          │    (normal OR near_normal)│        │  • doctor reads as today   │
          │  • X% random AUDIT to doc │        │                            │
          └────────────┬─────────────┘        └────────────┬─────────────┘
                       │                                    │
                       └──────────────┬─────────────────────┘
                                      ▼
                       ┌──────────────────────────────────┐
                       │ 8. Monitoring & feedback loop      │
                       │  • auto-clear rate, audited        │
                       │    ABNORMAL-miss rate              │
                       │  • score drift, input prevalence   │
                       │  • all doctor overrides logged     │
                       └──────────────────────────────────┘
```

**Safety nets (identical to the not-normal flow):** QC fail-safe → doctor; random audit of auto-cleared studies to measure the *real* abnormal-miss rate; prevalence/drift monitoring with auto-revert to "all-to-doctor"; human override always wins.

---

## 2b. Test-set composition

Held-out test split (patient-grouped, never seen in training/selection). Scored **per patient** → denominators are patients. Abnormal prevalence = **43.5%** (enriched — not a deployment population; see §4).

| class | series | patients | role here |
|---|--:|--:|---|
| normal | 546 | 336 | negative (auto-clearable) |
| near_normal | 942 | 558 | negative (auto-clearable) |
| abnormal | 1,135 | 689 | **positive — must escalate** |
| **non-abnormal (normal + near_normal)** | **1,488** | **894** | **Negative (Neg)** |
| **total** | **2,623** | **1,583** | |

**Why this framing** — the must-not-miss class (abnormal) sits on the model's *cleanest* boundary; near_normal errors are removed from the safety target:

| boundary | AUC |
|---|--:|
| abnormal vs normal only — cleanest, what this objective leans on | **0.95** |
| not-normal vs normal — the other report's objective | 0.90 |
| **abnormal vs (normal + near_normal)** — *this objective* | 0.88 |
| near_normal vs normal — weak boundary, dropped from the safety target | 0.75 |

This objective's headline AUC (0.88) is *slightly lower* than the not-normal objective's (0.90) — near_normal blurs the negative pool — but the only class you must not miss rides the 0.95 boundary.

---

## 3. Choosing the threshold = choosing a safety budget

Higher threshold = safer (fewer missed abnormals) but less automation. Threshold set on val, measured on test. **Auto-Cleared (PN)** = predicted-negative = every study that leaves the worklist; it contains the correct clears (normal + near_normal TN) **and** the missed abnormals (FN) — so the misses sit *inside* this column, not beside it.

| Target Sens | Threshold | Sensitivity (TP/Pos) | Auto-Cleared (PN) | ↳ normal (TN ✓) | ↳ near_normal (TN ✓) | ↳ abnormal (FN ✗ = missed) | Miss rate (FN/Pos) | Negative Precision |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 95% | 0.025 | 94.9% (654/689) | 26.5% (420/1,583) | 234 | 151 | **35** | 5.1% (35/689) | 91.7% |
| 98% | 0.012 | 98.0% (675/689) | 16.6% (262/1,583) | 174 | 74 | **14** | 2.0% (14/689) | 94.7% |
| 99% | 0.004 | 99.1% (683/689) | 8.3% (132/1,583) | 100 | 26 | **6** | 0.87% (6/689) | 95.5% |
| 99.5% | 0.003 | 99.4% (685/689) | 6.4% (101/1,583) | 78 | 19 | **4** | 0.58% (4/689) | 96.0% |
| 99.9% | 0.003 | 99.4% (685/689) | 5.7% (91/1,583) | 71 | 16 | **4** | 0.58% (4/689) | 95.6% |

*Columns: **Auto-Cleared (PN)** = (TN + FN)/1,583 = normal(TN) + near_normal(TN) + abnormal(FN); only the **abnormal(FN)** sub-column is a miss. **Negative Precision** (NPV) = (normal + near_normal TN) / Auto-Cleared = of everything cleared, the % genuinely non-abnormal (safe). **Miss rate** = abnormal(FN)/689. **specificity** = TN/894 = % of clearable studies automated (43.1% / 27.7% / 14.1% / 10.9% / 9.7%). Pos = 689 abnormal, Neg = 894 non-abnormal. Enriched 43.5%-abnormal test set — NPV rises at realistic abnormal-light prevalence (see §4 caveats).*

**Irreducible miss floor.** The minimum usable threshold (0.003) still auto-clears **4 abnormal patients** (their mean `P(abnormal)` sits below any threshold), so miss **floors at 0.58% (4/689)** — it can't reach the near-zero values the not-normal objective could. These 4 are hard, confidently-mis-scored abnormals (label-review / hard-negative candidates).

**Reading:** at the **99.5% / 0.58%-miss** point the AI clears ~6.4% of the worklist at 96% negative precision; the **95%** point clears 26.5% but misses 5.1% of abnormals — too high for autonomous rule-out. Note a large share of the clears are **near_normal** (auto-reported despite minor findings): 151 of 385 correct clears (≈39%) at 95%, shrinking to ≈20% at 99.5% — the main thing a clinical reviewer must sign off on (see §4).

---

## 4. Caveats & how to automate *more* safely

| Caveat | What it means | Action |
|---|---|---|
| ⚠️ **Prevalence** | NPV is on the enriched 43.5%-abnormal test set (not a real population) and shifts with prevalence. | Re-measure NPV on the live distribution before quoting any number. |
| **near_normal auto-reported** | This objective clears near_normal (minor findings) as well as normals — ≈39% of clears at the 95% point. | Get clinical sign-off on auto-reporting near_normal before counting the higher volume as a win. |
| **4-case miss floor** | Miss can't go below 0.58% — 4 abnormals are confidently mis-scored. | Pull those 4 for label review + hard-negative mining. |
| **Calibration** | Threshold is dataset-specific. | Re-fit on a held-out slice of *production* data before go-live; re-check periodically. |
| **v3 ceiling** | Safe volume capped by v3's discrimination (abn-vs-rest AUC 0.88; abn-vs-normal 0.95). | Lift AUC: rule-out multi-task head + top-k pooling (v5), near_normal label cleaning, enrich normal calibration set, conformal selective prediction (see `run_comparison_v1_v2_v3.md` §4, `ct-brain-autoreport-strategy`). |
| **Regulatory** | Autonomous rule-out is a clinical decision. | Scoping analysis only — needs prospective validation + sign-off. |

---
*Operating points and projections from `eval_autorule_abnormal.py` over the dumped `series_probs_{val,test}.csv` (patient-level, mean aggregation). No leakage — thresholds set on val, measured on test. Companion: `v3_production_deployment_pos(near-normal_abnormal).md` (not-normal objective); objective comparison in `v3_objective_comparison.md`.*
