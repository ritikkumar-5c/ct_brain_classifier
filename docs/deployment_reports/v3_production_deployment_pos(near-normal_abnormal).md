# v3 Production Deployment — AI Auto-Rule-Out of Normal CT-Brain Studies

> **Model:** ct_brain v3 (`runs/maxvit384_3class_clinical_v3`, best.pt = ep3), MaxViT-MIL, 3-class.
> **Date:** 2026-06-29
> **Goal:** today every study (normal or not) is read by a doctor. Use the AI to **safely auto-clear confidently-normal studies** so radiologists only read the rest — without missing pathology.
> **Scoring unit:** per-patient, **mean** aggregation over the patient's series (best & most robust — see `v3_detailed_report.md` §6).

---

## 1. Core principle — rule-out only, never auto-diagnose

The AI does **one** autonomous thing: **auto-clear a confidently-normal study** off the worklist. It never diagnoses pathology.

| | |
|---|---|
| **Positive (Pos)** | not-normal = near_normal + abnormal |
| **Negative (Neg)** | normal |
| **Decision score** | `s = 1 − P(normal)` (per patient) |
| **Auto-clear if** | `s < threshold` → skip doctor |
| **Otherwise** | → radiologist, exactly as today |
| **Threshold set by** | max tolerable **miss rate** (safety budget), *not* workload saved |
| **Only dangerous error** | a **pathology auto-cleared as normal** (false-negative leaving the queue) |

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
                              │ 5. Patient aggregate  │  mean over series → P(normal)
                              │                       │  s = 1 − P(normal)
                              └──────────┬───────────┘
                                         ▼
                        ┌────────────────┴─────────────────┐
                        │  6. Decision:  s < threshold ?    │
                        └───────┬───────────────────┬───────┘
                       YES ◄────┘                   └────► NO
                        │                                  │
                        ▼                                  ▼
          ┌──────────────────────────┐        ┌──────────────────────────┐
          │ 7a. AUTO-CLEAR normal     │        │ 7b. FLAG → radiologist     │
          │  • skip doctor read       │        │  • worklist (prioritize by │
          │  • auto-draft normal report│       │    score for likely path.) │
          │  • X% random AUDIT to doc │        │  • doctor reads as today   │
          └────────────┬─────────────┘        └────────────┬─────────────┘
                       │                                    │
                       └──────────────┬─────────────────────┘
                                      ▼
                       ┌──────────────────────────────────┐
                       │ 8. Monitoring & feedback loop      │
                       │  • auto-clear rate, audit miss rate│
                       │  • score drift, input prevalence   │
                       │  • all doctor overrides logged     │
                       └──────────────────────────────────┘
```

**Safety nets baked into the flow:**
- **QC fail-safe (step 3):** any study the pipeline can't process confidently → doctor (never silently dropped).
- **Random audit (step 7a):** a fixed % of auto-cleared studies are still read by a radiologist to measure the *real* miss rate continuously.
- **Monitoring (step 8):** alert if input prevalence drifts (the safety math depends on it), if score distribution shifts (model drift), or if audited miss rate exceeds the budget → auto-revert to "all-to-doctor".
- **Human override always wins;** AI output is advisory infrastructure, not a locked decision.

---

## 2b. Test-set composition

Held-out test split (patient-grouped, never seen in training/selection). Scored **per patient** → denominators are patients.

| class | series | patients |
|---|--:|--:|
| normal | 546 | 336 |
| near_normal | 942 | 558 |
| abnormal | 1,135 | 689 |
| **not-normal (near_normal + abnormal)** | **2,077** | **1,247** |
| **total** | **2,623** | **1,583** |

---

## 3. Choosing the threshold = choosing a safety budget

Higher threshold = safer (fewer misses) but less automation. Threshold set on val, measured on test. **Auto-Cleared (PN)** = predicted-negative = every study that leaves the worklist; it contains the correct clears (normal TN) **and** the misses (near_normal + abnormal FN) — so the misses sit *inside* this column, not beside it.

| Target Sens | Threshold | Sensitivity (TP/Pos) | Auto-Cleared (PN) | ↳ normal (TN ✓) | ↳ near_normal (FN ✗ = missed) | ↳ abnormal (FN ✗ = missed) | Miss rate (FN/Pos) | Negative Precision |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 95% | 0.208 | 94.5% (1,178/1,247) | 14.7% (233/1,583) | 164 | **50** | **19** | 5.5% (69/1,247) | 70.4% |
| 98% | 0.066 | 97.8% (1,219/1,247) | 7.9% (125/1,583) | 97 | **18** | **10** | 2.2% (28/1,247) | 77.6% |
| 99% | 0.038 | 98.6% (1,230/1,247) | 5.2% (83/1,583) | 66 | **10** | **7** | 1.4% (17/1,247) | 79.5% |
| 99.5% | 0.023 | 99.5% (1,241/1,247) | 2.5% (40/1,583) | 34 | **4** | **2** | 0.48% (6/1,247) | 85.0% |
| 99.9% | 0.018 | 99.6% (1,242/1,247) | 1.8% (29/1,583) | 24 | **4** | **1** | 0.40% (5/1,247) | 82.8% |
| 100% | 0.016 | 99.8% (1,244/1,247) | 1.4% (22/1,583) | 19 | **2** | **1** | 0.24% (3/1,247) | 86.4% |

*Columns: **Auto-Cleared (PN)** = (TN + FN)/1,583 = normal(TN) + near_normal(FN) + abnormal(FN); only the **normal(TN)** sub-column is a correct clear. **Negative Precision** (NPV) = normal(TN) / Auto-Cleared = of everything cleared, the % genuinely normal (safe). **Miss rate** = (near + abn FN)/1,247. Pos = 1,247 not-normal, Neg = 336 normal. Enriched 21%-normal test set — NPV rises at realistic normal-heavy prevalence (see §4 prevalence caveat).*

**Reading:** operate high on this list to be safe. At the common **≤0.5% miss** bar (the **99.5% row**) the AI clears ~2.5% of the worklist at 85% negative precision; a near-zero miss (0.24%) clears only ~1.4%. This modest ceiling is the direct consequence of v3's modest discrimination (3-class macro AUC 0.85; not-normal-vs-normal AUC 0.90).

---

## 4. Caveats & how to automate *more* safely

| Caveat | What it means | Action |
|---|---|---|
| ⚠️ **Prevalence** | NPV is measured on the enriched 21%-normal test set (not a real population) and shifts with prevalence — falls when pathology is heavier. | Re-measure NPV on the live distribution before quoting any number. |
| **Calibration** | Threshold is dataset-specific. | Re-fit on a held-out slice of *production* data before go-live; re-check periodically. |
| **v3 ceiling** | Safe auto-clear volume is capped by v3's modest discrimination (3-class macro AUC 0.85; not-normal-vs-normal AUC 0.90). | Lift AUC: **2-class reframe** (normal vs not-normal), **near_normal label cleaning**, **ensemble + TTA** (see `run_comparison_v1_v2_v3.md` §4). |
| **Regulatory** | Autonomous rule-out is a clinical decision. | Treat this as scoping analysis, not a green light — needs prospective validation + sign-off. |

---
*Operating points from `eval_autorule.py` over the dumped `series_probs_{val,test}.csv` (patient-level, mean aggregation). No leakage — thresholds set on val, measured on test. Companion: `v3_production_deployment_pos(abnormal).md` (abnormal objective); objective comparison in `v3_objective_comparison.md`.*
