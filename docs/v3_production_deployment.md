# v3 Production Deployment — AI Auto-Rule-Out of Normal CT-Brain Studies

> **Model:** ct_brain v3 (`runs/maxvit384_3class_clinical_v3`, best.pt = ep3), MaxViT-MIL, 3-class.
> **Date:** 2026-06-29
> **Goal:** today every study (normal or not) is read by a doctor. Use the AI to **safely auto-clear confidently-normal studies** so radiologists only read the rest — without missing pathology.
> **Scoring unit:** per-patient, **mean** aggregation over the patient's series (best & most robust — see `v3_detailed_report.md` §6).

---

## 1. Core principle — rule-out only, never auto-diagnose

The AI is allowed to do **one** thing autonomously: **remove a study from the doctor's worklist when it is confidently normal.** It never issues a pathology diagnosis on its own. Therefore:

- The only dangerous error is a **missed pathology auto-cleared as normal** (a false-negative leaving the queue).
- The operating threshold is chosen by the **maximum tolerable miss rate** (a safety budget), *not* by how much work we want to save.
- Everything not confidently-normal → radiologist, exactly as today.

Decision score per patient: **`s = 1 − P(normal)`** (the not-normal/pathology score). Auto-clear if `s < threshold`.

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

## 2b. Test-set composition (where the denominators come from)

All numbers below are on the **held-out test split** (patient-grouped, never seen in training/selection):

| class | series | patients |
|---|--:|--:|
| normal | 546 | 336 |
| near_normal | 942 | 558 |
| abnormal | 1,135 | 689 |
| **not-normal (near_normal + abnormal)** | **2,077** | **1,247** |
| **total** | **2,623** | **1,583** |

The deployment analysis scores **per patient** (mean over a patient's series), so denominators are **patients**: **1,247 not-normal** and **336 normal**. Hence "missed pathology (of 1,247)" = not-normal *patients* that were auto-cleared in error, and "% of normal studies auto-cleared" is out of the **336** normal patients. (Full train/val/test split composition is in `v3_detailed_report.md` §0.)

---

## 3. Choosing the threshold = choosing a safety budget

Each threshold trades **miss rate** (safety) against **% of normals automated** (savings). Measured on the held-out test set (patient-level, mean aggregation; threshold set on val, applied to test):

Denominators: not-normal patients (Pos) = **1,247**; normal patients (Neg) = **336**.

| target sens | threshold | Not-Normal Sensitivity (TP/Pos) | % NORMAL auto-cleared = specificity (TN/Neg) | Missed pathology (FN/Pos, miss rate) | NPV (auto-clear safety) | TP / FP / FN / TN |
|--:|--:|--:|--:|--:|--:|--|
| 0.95 | 0.208 | 94.5% (1,178/1,247) | 48.8% (164/336) | 69/1,247 (5.5%) | 70.4% | 1178 / 172 / 69 / 164 |
| 0.98 | 0.066 | 97.8% (1,219/1,247) | 28.9% (97/336) | 28/1,247 (2.3%) | 77.6% | 1219 / 239 / 28 / 97 |
| 0.99 | 0.038 | 98.6% (1,230/1,247) | 19.6% (66/336) | 17/1,247 (1.4%) | 79.5% | 1230 / 270 / 17 / 66 |
| 0.995 | 0.023 | 99.5% (1,241/1,247) | **10.1% (34/336)** | **6/1,247 (0.48%)** | **85.0%** | 1241 / 302 / 6 / 34 |
| 0.999 | 0.018 | 99.6% (1,242/1,247) | 7.1% (24/336) | 5/1,247 (0.40%) | 82.8% | 1242 / 312 / 5 / 24 |
| 1.000 (test) | 0.016 | 99.8% (1,244/1,247) | 5.7% (19/336) | 3/1,247 (0.24%) | 86.4% | 1244 / 317 / 3 / 19 |

*NPV (auto-clear safety) = TN/(TN+FN) = of all auto-cleared studies, the % truly normal. Values are on the **enriched 21%-normal test set** — they rise substantially at a realistic normal-heavy prevalence (see §4).*

*TP/FP/FN/TN in this auto-rule-out context: **TP** = not-normal correctly sent to doctor; **FP** = normal sent to doctor (safe, just not automated); **FN** = not-normal auto-cleared = **missed pathology (dangerous)**; **TN** = normal correctly auto-cleared (the automation win). Pos=1,247 not-normal, Neg=336 normal patients.*

**Reading:** to be clinically safe you must operate high on this list. A common radiology auto-rule-out bar is **miss rate ≤ 0.5%** → that's the **0.995 row**: the AI can safely auto-clear **~10% of normal studies**, missing ~5 in 1,000 pathologies. Pushing to a near-zero miss (0.24%) drops automation to ~6% of normals. This modest ceiling on safe automation is the direct consequence of v3's ~0.85 AUC (confident-normal region is small).

---

## 4. Deployment projection (combined: automation + workload + NPV) — held-out test set

This merges "how many normal cases can be automated" with "workload saved & rule-out safety" into one table, **measured directly on the held-out test set** (patient-level, mean aggregation — no projection). Test composition: 1,583 patients = **336 normal (21%)** / 1,247 not-normal (79%).

| target sens | miss rate | % NORMAL automated (of 336) | workload saved (of 1,583) | NPV (auto-clear safety) | pathology missed (of 1,247, % of all) |
|--:|--:|--:|--:|--:|--:|
| 0.95 | 5.5% | 48.8% (164/336) | 14.7% (233/1,583) | 70.4% | 69 (4.4%) |
| 0.98 | 2.3% | 28.9% (97/336) | 7.9% (125/1,583) | 77.6% | 28 (1.8%) |
| 0.99 | 1.4% | 19.6% (66/336) | 5.2% (83/1,583) | 79.5% | 17 (1.1%) |
| 0.995 | 0.48% | **10.1% (34/336)** | 2.5% (40/1,583) | **85.0%** | 6 (0.38%) |
| 0.999 | 0.40% | 7.1% (24/336) | 1.8% (29/1,583) | 82.8% | 5 (0.32%) |
| ~1.0 | 0.24% | 5.7% (19/336) | 1.4% (22/1,583) | 86.4% | 3 (0.19%) |

*"% NORMAL automated" = specificity (TN/336); "workload saved" = all auto-cleared (TN+FN)/1,583; NPV = TN/(TN+FN); "pathology missed" = FN (of 1,247 not-normal). All measured on the test set.*

**Reading.** At the recommended **0.995 / 0.48%-miss** point: the AI auto-clears **~10% of normal studies** (2.5% of the worklist), at **NPV 85%** — i.e. ~15% of auto-cleared studies are still pathology, which is borderline-to-unsafe on this set. Pushing miss rate lower (→0.24%) shrinks automation to ~6% of normals.

⚠️ **Prevalence caveat.** The test set is *enriched* (only 21% normal), which is NOT a deployment population. NPV is prevalence-dependent: it rises in a normal-heavy population and falls in a pathology-heavy one — so these numbers **must be re-measured on the real production distribution** before go-live (a 5.9%-normal week, for example, would push NPV well below acceptable).

---

## 5. Caveats & how to automate *more* safely

- **Calibration & threshold are dataset-specific.** Re-fit the threshold on a held-out slice of the *production* distribution before go-live, and re-check periodically.
- **The enriched-data NPV warning is real** — §4 is on the 21%-normal test set; re-measure NPV on the true production prevalence before quoting any workload/safety number (a pathology-heavy population pushes NPV down sharply).
- **This is the v3 ceiling.** To auto-clear a *larger* share of normals at the same safety, the model needs higher AUC. Highest-leverage routes (see `run_comparison_v1_v2_v3.md` §4): **2-class reframe** (normal vs not-normal — directly optimizes this exact decision), **near_normal label cleaning**, and **ensemble + TTA**. A 2-class model targeting this rule-out task could materially raise the safe specificity.
- **Regulatory:** autonomous rule-out is a clinical decision and typically requires prospective validation and sign-off; treat the above as the analysis that scopes such a study, not a green light.

---
*Operating points and projections from `eval_autorule.py` over the dumped `series_probs_{val,test}.csv` (patient-level, mean aggregation). No leakage — thresholds set on val, measured on test.*
