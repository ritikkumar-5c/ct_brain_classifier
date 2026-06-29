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

| target sens | threshold | Not-Normal Sensitivity (TP/Pos) | % NORMAL auto-cleared = specificity (TN/Neg) | Missed pathology (FN/Pos, miss rate) |
|--:|--:|--:|--:|--:|
| 0.95 | 0.208 | 94.5% (1,178/1,247) | 48.8% (164/336) | 69/1,247 (5.5%) |
| 0.98 | 0.066 | 97.8% (1,219/1,247) | 28.9% (97/336) | 28/1,247 (2.3%) |
| 0.99 | 0.038 | 98.6% (1,230/1,247) | 19.6% (66/336) | 17/1,247 (1.4%) |
| 0.995 | 0.023 | 99.5% (1,241/1,247) | **10.1% (34/336)** | **6/1,247 (0.48%)** |
| 0.999 | 0.018 | 99.6% (1,242/1,247) | 7.1% (24/336) | 5/1,247 (0.40%) |
| 1.000 (test) | 0.016 | 99.8% (1,244/1,247) | 5.7% (19/336) | 3/1,247 (0.24%) |

**Reading:** to be clinically safe you must operate high on this list. A common radiology auto-rule-out bar is **miss rate ≤ 0.5%** → that's the **0.995 row**: the AI can safely auto-clear **~10% of normal studies**, missing ~5 in 1,000 pathologies. Pushing to a near-zero miss (0.24%) drops automation to ~6% of normals. This modest ceiling on safe automation is the direct consequence of v3's ~0.85 AUC (confident-normal region is small).

---

## 4. The prevalence effect — workload saved & rule-out safety (NPV)

**% of normals automated (specificity) is a fixed model property**, but the **total workload saved** and the **rule-out safety (NPV = of auto-cleared studies, the fraction truly normal)** depend heavily on the **real-world prevalence of normals**. Our test set is *enriched* (only 21% normal), which is NOT a deployment population. Projection at the **0.995-sensitivity / 0.48%-miss** operating point:

| deployment population | % normal | **% of ALL studies auto-cleared** | **NPV** (auto-clear safety) | pathology missed per 1,000 studies |
|---|--:|--:|--:|--:|
| enriched test set | 21% | 2.5% | 0.847 | 3.8 |
| balanced | 50% | 5.3% | 0.955 | 2.4 |
| typical ED/outpatient | 80% | 8.2% | 0.988 | 1.0 |
| screening-heavy | 90% | 9.2% | 0.995 | 0.5 |

**Two critical takeaways:**
1. **Validate at the true prevalence.** On the enriched test set the auto-clear NPV is only ~0.85 (≈15% of auto-cleared are actually pathology — unacceptable). At a realistic mostly-normal population the **same model** reaches NPV ≈ 0.99, because confidently-normal predictions are far more likely to be right when normals dominate. *Deployment safety is acceptable only in a genuinely normal-heavy population.*
2. **Workload savings are modest:** ~8–9% of *all* studies removed from the worklist at a safe operating point in a realistic population — meaningful but not dramatic, because we only remove normals and the safe specificity is low.

---

## 5. Direct answer — how many normal cases can the AI automate?

| safety budget (miss rate) | required sensitivity | **% of NORMAL studies auto-cleared** | workload saved @80% normal pop. |
|---|--:|--:|--:|
| ≤ 1.4% | 99% | **19.6% (66/336)** | ~16% of all studies |
| **≤ 0.5%** (recommended) | 99.5% | **10.1% (34/336)** | ~8% of all studies |
| ≤ 0.25% (near-zero) | ~100% | **5.7% (19/336)** | ~5% of all studies |

**Recommended deployment:** operate at **≥99.5% not-normal sensitivity** (≈0.5% miss), score **per-patient with mean aggregation**, deploy **only in a normal-heavy population** (≥80% normal, where NPV ≥ 0.99), keep a **random audit** of auto-cleared studies, and **monitor prevalence/drift** with auto-revert. Expected effect: **~10% of normal studies (≈8% of total worklist) safely removed from radiologist review.**

---

## 6. Caveats & how to automate *more* safely

- **Calibration & threshold are dataset-specific.** Re-fit the threshold on a held-out slice of the *production* distribution before go-live, and re-check periodically.
- **The enriched-data NPV warning is real** — do not quote the 8–9% workload number until prevalence is confirmed in production.
- **This is the v3 ceiling.** To auto-clear a *larger* share of normals at the same safety, the model needs higher AUC. Highest-leverage routes (see `run_comparison_v1_v2_v3.md` §4): **2-class reframe** (normal vs not-normal — directly optimizes this exact decision), **near_normal label cleaning**, and **ensemble + TTA**. A 2-class model targeting this rule-out task could materially raise the safe specificity.
- **Regulatory:** autonomous rule-out is a clinical decision and typically requires prospective validation and sign-off; treat the above as the analysis that scopes such a study, not a green light.

---
*Operating points and projections from `eval_autorule.py` over the dumped `series_probs_{val,test}.csv` (patient-level, mean aggregation). No leakage — thresholds set on val, measured on test.*
