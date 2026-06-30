# v3 Production Deployment — AI Auto-Rule-Out (ABNORMAL objective)

> **Model:** ct_brain v3 (`runs/maxvit384_3class_clinical_v3`, best.pt = ep3), MaxViT-MIL, 3-class.
> **Date:** 2026-06-30
> **Objective (this report):** **positive = abnormal**, **negative = normal + near_normal**. The AI escalates only studies that look **abnormal**; everything confidently non-abnormal (normal *and* near_normal) is auto-cleared. This is the companion to `v3_production_deployment.md`, which uses **positive = not-normal (near_normal + abnormal)**.
> **Why this framing:** v3's dangerous-miss class (abnormal) sits on the model's *cleanest* decision boundary (abnormal-vs-normal AUC **0.95**), while the normal↔near_normal boundary is the weak one (AUC 0.75). Defining the safety target as "never auto-clear an **abnormal** study" — and treating near_normal as clearable — aligns the deployment with what the model is actually good at.
> **Scoring unit:** per-patient, **mean** aggregation over the patient's series.

---

## 1. Core principle — escalate abnormal, auto-clear the rest

The AI does **one** thing autonomously: **remove a study from the doctor's worklist when it is confidently non-abnormal.** It never issues a diagnosis. The difference from the not-normal report:

- **What counts as a dangerous miss is narrower:** only a **truly abnormal** study auto-cleared. A near_normal study auto-cleared is **not** a safety miss here — it is an accepted outcome of this objective.
- **What gets auto-cleared is broader:** both **normal** and **near_normal** studies leave the worklist (near_normal is auto-reported as "no abnormal pathology / minor findings only").
- The operating threshold is set by the **maximum tolerable abnormal-miss rate** (a safety budget).

Decision score per patient: **`s = P(abnormal)`**. Auto-clear if `s < threshold`; escalate to a radiologist if `s ≥ threshold`.

> ⚠️ **This is a more aggressive clinical promise than the not-normal report.** It auto-reports near_normal studies (which *do* have findings, by definition). Whether that is acceptable is a clinical/regulatory decision, not a modelling one — see §5. The numbers below look much better than the not-normal report partly because near_normal misses are no longer counted, and partly because more volume is cleared.

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

## 2b. Test-set composition (where the denominators come from)

All numbers below are on the **held-out test split** (patient-grouped, never seen in training/selection):

| class | series | patients | role in THIS objective |
|---|--:|--:|---|
| normal | 546 | 336 | negative (auto-clearable) |
| near_normal | 942 | 558 | negative (auto-clearable) |
| abnormal | 1,135 | 689 | **positive (must escalate)** |
| **non-abnormal (normal + near_normal)** | **1,488** | **894** | **negative (Neg)** |
| **total** | **2,623** | **1,583** | |

The deployment analysis scores **per patient** (mean over series), so denominators are **patients**: **689 abnormal (Pos)** and **894 non-abnormal (Neg)**. Test prevalence of abnormal = **43.5%** (enriched — not a deployment population; see §4).

**Discrimination on test (patient-mean):**

| boundary | AUC | note |
|---|--:|---|
| **abnormal vs (normal + near_normal)** — *this objective* | **0.882** | near_normal dilutes the negative pool |
| not-normal vs normal — *the other report's objective* | 0.903 | |
| abnormal vs normal only | **0.947** | the model's cleanest boundary — what this objective leans on |
| near_normal vs normal | 0.748 | the weak boundary, removed from the safety target here |

> Note: this objective's headline AUC (0.882) is **slightly lower** than the not-normal objective's (0.903) — because adding near_normal into the *negative* pool blurs it. The advantage of this framing is **not** a higher AUC; it is that the only class you must not miss (abnormal) lives on the 0.95-AUC boundary, and near_normal errors no longer count against safety.

---

## 3. Choosing the threshold = choosing a safety budget

Each threshold trades **abnormal-miss rate** (safety) against **% of non-abnormal studies automated**. Measured on the held-out test set (patient-level, mean aggregation; threshold set on val, applied to test).

Denominators: abnormal patients (Pos) = **689**; non-abnormal patients (Neg) = **894**.

| target sens | threshold | Abnormal Sensitivity (TP/Pos) | % NON-ABN auto-cleared = specificity (TN/Neg) | Missed abnormal (FN/Pos, miss rate) | NPV (auto-clear safety) | TP / FP / FN / TN |
|--:|--:|--:|--:|--:|--:|--|
| 0.95 | 0.025 | 94.9% (654/689) | 43.1% (385/894) | 35/689 (5.1%) | 91.7% | 654 / 509 / 35 / 385 |
| 0.98 | 0.012 | 98.0% (675/689) | 27.7% (248/894) | 14/689 (2.0%) | 94.7% | 675 / 646 / 14 / 248 |
| 0.99 | 0.004 | 99.1% (683/689) | 14.1% (126/894) | 6/689 (0.87%) | 95.5% | 683 / 768 / 6 / 126 |
| 0.995 | 0.003 | 99.4% (685/689) | **10.9% (97/894)** | **4/689 (0.58%)** | **96.0%** | 685 / 797 / 4 / 97 |
| 0.999 | 0.003 | 99.4% (685/689) | 9.7% (87/894) | 4/689 (0.58%) | 95.6% | 685 / 807 / 4 / 87 |

*NPV (auto-clear safety) = TN/(TN+FN) = of all auto-cleared studies, the % truly non-abnormal. Values are on the **enriched 43.5%-abnormal test set** — they rise substantially at a realistic abnormal-light prevalence (see §4).*

*TP/FP/FN/TN in this auto-rule-out context: **TP** = abnormal correctly escalated; **FP** = non-abnormal escalated (safe, just not automated); **FN** = abnormal auto-cleared = **missed pathology (dangerous)**; **TN** = non-abnormal correctly auto-cleared (the automation win). Pos = 689 abnormal, Neg = 894 non-abnormal patients.*

**Irreducible miss floor.** Pushing the threshold to its minimum (0.003) still leaves **4 abnormal patients** auto-cleared — their mean `P(abnormal)` sits below any usable threshold. So on this set the abnormal-miss rate **floors at ~0.58% (4/689)** and cannot reach the near-zero values the not-normal objective could. These 4 are the hard, confidently-mis-scored abnormal cases (candidates for label review / hard-negative mining).

**Reading:** at the **0.995 / 0.58%-miss** point the AI auto-clears **~11% of non-abnormal studies** at **NPV 96.0%**. At the looser **0.95** point it clears **43% of non-abnormal studies** but misses **5.1% of abnormal** — too high for autonomous rule-out. The safety-vs-volume knob is the same shape as the not-normal report, just shifted by the narrower miss definition.

---

## 3b. What gets auto-cleared, and what gets missed

Because the positive class here is **pure abnormal**, every false-negative is an abnormal study — there is no near/abn split to make on the misses (unlike the not-normal report's §3b). The clinically important breakdown for *this* objective is instead **the composition of the auto-cleared bucket** — how many of the auto-reported studies are genuinely normal vs near_normal (which carry minor findings):

| target sens | threshold | auto-cleared (TN) | of which normal | of which near_normal | missed abnormal (FN) |
|--:|--:|--:|--:|--:|--:|
| **0.95** | **0.025** | **385** | **234** | **151** | **35** |
| 0.98 | 0.012 | 248 | 174 | 74 | 14 |
| 0.99 | 0.004 | 126 | 100 | 26 | 6 |
| 0.995 | 0.003 | 97 | 78 | 19 | 4 |
| 0.999 | 0.003 | 87 | 71 | 16 | 4 |

**At the 0.95 operating point:** of **385** auto-cleared non-abnormal studies, **234 are normal and 151 are near_normal** — so **~39% of what this objective auto-reports carries minor (near_normal) findings**. As you tighten the threshold the near_normal share of the cleared bucket shrinks (only the most confidently-non-abnormal survive): 151/385 ≈ 39% at 0.95 → 19/97 ≈ 20% at 0.995. This is the price of the more aggressive promise, and the main thing a clinical reviewer must sign off on.

---

## 4. Deployment projection (automation + workload + NPV) — held-out test set

Test prevalence of abnormal = 43.5%, which is **enriched** and not a deployment population — NPV is prevalence-dependent and must be re-measured before go-live. Projection at several abnormal prevalences:

| target sens | abn-miss rate | % NON-ABN automated (of 894) | workload saved (of 1,583) | NPV @ test prev (43.5% abn) | NPV @ 10% abn | NPV @ 5% abn |
|--:|--:|--:|--:|--:|--:|--:|
| 0.95 | 5.1% | 43.1% (385/894) | 26.5% (420/1,583) | 91.7% | 98.7% | 99.4% |
| 0.98 | 2.0% | 27.7% (248/894) | 16.6% (262/1,583) | 94.7% | 99.2% | 99.6% |
| 0.99 | 0.87% | 14.1% (126/894) | 8.3% (132/1,583) | 95.5% | 99.3% | 99.7% |
| 0.995 | 0.58% | **10.9% (97/894)** | 6.4% (101/1,583) | **96.0%** | 99.4% | 99.7% |
| 0.999 | 0.58% | 9.7% (87/894) | 5.7% (91/1,583) | 95.6% | 99.3% | 99.7% |

*"% NON-ABN automated" = specificity (TN/894); "workload saved" = all auto-cleared (TN+FN)/1,583; NPV = TN/(TN+FN). The NPV-at-prevalence columns reprice the test operating point onto abnormal-light populations (the realistic direction).*

**Reading.** At the recommended **0.995 / 0.58%-miss** point, the AI auto-clears **~11% of non-abnormal studies** (6.4% of the worklist) at **NPV 96%** on the enriched set — rising to **>99% NPV** at realistic (5–10% abnormal) prevalence. The abnormal-miss rate cannot be driven below ~0.58% on this model (the 4-case floor, §3).

⚠️ **Prevalence caveat.** Same as the not-normal report: NPV rises in an abnormal-light population and falls in an abnormal-heavy one. Re-measure on the true production distribution before quoting any number.

---

## 5. ABNORMAL objective vs NOT-NORMAL objective — which to deploy?

Both are computed identically (same model, same patient-mean aggregation, threshold-on-val → measured-on-test). Side-by-side at the **0.95** and **0.995** target-sensitivity points:

| metric (@0.95 target) | NOT-NORMAL objective | ABNORMAL objective |
|---|--:|--:|
| dangerous-miss class | near_normal **+** abnormal | abnormal **only** |
| miss rate | 5.5% (69/1,247) | 5.1% (35/689) |
| NPV (auto-clear safety) | 70.4% | **91.7%** |
| % of negatives auto-cleared | 48.8% (164/336 normal) | 43.1% (385/894 non-abn) |
| workload saved (of 1,583) | 14.7% (233) | **26.5% (420)** |
| what's auto-reported | normal only | normal **+ near_normal** |

| metric (@0.995 target) | NOT-NORMAL objective | ABNORMAL objective |
|---|--:|--:|
| miss rate | 0.48% (6/1,247) | 0.58% (4/689) |
| NPV | 85.0% | **96.0%** |
| workload saved (of 1,583) | 2.5% (40) | **6.4% (101)** |
| min achievable miss | 0.24% (3 cases) | 0.58% floor (4 cases) |

**Trade-off in one line:** the abnormal objective **clears ~1.8× the volume at much higher NPV** (91.7% vs 70.4% at 0.95) — *but it does so by auto-reporting near_normal studies and by no longer counting near_normal as a miss.* The not-normal objective is the **stricter clinical promise** ("auto-cleared = truly normal"); the abnormal objective is the **higher-throughput promise** ("auto-cleared = no abnormal pathology, possibly minor findings").

- If the clinical/regulatory requirement is "**auto-reported studies must be normal**" → use the **not-normal** report. near_normal goes to a doctor.
- If "**auto-reported studies must contain no abnormal pathology** (minor incidental findings acceptable in an auto-draft)" → the **abnormal** objective is materially more useful and rides the model's strongest boundary.

Neither is yet at a safe-and-useful operating point on v3 (the abnormal objective floors at 0.58% miss; the not-normal one collapses to ~10% volume at the ≤0.5% bar). The model improvements that lift both are the same (see §6).

---

## 6. Caveats & how to automate *more* safely

- **Calibration & threshold are dataset-specific.** Re-fit the threshold on a held-out slice of the *production* distribution before go-live; re-check periodically.
- **Enriched-data NPV warning is real** — §4 is on the 43.5%-abnormal test set. Re-measure NPV on true production prevalence before quoting any workload/safety number.
- **The 4-case abnormal-miss floor** is the binding limit for this objective. Those 4 confidently-mis-scored abnormal patients should be pulled for **label review + hard-negative mining**; they cap how safe this objective can ever get on v3.
- **near_normal in the cleared bucket** (§3b) is a clinical-acceptability question, not a modelling one — get sign-off on auto-reporting near_normal before treating this objective's higher volume as a win.
- **This is the v3 ceiling.** Higher AUC lifts both objectives. Highest-leverage routes (see `run_comparison_v1_v2_v3.md` §4 and `ct-brain-autoreport-strategy`): **rule-out multi-task head + top-k pooling** (v5, trained), **near_normal label cleaning**, **enrich the normal calibration set**, and **conformal selective prediction** for a guaranteed miss bound.
- **Regulatory:** autonomous rule-out is a clinical decision requiring prospective validation and sign-off; treat this as the analysis that scopes such a study, not a green light.

---
*Operating points and projections from `eval_autorule_abnormal.py` over the dumped `series_probs_{val,test}.csv` (patient-level, mean aggregation). No leakage — thresholds set on val, measured on test. Companion: `v3_production_deployment.md` (not-normal objective).*
