# v3 Production Deployment — AI Auto-Rule-Out (ABNORMAL objective) (TEST: June 21–27 production week)

> **Model:** ct_brain v3 (`runs/maxvit384_3class_clinical_v3`, best.pt = ep3), MaxViT-MIL, 3-class.
> **Date:** 2026-07-01
> **Test set:** **new, real consecutive-week production sample** — studies reported **21–27 June 2026** (`disk_vdc/test_data/csvs/test_june_21_27.csv`). Reflects the **live case mix**: abnormal prevalence **41.0%**, normal only 4.9%. Downloaded from OCI `secure-dcm`, pruned to the primary axial series, scored per-study.
> **Objective (this report):** **positive = abnormal**, **negative = normal + near_normal**. The AI escalates only studies that look **abnormal**; everything confidently non-abnormal is auto-cleared. Companion to `v3_production_deployment_pos(near-normal_abnormal)_test_21_27.md` (positive = not-normal).
> **Why this framing:** v3's dangerous-miss class (abnormal) sits on the model's *cleanest* decision boundary (abnormal-vs-normal AUC **0.94** here), while the normal↔near_normal boundary is the weak one (AUC 0.76).
> **Scoring unit:** per-study, **mean** aggregation. Thresholds **transferred from the original val** (no recalibration).

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
                              │ 5. Study aggregate    │  mean over series → P(abnormal)
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

**Safety nets (identical to the not-normal flow):** QC fail-safe → doctor; random audit of auto-cleared studies to measure the *real* abnormal-miss rate; prevalence/drift monitoring with auto-revert; human override always wins. *(An earlier run zero-placeholdered 42 JPEG-Lossless studies for lack of a decoder; now fixed — `pylibjpeg-libjpeg` + `python-gdcm` installed and the 23 affected studies re-scored, so all numbers below are on real pixels.)*

---

## 2b. Test-set composition — June 21–27 production week

Real consecutive-week sample (not enriched). Scored **per study** → denominators are studies. Abnormal prevalence = **41.0%**.

| class | series | studies | role here |
|---|--:|--:|---|
| normal | 508 | 327 | negative (auto-clearable) |
| near_normal | 5,791 | 3,614 | negative (auto-clearable) |
| abnormal | 4,346 | 2,741 | **positive — must escalate** |
| **non-abnormal (normal + near_normal)** | **6,299** | **3,941** | **Negative (Neg)** |
| **total** | **10,645** | **6,682** | |

**Why this framing** — the must-not-miss class (abnormal) sits on the model's *cleanest* boundary:

| boundary | AUC (this test) | (orig. held-out split) |
|---|--:|--:|
| abnormal vs normal only — cleanest, what this objective leans on | **0.937** | 0.95 |
| not-normal vs normal — the other report's objective | 0.882 | 0.90 |
| **abnormal vs (normal + near_normal)** — *this objective* | **0.832** | 0.88 |
| near_normal vs normal — weak boundary, dropped from the safety target | 0.759 | 0.75 |

This objective's headline AUC (0.832) is lower than the not-normal objective's (0.882) — near_normal blurs the negative pool — but the only class you must not miss rides the 0.937 boundary.

---

## 3. Choosing the threshold = choosing a safety budget

Higher threshold = safer (fewer missed abnormals) but less automation. **Thresholds set on the original val, applied to this new week.** **Auto-Cleared (PN)** = predicted-negative = every study that leaves the worklist; it contains the correct clears (normal + near_normal TN) **and** the missed abnormals (FN).

| Target Sens (val) | Threshold | Sensitivity achieved (TP/Pos) | Auto-Cleared (PN) | ↳ normal (TN ✓) | ↳ near_normal (TN ✓) | ↳ abnormal (FN ✗) | Miss rate (FN/Pos) | Negative Precision |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 95% | 0.025 | 93.7% (2,568/2,741) | 23.2% (1,547/6,682) | 247 | 1,127 | **173** | 6.31% (173/2,741) | 88.8% |
| 98% | 0.012 | 96.2% (2,636/2,741) | 14.5% (969/6,682) | 209 | 655 | **105** | 3.83% (105/2,741) | 89.2% |
| 99% | 0.004 | 98.1% (2,688/2,741) | 7.6% (507/6,682) | 134 | 320 | **53** | 1.93% (53/2,741) | 89.5% |
| 99.5% | 0.003 | 98.8% (2,708/2,741) | 5.4% (360/6,682) | 102 | 225 | **33** | 1.20% (33/2,741) | 90.8% |
| 99.9% | 0.003 | 98.9% (2,711/2,741) | 4.6% (310/6,682) | 90 | 190 | **30** | 1.09% (30/2,741) | 90.3% |

*Columns: **Auto-Cleared (PN)** = (TN + FN)/6,682 = normal(TN) + near_normal(TN) + abnormal(FN); only the **abnormal(FN)** sub-column is a miss. **Negative Precision** (NPV) = (normal + near_normal TN) / Auto-Cleared. **Miss rate** = abnormal(FN)/2,741. **specificity** = TN/3,941 = % of clearable studies automated (34.9% / 21.9% / 11.5% / 8.3% / 7.1%). Pos = 2,741 abnormal, Neg = 3,941 non-abnormal.*

**Findings on this real week:**

- **Threshold transfer undershoots** (as in the not-normal report): the val threshold for 95% abnormal-sensitivity delivers **93.7%** here, and the 99% target delivers 98.1% — **recalibrate on recent production data.**
- **Miss floor.** Even the lowest usable threshold still auto-clears **30 abnormal studies** (mean `P(abnormal)` below any threshold), so miss **floors at ~1.09% (30/2,741)** — hard, confidently-mis-scored abnormals (label-review / hard-negative candidates).
- **NPV holds up** (88–91%) because abnormal prevalence is high (41%): of everything cleared, ~89–91% is genuinely non-abnormal. But a large share of clears are **near_normal** (auto-reported despite minor findings): 1,127 of 1,374 correct clears (≈82%) at the 95% point, still ≈69% at 99.5% — the main thing a clinical reviewer must sign off on.

**Reading:** at the **99.5% / 1.20%-miss** point the AI clears ~5.4% of the worklist at 91% negative precision (specificity 8.3%); the **95%** point clears 23.2% but misses 6.3% of abnormals — too high for autonomous rule-out. The abnormal objective again has more usable NPV than the not-normal one on this prevalence, but its volume is capped by the miss floor and by transfer degradation.

---

## 4. Caveats & how to automate *more* safely

| Caveat | What it means | Action |
|---|---|---|
| ⚠️ **Threshold transfer** | Val thresholds miss their sensitivity target on this week (93% vs 95%). | **Re-fit thresholds on recent production data**; re-check continuously (step 8). |
| ⚠️ **Prevalence** | NPV is on the real 41%-abnormal week and shifts with prevalence. | Re-measure NPV on the live distribution before quoting any number. |
| **near_normal auto-reported** | This objective clears near_normal (minor findings) — ≈82% of clears at the 95% point. | Get clinical sign-off on auto-reporting near_normal before counting the volume as a win. |
| **~30-case miss floor** | Miss can't go below ~1.09% — 30 abnormals are confidently mis-scored. | Pull those for label review + hard-negative mining. |
| **Label quality** | `classification` labels auto-derived from reports, not adjudicated. | Spot-adjudicate a sample. |
| **JPEG-Lossless decode** | ✅ Resolved — 42 studies (transfer syntax `1.2.840.10008.1.2.4.70`) previously failed to decode; `pylibjpeg-libjpeg` + `python-gdcm` now installed and the 23 affected studies re-scored. | Keep these decoders in the deployment image; keep the QC fail-safe for any future unknown syntax. |
| **v3 ceiling** | Safe volume capped by v3's discrimination (abn-vs-rest AUC 0.83; abn-vs-normal 0.94). | Lift AUC: rule-out multi-task head + top-k pooling (v5, now training), near_normal label cleaning, conformal selective prediction. |
| **Regulatory** | Autonomous rule-out is a clinical decision. | Scoping analysis only — needs prospective validation + sign-off. |

---
*Operating points from `eval_autorule_abnormal.py` over the dumped `series_probs_{val,test}.csv` in `runs/maxvit384_3class_clinical_v3/eval_test_21_27/` (per-study, mean aggregation). Thresholds set on the original val, measured on the June 21–27 test set. Companion: `v3_production_deployment_pos(near-normal_abnormal)_test_21_27.md` (not-normal objective).*
