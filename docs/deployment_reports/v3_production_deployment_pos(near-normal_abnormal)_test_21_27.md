# v3 Production Deployment — AI Auto-Rule-Out of Normal CT-Brain Studies (TEST: June 21–27 production week)

> **Model:** ct_brain v3 (`runs/maxvit384_3class_clinical_v3`, best.pt = ep3), MaxViT-MIL, 3-class.
> **Date:** 2026-07-01
> **Test set:** **new, real consecutive-week production sample** — studies reported **21–27 June 2026** (`disk_vdc/test_data/csvs/test_june_21_27.csv`). Unlike the enriched held-out split in the original report, this reflects the **live case mix** (normal prevalence only **4.9%**). Downloaded from OCI `secure-dcm`, pruned to the primary axial series, scored per-study.
> **Goal:** use the AI to **safely auto-clear confidently-normal studies** so radiologists only read the rest — without missing pathology.
> **Scoring unit:** per-study, **mean** aggregation over the study's series. Thresholds are **transferred from the original validation set** (no recalibration) and measured on this new test set.

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
                              │ 5. Study aggregate    │  mean over series → P(normal)
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
- **QC fail-safe (step 3):** any study the pipeline can't process confidently → doctor (never silently dropped). *(An earlier run zero-placeholdered 42 JPEG-Lossless studies for lack of a decoder; now fixed — `pylibjpeg-libjpeg` + `python-gdcm` installed and those 23 studies re-scored, so all numbers below are on real pixels.)*
- **Random audit (step 7a):** a fixed % of auto-cleared studies are still read by a radiologist to measure the *real* miss rate continuously.
- **Monitoring (step 8):** alert if input prevalence drifts (the safety math depends on it), if score distribution shifts (model drift), or if audited miss rate exceeds the budget → auto-revert to "all-to-doctor".
- **Human override always wins;** AI output is advisory infrastructure, not a locked decision.

---

## 2b. Test-set composition — June 21–27 production week

Real consecutive-week sample (not enriched). Scored **per study** → denominators are studies. **Normal prevalence is only 4.9%** — close to the true clinical mix, and far below the enriched 21% of the original held-out split.

| class | series | studies |
|---|--:|--:|
| normal | 508 | 327 |
| near_normal | 5,791 | 3,614 |
| abnormal | 4,346 | 2,741 |
| **not-normal (near_normal + abnormal)** | **10,137** | **6,355** |
| **total** | **10,645** | **6,682** |

Discrimination on this set (per-study, mean aggregation):

| boundary | AUC (this test) | (orig. held-out split) |
|---|--:|--:|
| not-normal vs normal — *this objective* | **0.882** | 0.90 |
| abnormal vs normal only | 0.937 | 0.95 |
| abnormal vs (normal + near_normal) | 0.832 | 0.88 |
| near_normal vs normal (weak boundary) | 0.759 | 0.75 |
| 3-class macro (OvR) | 0.823 | 0.85 |

AUC is modestly lower than the original split — expected for a fresh production week (distribution shift, auto-generated `classification` labels).

---

## 3. Choosing the threshold = choosing a safety budget

Higher threshold = safer (fewer misses) but less automation. **Thresholds are set on the original validation set and applied to this new test set** — i.e. an honest test of transferring last month's operating points to this week's data. **Auto-Cleared (PN)** = predicted-negative = every study that leaves the worklist; it contains the correct clears (normal TN) **and** the misses (near_normal + abnormal FN) — so the misses sit *inside* this column.

| Target Sens (val) | Threshold | Sensitivity achieved (TP/Pos) | Auto-Cleared (PN) | ↳ normal (TN ✓) | ↳ near_normal (FN ✗) | ↳ abnormal (FN ✗) | Miss rate (FN/Pos) | Negative Precision |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 95% | 0.208 | 89.6% (5,691/6,355) | 12.8% (855/6,682) | 191 | **557** | **107** | 10.45% (664/6,355) | 22.3% |
| 98% | 0.066 | 94.9% (6,032/6,355) | 6.8% (456/6,682) | 133 | **273** | **50** | 5.08% (323/6,355) | 29.2% |
| 99% | 0.038 | 97.1% (6,171/6,355) | 4.1% (277/6,682) | 93 | **152** | **32** | 2.90% (184/6,355) | 33.6% |
| 99.5% | 0.023 | 98.4% (6,255/6,355) | 2.5% (166/6,682) | 66 | **81** | **19** | 1.57% (100/6,355) | 39.8% |
| 99.9% | 0.018 | 98.8% (6,280/6,355) | 2.0% (133/6,682) | 58 | **60** | **15** | 1.18% (75/6,355) | 43.6% |
| 100% | 0.016 | 99.1% (6,296/6,355) | 1.6% (110/6,682) | 51 | **46** | **13** | 0.93% (59/6,355) | 46.4% |

*Columns: **Auto-Cleared (PN)** = (TN + FN)/6,682 = normal(TN) + near_normal(FN) + abnormal(FN); only the **normal(TN)** sub-column is a correct clear. **Negative Precision** (NPV) = normal(TN) / Auto-Cleared. **Miss rate** = (near + abn FN)/6,355. Pos = 6,355 not-normal, Neg = 327 normal.*

**Two production-critical findings on this real week:**

1. **Threshold transfer undershoots.** Val thresholds that targeted 95%/99% sensitivity deliver only **89.6%/97.1%** here — the score distribution shifted between the val month and this week. **Recalibrate on recent production data before quoting any operating point.**
2. **NPV collapses at real prevalence.** Because only 4.9% of studies are truly normal, of everything auto-cleared just **22–46%** is genuinely normal (vs 70–86% on the enriched split). At the 4.9%-normal live mix, most "clears" are near_normals/abnormals with a low normal score — so aggressive rule-out is **not safe** here. Operate very high (99.9–100% target), where the AI clears only ~1.6–2.0% of the worklist at a ~0.9–1.2% miss rate, and even then it is auto-clearing more pathology than normals.

**Reading:** on this production week v3's rule-out headroom is small — a direct consequence of (a) low normal prevalence and (b) v3's modest discrimination (not-normal-vs-normal AUC 0.88). Treat as a calibration/monitoring baseline, not a green light.

---

## 4. Caveats & how to automate *more* safely

| Caveat | What it means | Action |
|---|---|---|
| ⚠️ **Threshold transfer** | Val thresholds miss their sensitivity target on this week (89% vs 95%). | **Re-fit thresholds on recent production data** before go-live; re-check continuously (step 8). |
| ⚠️ **Prevalence** | NPV is measured at the real 4.9%-normal mix and is low (22–46%); it rises only if normals are more common. | Re-measure NPV on the live distribution before quoting any number. |
| **Label quality** | `classification` labels here are auto-derived from reports, not adjudicated; near_normal is noisy. | Spot-adjudicate a sample; treat near_normal miss counts as soft. |
| **JPEG-Lossless decode** | ✅ Resolved — 42 studies (transfer syntax `1.2.840.10008.1.2.4.70`) previously failed to decode; `pylibjpeg-libjpeg` + `python-gdcm` now installed in the `ct_brain` env and the 23 affected studies re-scored. | Keep these decoders in the deployment image; keep the QC fail-safe for any future unknown syntax. |
| **v3 ceiling** | Safe auto-clear volume is capped by v3's discrimination (macro AUC 0.82; not-normal-vs-normal 0.88). | Lift AUC: 2-class reframe, near_normal label cleaning, ensemble + TTA, conformal selective prediction. |
| **Regulatory** | Autonomous rule-out is a clinical decision. | Scoping analysis only — needs prospective validation + sign-off. |

---
*Operating points from `eval_autorule.py` over the dumped `series_probs_{val,test}.csv` in `runs/maxvit384_3class_clinical_v3/eval_test_21_27/` (per-study, mean aggregation). Thresholds set on the original val, measured on the June 21–27 test set. Companion: `v3_production_deployment_pos(abnormal)_test_21_27.md` (abnormal objective).*
