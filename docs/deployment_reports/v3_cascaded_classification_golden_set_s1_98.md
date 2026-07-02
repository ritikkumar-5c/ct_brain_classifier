# v3 Cascaded (Hierarchical) Classification — curated GOLDEN SET (Stage 1 @98%, Stage 2 swept)

> **Model:** ct_brain v3 (`runs/maxvit384_3class_clinical_v3`), 3-class softmax, **no retraining**.
> **Date:** 2026-07-02
> **Test set:** curated golden set (`ct_brain_golden_set_storage.csv`), DICOM under `disk_vdc/ct_brain_dicom`. **945 studies** scored per study (mean over series, all series ≥10 slices). Case mix: normal **5.0%**, abnormal **60.6%**. **Thresholds transferred from the original val** (no recalibration).
> **Design:** abnormal-first cascade. **Stage 1 fixed at 98% target sensitivity**; **Stage 2 target sensitivity swept** (80→98%).
> Companions: `v3_production_deployment_pos(abnormal)_golden_set.md` (Stage 1 alone = the flat rule-out), `v3_cascaded_classification_test_21_27_s1_98.md` (production-week equivalent).

---

## 1. Method — two thresholded stages on one softmax

One forward pass → `P(normal), P(near), P(abn)` (sum to 1). Two sequential thresholds, **both set on the original val**:

| Stage | Score | Rule | Threshold |
|---|---|---|--:|
| **1 — isolate abnormal** | `s₁ = P(abn)` | `s₁ ≥ T1` → **ABNORMAL** (escalate); else → Stage 2 | **T1 = 0.012** (val 98% target) |
| **2 — split the rest** | `s₂ = Padj(near) = P(near)/(P(near)+P(normal))` | `s₂ ≥ T2` → **NEAR_NORMAL** (light review); else → **NORMAL** (auto-clear) | swept (§3) |

Stage 2 re-normalizes (Stage 1 consumed `P(abn)` mass) and uses **Positive = near_normal, Negative = normal**.

---

## 2. Stage 1 (fixed @98%) on the golden set

| metric | value |
|---|--:|
| val threshold T1 | 0.012 |
| **achieved abnormal sensitivity** | **95.3% (546/573)** — transfer undershoot (target 98%) |
| escalated → doctor | **843 (89% of 945)** |
| passed to Stage 2 | **102** (26 normal / 49 near / **27 abnormal**) |
| abnormals lost to Stage 2 | 27 (4.71% of abnormals) |

Without Stage 2 you would auto-clear all 102 → **4.71% abnormal miss (27 studies)**, NPV-nonabn 73.5% (= the flat rule-out's 98% row). Stage 2 exists to claw that miss down.

---

## 3. Stage 2 target-sensitivity sweep (on the 102 pass-through)

Splits the pass-through into **light-review** (predicted near) vs **auto-clear** (predicted normal). Higher target → more near_normal (and abnormal) pulled into light-review → smaller but safer auto-clear.

| S2 target (near-sens) | T2 | near sens achieved | normal spec | AUTO-CLEAR total (n/ne/ab) | % worklist | **abn miss** (of 573) | NPV non-abn |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 80% | 0.043 | 67.3% | 61.5% | **41** (16/16/9) | 4.3% | 9 = **1.57%** | 78.0% |
| 85% | 0.037 | 69.4% | 57.7% | 38 (15/15/8) | 4.0% | 8 = **1.40%** | 78.9% |
| 90% | 0.029 | 79.6% | 42.3% | 28 (11/10/7) | 3.0% | 7 = **1.22%** | 75.0% |
| 95% | 0.021 | 85.7% | 34.6% | 20 (9/7/4) | 2.1% | 4 = **0.70%** | 80.0% |
| 98% | 0.017 | 87.8% | 26.9% | **15** (7/6/2) | 1.6% | 2 = **0.35%** | 86.7% |

*AUTO-CLEAR (n/ne/ab) = true normal / near / abnormal in the auto-cleared bucket. **abn miss** = abnormals auto-cleared as normal ÷ 573. NPV non-abn = (normal+near)/total. near-vs-normal AUC on the pass-through = 0.66 (the confident-non-abnormal region is the hardest to separate).*

---

## 4. What Stage 2 buys (vs Stage 1 alone)

| Configuration | auto-clear vol | abn miss | NPV non-abn |
|---|--:|--:|--:|
| **Stage 1 only** (auto-clear all 102) | 102 (10.8%) | 27 = 4.71% | 73.5% |
| + **Stage 2 @85%** | 38 (4.0%) | 8 = 1.40% | 78.9% |
| + **Stage 2 @90%** | 28 (3.0%) | 7 = 1.22% | 75.0% |
| + **Stage 2 @98%** | 15 (1.6%) | 2 = 0.35% | 86.7% |

**Stage 2 adds value on the golden set** — a light-review tier that cuts the auto-clear abnormal miss from 4.71% to **0.35–1.57%**, at the cost of auto-clear volume (10.8% → 1.6–4.3% of the worklist). As on the production week (and unlike the clean held-out split), the pass-through is contaminated (27 abnormals + 49 near in 102), so a second filter helps.

**But the auto-clear bucket stays small and mixed.** Auto-clear is only 1.6–4.3% of the worklist, and much of it is near_normal (auto-reported by design). Only deploy if near_normal auto-reporting is clinically acceptable.

---

## 5. Recommended operating point

**Stage 1 @98% + Stage 2 @90–95%:** at **@95%** the AI auto-clears **20 studies (2.1% of the worklist)** at **0.70% abnormal miss** and **80% non-abnormal NPV**. For maximum safety, **@98%** clears 15 (1.6%) at 0.35% miss; for a touch more volume, **@85%** clears 38 (4.0%) at 1.40% miss.

Either way Stage 1 escalates **89%** of the worklist — the automation ceiling is set by v3's abnormal discrimination on this abnormal-heavy set (abn-vs-rest AUC **0.77**), not by the cascade.

---

## 6. Caveats

| Caveat | What it means | Action |
|---|---|---|
| ⚠️ **Threshold transfer** | Val thresholds undershoot here (Stage 1 95.3% vs 98% target). | Re-fit T1/T2 on recent production data; re-check continuously. |
| ⚠️ **Weak Stage-2 boundary** | near-vs-normal AUC on the pass-through is only **0.66**. | Limits how clean/large the auto-clear bucket can be. |
| ⚠️ **near_normal auto-reported** | Auto-clear bucket is majority near_normal at looser thresholds. | Clinical sign-off on auto-reporting near_normal. |
| ⚠️ **Cascading error** | The 27 abnormals not escalated at Stage 1 can only be recovered *into light-review*, never fully; 2–9 still reach auto-clear. | Higher (recalibrated) Stage-1 sensitivity is the only way to reduce input error. |
| **No primary-series pruning** | All series ≥10 slices scored and mean-aggregated. | A primary-axial selector may lift accuracy; re-score if available. |
| **Coverage / labels** | 945/989 studies scored; labels auto-derived from reports. | Spot-adjudicate; keep JPEG decoders + QC fail-safe. |
| **v3 ceiling** | abn-vs-rest 0.77, near-vs-normal 0.70 on this set. | Lift AUC: v5 rule-out head, near_normal label cleaning, conformal selective prediction. |
| **Regulatory** | Autonomous rule-out is a clinical decision. | Scoping analysis only — needs prospective validation + sign-off. |

**Bottom line.** On the golden set the cascade's Stage 2 helps — a light-review tier that cuts the auto-clear abnormal miss from 4.71% to ~0.35–1.57% — but only by shrinking auto-clear to 1.6–4.3% of the worklist, with the cleared bucket majority near_normal. Stage 1 still escalates 89%. Useful as 3-tier triage; safe auto-clear volume stays small on this abnormal-heavy set until v3's discrimination improves.

---
*Computed by cascade inference over `runs/maxvit384_3class_clinical_v3/eval_golden_set/series_probs_test.csv` (per-study, mean). T1/T2 set on the original val, applied to the golden set. No recalibration.*
