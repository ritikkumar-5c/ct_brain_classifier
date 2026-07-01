# v3 Cascaded (Hierarchical) Classification — June 21–27 production week (Stage 1 @98%, Stage 2 swept)

> **Model:** ct_brain v3 (`runs/maxvit384_3class_clinical_v3`), 3-class softmax, **no retraining**.
> **Date:** 2026-07-01
> **Test set:** real consecutive-week production sample, studies reported **21–27 June 2026** (`eval_test_21_27/`). Live case mix: abnormal **41.0%**, normal only 4.9%. Scored **per study**, mean aggregation. **Thresholds transferred from the original val** (no recalibration).
> **Design:** abnormal-first cascade. **Stage 1 fixed at 98% target sensitivity**; **Stage 2 target sensitivity swept** (80→98%).
> Companions: `v3_production_deployment_pos(abnormal)_test_21_27.md` (Stage 1 alone = the flat rule-out), `v3_cascaded_classification_{95_95,95_90,98_98}.md` (original held-out split).

---

## 1. Method — two thresholded stages on one softmax

One forward pass → `P(normal), P(near), P(abn)` (sum to 1). Two sequential thresholds, **both set on the original val**:

| Stage | Score | Rule | Threshold |
|---|---|---|--:|
| **1 — isolate abnormal** | `s₁ = P(abn)` | `s₁ ≥ T1` → **ABNORMAL** (escalate); else → Stage 2 | **T1 = 0.012** (val 98% target) |
| **2 — split the rest** | `s₂ = Padj(near) = P(near)/(P(near)+P(normal))` | `s₂ ≥ T2` → **NEAR_NORMAL** (light review); else → **NORMAL** (auto-clear) | swept (§3) |

Stage 2 re-normalizes (Stage 1 consumed `P(abn)` mass) and uses **Positive = near_normal, Negative = normal** (protect near_normal recall).

---

## 2. Stage 1 (fixed @98%) on the production week

| metric | value |
|---|--:|
| val threshold T1 | 0.012 |
| **achieved abnormal sensitivity** | **96.2% (2,636/2,741)** — transfer undershoot (target was 98%) |
| escalated → doctor | **5,713 (85% of 6,682)** |
| passed to Stage 2 | **969** (209 normal / 655 near / **105 abnormal**) |
| abnormals lost to Stage 2 | 105 (3.83% of abnormals) |

The 969 pass-through is the flat rule-out's entire auto-clear bucket. **Without Stage 2 you would auto-clear all 969 → 3.83% abnormal miss (105 studies), NPV-nonabn 89.2%.** Stage 2 exists to claw that miss back down.

---

## 3. Stage 2 target-sensitivity sweep (on the 969 pass-through)

Splits the pass-through into **light-review** (predicted near) vs **auto-clear** (predicted normal). Higher target → more near_normal (and abnormal) pulled into light-review → smaller but safer auto-clear bucket.

| S2 target (near-sens) | T2 | near sens achieved | normal spec | AUTO-CLEAR total (n/ne/ab) | % worklist | **abn miss** (of 2,741) | NPV non-abn | NPV normal |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 80% | 0.043 | 72.4% | 50.7% | **326** (106/181/39) | 4.9% | 39 = **1.42%** | 88.0% | 32.5% |
| 85% | 0.037 | 75.1% | 45.9% | 292 (96/163/33) | 4.4% | 33 = **1.20%** | 88.7% | 32.9% |
| 90% | 0.029 | 81.7% | 40.2% | 230 (84/120/26) | 3.4% | 26 = **0.95%** | 88.7% | 36.5% |
| 95% | 0.021 | 87.0% | 33.5% | 176 (70/85/21) | 2.6% | 21 = **0.77%** | 88.1% | 39.8% |
| 98% | 0.017 | 90.1% | 29.7% | **143** (62/65/16) | 2.1% | 16 = **0.58%** | 88.8% | 43.4% |

*AUTO-CLEAR (n/ne/ab) = true normal / near / abnormal in the auto-cleared bucket. **abn miss** = abnormals auto-cleared as normal ÷ 2,741. NPV non-abn = (normal+near)/total; NPV normal = normal/total. Stage-2 near-sensitivity also undershoots its val target here (val 80%→test 72%), same transfer gap as Stage 1.*

---

## 4. What Stage 2 buys (vs Stage 1 alone)

| Configuration | auto-clear vol | abn miss | NPV non-abn |
|---|--:|--:|--:|
| **Stage 1 only** (auto-clear all 969) | 969 (14.5%) | 105 = 3.83% | 89.2% |
| + **Stage 2 @80%** | 326 (4.9%) | 39 = 1.42% | 88.0% |
| + **Stage 2 @95%** | 176 (2.6%) | 21 = 0.77% | 88.1% |
| + **Stage 2 @98%** | 143 (2.1%) | 16 = 0.58% | 88.8% |

**Stage 2 is useful on this real week** — it converts Stage-1's 3.83% auto-clear miss into **0.58–1.42%** by diverting suspected near_normal/abnormal into a light-review queue. The price is auto-clear volume: 14.5% → 2–5% of the worklist. This is the opposite of the original held-out split, where Stage 2 collapsed (there the pass-through was cleaner, so splitting only shed volume). Here the pass-through is contaminated (105 abnormals + 655 near in 969), so a second filter genuinely helps.

**But the auto-clear bucket is still mostly near_normal, not normal.** NPV-normal is only 33–43% — of what's auto-cleared, well under half is *truly* normal; the rest is near_normal (minor findings, auto-reported by design under this objective). Only deploy if near_normal auto-reporting is clinically acceptable.

---

## 5. Recommended operating point

**Stage 1 @98% + Stage 2 @95–98%.** At **Stage 2 @98%**: auto-clear **143 studies (2.1% of the worklist)** at **0.58% abnormal miss** and **88.8% non-abnormal NPV** — the safest usable point. If a bit more volume is worth a bit more risk, **@90%** clears 230 (3.4%) at 0.95% miss.

All points still escalate **85%** of the worklist (Stage 1) — the automation ceiling is set by v3's abnormal discrimination on this week (abn-vs-rest AUC **0.83**), not by the cascade.

---

## 6. Caveats

| Caveat | What it means | Action |
|---|---|---|
| ⚠️ **Threshold transfer** | Val thresholds undershoot on this week — Stage 1 96.2% (target 98%), Stage 2 near-sens likewise. | **Re-fit T1/T2 on recent production data**; re-check continuously. |
| ⚠️ **Weak Stage-2 boundary** | near-vs-normal AUC on the pass-through is only **0.66** (the confident-non-abnormal region is the hardest place to separate normal from near). | Limits how clean/large the auto-clear bucket can be; the v5 rule-out head targets this. |
| ⚠️ **near_normal auto-reported** | Auto-clear bucket is 57–68% near_normal (NPV-normal 33–43%). | Clinical sign-off on auto-reporting near_normal before counting volume. |
| ⚠️ **Cascading error** | The 105 abnormals not escalated at Stage 1 can only be recovered *into light-review* by Stage 2, never fully; 16–39 still reach auto-clear. | Higher Stage-1 sensitivity (recalibrated) is the only way to reduce the input error. |
| **Miss floor** | ~30 abnormals are confidently mis-scored (below any threshold). | Label review + hard-negative mining. |
| **v3 ceiling** | abn-vs-rest 0.83, near-vs-normal 0.66 on this week. | Lift AUC: **v5 rule-out head** (training), near_normal label cleaning, conformal selective prediction for a guaranteed bound. |

**Bottom line.** On the real production week the cascade's Stage 2 *does* add value — a light-review tier that cuts the auto-clear abnormal miss from 3.83% to ~0.58% — but only by shrinking auto-clear to ~2% of the worklist, and the cleared bucket is majority near_normal. Stage 1 still escalates 85% of studies. Useful as a 3-tier triage, but the safe auto-clear volume stays small until v3's discrimination improves.

---
*Computed by cascade inference over `eval_test_21_27/series_probs_test.csv` (per-study, mean). T1/T2 set on the original val (`series_probs_val.csv`), applied to the June 21–27 week. No recalibration.*
