# v5 Cascaded (Hierarchical) Classification — June 21–27 production week (Stage 1 @98%, Stage 2 swept)

> **Model:** ct_brain v5 (`runs/maxvit384_3class_clinical_v5`, ep8 snapshot; **training still ongoing**), MaxViT-MIL, 3-class (+ rule-out multi-task head). Cascade runs on the 3-class softmax — **no retraining**.
> **Date:** 2026-07-01
> **Test set:** real consecutive-week production sample, studies reported **21–27 June 2026** (`eval_ep8/test_21_27/`). Live case mix: abnormal **41.0%**, normal only 4.9%. Scored **per study**, mean aggregation. **Thresholds transferred from the original val** (no recalibration).
> **Design:** abnormal-first cascade. **Stage 1 fixed at 98% target sensitivity**; **Stage 2 target sensitivity swept** (80→98%).
> Companions: `v5_production_deployment_pos(abnormal)_test_21_27.md` (Stage 1 alone = the flat rule-out), `v3_cascaded_classification_test_21_27_s1_98.md` (v3 equivalent).

---

## 1. Method — two thresholded stages on one softmax

One forward pass → `P(normal), P(near), P(abn)` (sum to 1). Two sequential thresholds, **both set on the original val**:

| Stage | Score | Rule | Threshold |
|---|---|---|--:|
| **1 — isolate abnormal** | `s₁ = P(abn)` | `s₁ ≥ T1` → **ABNORMAL** (escalate); else → Stage 2 | **T1 = 0.048** (val 98% target) |
| **2 — split the rest** | `s₂ = Padj(near) = P(near)/(P(near)+P(normal))` | `s₂ ≥ T2` → **NEAR_NORMAL** (light review); else → **NORMAL** (auto-clear) | swept (§3) |

Stage 2 re-normalizes (Stage 1 consumed `P(abn)` mass) and uses **Positive = near_normal, Negative = normal**.

---

## 2. Stage 1 (fixed @98%) on the production week

| metric | value |
|---|--:|
| val threshold T1 | 0.048 |
| **achieved abnormal sensitivity** | **96.7% (2,650/2,741)** — transfer undershoot (target 98%) |
| escalated → doctor | **5,794 (87% of 6,682)** |
| passed to Stage 2 | **888** (179 normal / 618 near / **91 abnormal**) |
| abnormals lost to Stage 2 | 91 (3.32% of abnormals) |

Without Stage 2 you would auto-clear all 888 → **3.32% abnormal miss (91 studies)**, NPV-nonabn 89.8% (= the flat rule-out's 98% row). Stage 2 exists to claw that miss down. *(v5's Stage 1 is marginally cleaner than v3's on this week: 91 abnormals leaked vs 105.)*

---

## 3. Stage 2 target-sensitivity sweep (on the 888 pass-through)

Splits the pass-through into **light-review** (predicted near) vs **auto-clear** (predicted normal). Higher target → more near_normal (and abnormal) pulled into light-review → smaller but safer auto-clear.

| S2 target (near-sens) | T2 | near sens achieved | normal spec | AUTO-CLEAR total (n/ne/ab) | % worklist | **abn miss** (of 2,741) | NPV non-abn | NPV normal |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 80% | 0.129 | 74.3% | 54.2% | **289** (97/159/33) | 4.3% | 33 = **1.20%** | 88.6% | 33.6% |
| 85% | 0.107 | 80.9% | 43.6% | 223 (78/118/27) | 3.3% | 27 = **0.99%** | 87.9% | 35.0% |
| 90% | 0.093 | 84.6% | 35.8% | 183 (64/95/24) | 2.7% | 24 = **0.88%** | 86.9% | 35.0% |
| 95% | 0.054 | 96.8% | 14.5% | 56 (26/20/10) | 0.8% | 10 = **0.36%** | 82.1% | 46.4% |
| 98% | 0.044 | 98.2% | 8.4% | **33** (15/11/7) | 0.5% | 7 = **0.26%** | 78.8% | 45.5% |

*AUTO-CLEAR (n/ne/ab) = true normal / near / abnormal in the auto-cleared bucket. **abn miss** = abnormals auto-cleared as normal ÷ 2,741. NPV non-abn = (normal+near)/total; NPV normal = normal/total. Stage-2 near-sensitivity transfers unevenly: val 80%→test 74%, but val 95%→test 97% (the tight thresholds happen to transfer better on this week).*

---

## 4. What Stage 2 buys (vs Stage 1 alone)

| Configuration | auto-clear vol | abn miss | NPV non-abn |
|---|--:|--:|--:|
| **Stage 1 only** (auto-clear all 888) | 888 (13.3%) | 91 = 3.32% | 89.8% |
| + **Stage 2 @85%** | 223 (3.3%) | 27 = 0.99% | 87.9% |
| + **Stage 2 @90%** | 183 (2.7%) | 24 = 0.88% | 86.9% |
| + **Stage 2 @98%** | 33 (0.5%) | 7 = 0.26% | 78.8% |

**Stage 2 is useful on this real week** — it converts Stage-1's 3.32% auto-clear miss into **0.26–1.20%** by diverting suspected near_normal/abnormal into a light-review queue, at the cost of auto-clear volume (13.3% → 0.5–4.3% of the worklist). As with v3, the pass-through is contaminated (91 abnormals + 618 near in 888), so a second filter helps — the opposite of the clean held-out split, where Stage 2 only shed volume.

**The auto-clear bucket is still majority near_normal, not normal.** NPV-normal 34–46% — under half of what's auto-cleared is *truly* normal; the rest is near_normal (minor findings, auto-reported by design). Deploy only if near_normal auto-reporting is clinically acceptable.

---

## 5. Recommended operating point

**Stage 1 @98% + Stage 2 @90%:** auto-clear **183 studies (2.7% of the worklist)** at **0.88% abnormal miss** and **86.9% non-abnormal NPV** — a good safety/volume balance. For maximum safety, **@98%** clears 33 (0.5%) at 0.26% miss; for more volume, **@85%** clears 223 (3.3%) at 0.99% miss.

Either way Stage 1 escalates **87%** of the worklist — the automation ceiling is set by v5's abnormal discrimination on this week (abn-vs-rest AUC **0.81**), not by the cascade.

---

## 6. v5 vs v3 on this week (cascade, Stage 1 @98%)

| | Stage-1 escalate | Stage-1 abn leaked | S2@90% auto-clear | S2@90% abn miss |
|---|--:|--:|--:|--:|
| **v3** | 85% | 105 (3.83%) | 230 (3.4%) | 0.95% |
| **v5 (ep8)** | 87% | 91 (3.32%) | 183 (2.7%) | 0.88% |

v5's Stage 1 leaks slightly fewer abnormals (91 vs 105) and reaches a marginally lower auto-clear miss — but v5 is an **ep8 snapshot of an unfinished run**, and the near↔normal boundary on the pass-through is still weak (AUC 0.67 vs v3's 0.66). No decisive cascade advantage yet.

---

## 7. Caveats

| Caveat | What it means | Action |
|---|---|---|
| **Snapshot** | v5 is still training; this is ep8. Re-run when training finishes. |
| ⚠️ **Threshold transfer** | Val thresholds undershoot on this week (Stage 1 96.7% vs 98% target). | Re-fit T1/T2 on recent production data; re-check continuously. |
| ⚠️ **Weak Stage-2 boundary** | near-vs-normal AUC on the pass-through is only **0.67** (the confident-non-abnormal region is the hardest place to separate normal from near). | Limits how clean/large the auto-clear bucket can be. |
| ⚠️ **near_normal auto-reported** | Auto-clear bucket is 54–66% near_normal (NPV-normal 34–46%). | Clinical sign-off on auto-reporting near_normal. |
| ⚠️ **Cascading error** | The 91 abnormals not escalated at Stage 1 can only be recovered *into light-review*, never fully; 7–33 still reach auto-clear. | Higher (recalibrated) Stage-1 sensitivity is the only way to reduce input error. |
| **Label quality** | `classification` labels auto-derived from reports, not adjudicated. | Spot-adjudicate a sample. |
| **Regulatory** | Autonomous rule-out is a clinical decision. | Scoping analysis only — needs prospective validation + sign-off. |

**Bottom line.** On the real production week the v5 cascade's Stage 2 adds value — a light-review tier that cuts the auto-clear abnormal miss from 3.32% to ~0.88% (at S2@90%) — but only by shrinking auto-clear to ~2.7% of the worklist, with the cleared bucket majority near_normal. Stage 1 still escalates 87%. Useful as 3-tier triage; the safe auto-clear volume stays small until the model's discrimination improves (v5 not yet decisively ahead of v3 at ep8).

---
*Computed by cascade inference over `runs/maxvit384_3class_clinical_v5/eval_ep8/test_21_27/series_probs_test.csv` (per-study, mean). T1/T2 set on the original val (`.../test_21_27/series_probs_val.csv`), applied to the June 21–27 week. No recalibration. v5 ep8 snapshot.*
