# v3 Cascaded (Hierarchical) Classification — Stage 1 @98% / Stage 2 @98%

> **Model:** ct_brain v3 (`runs/maxvit384_3class_clinical_v3`), MaxViT-MIL, 3-class. **No retraining** — cascade runs on the existing softmax probabilities.
> **Date:** 2026-06-30
> **Operating point:** both stages set to **98% target sensitivity**. Abnormal-first: decide the critical class, then split the rest.
> **Scoring unit:** per-patient, mean over series. **Thresholds set on val, applied to test** (no leakage).
> Companions: `v3_cascaded_classification_95_95.md`, `v3_cascaded_classification_95_90.md`, `v3_objective_comparison.md`, the two `v3_production_deployment_pos(*).md` reports.

---

## 1. Method — two thresholded stages on one softmax

One forward pass → `P(normal), P(near), P(abn)` (sum to 1). Two sequential thresholds:

| Stage | Score | Rule | Threshold (val, 98% target) |
|---|---|---|--:|
| **1 — isolate abnormal** | `s₁ = P(abn)` | `s₁ ≥ T1` → **ABNORMAL** (escalate); else → Stage 2 | **T1 = 0.012** |
| **2 — split the rest** | `s₂ = Padj(near) = P(near) / (P(near)+P(normal))` | `s₂ ≥ T2` → **NEAR_NORMAL** (light review); else → **NORMAL** (auto-clear) | **T2 = 0.017** |

- **Re-normalize in Stage 2** (divide by `P(near)+P(normal)`): Stage 1 consumed `P(abn)` mass, so raw `P(near)` is deflated. Re-normalization sharpens the boundary — **near-vs-normal AUC 0.75 (raw) → 0.84 (re-normalized)**.
- **Stage 2 polarity: Positive = near_normal, Negative = normal** — protect near_normal recall.

---

## 2. Stage results (test)

**Stage 1 — abnormal vs (normal + near), T1 = 0.012**

| metric | value |
|---|--:|
| abnormal sensitivity | **98.0% (675/689)** |
| escalated to doctor | **1,321 (83% of 1,583)** |
| — of which false positives | 646 (162 normal + 484 near) |
| passed to Stage 2 | 262 (174 normal / 74 near / **14 abnormal**) |
| abnormals lost to Stage 2 | **14** (permanently mislabeled) |

**Stage 2 — near vs normal on the 262 pass-through, T2 = 0.017**

| metric | value |
|---|--:|
| near_normal sensitivity (target 98%) | **94.6% (70/74)** |
| normal specificity | **13.8% (24/174)** |

⚠️ The Stage-2 threshold was set for 98% near-recall **on val**, but achieves only **94.6% on test** — a val→test transfer gap amplified by the tiny pass-through (only 74 near_normal patients survive Stage 1). At 98% target, specificity collapses to **14%**: catching ~95% of near_normal forces 86% of true normals to be called "near_normal" too. This is the weak normal↔near boundary (AUC 0.75 raw / 0.84 re-normalized) at its limit.

---

## 3. End-to-end triage (98% / 98%)

Three buckets, counts as **normal / near / abnormal** (true class):

| Bucket | Total (% worklist) | normal | near | abnormal |
|---|--:|--:|--:|--:|
| **ESCALATE → doctor** | 1,321 (83%) | 162 | 484 | **675** ✓ |
| **LIGHT review** | 233 (15%) | 150 | 70 | 13 |
| **AUTO-CLEAR normal** | 29 (2%) | 24 | 4 | **1** ⚠️ |

- **Abnormal safety:** 675/689 escalated; **14 not escalated** (13 → light review, **1 auto-cleared as normal** — the dangerous miss). This is the safest of the three operating points.
- **Auto-clear yield:** only **29 studies** (2% of worklist), of which **24 truly normal, 4 near, 1 abnormal**. Negative precision = 24/29 = **83%** — the highest NPV of the three points, but on a tiny bucket.
- **Workload:** 83% to radiologist, 15% to light review — the cascade relieves almost nothing at this safety level.

---

## 4. As a labeler (98/98 vs flat argmax)

| | normal recall | near recall | abn recall | balanced acc |
|---|--:|--:|--:|--:|
| **Cascade 98/98** | 7.1% | 12.5% | **98.0%** | 39.2% |
| Flat argmax | 69.9% | 65.8% | 70.4% | **68.7%** |

Weakest *as a labeler* of the three operating points — by design; pushing both stages to 98% sensitivity maximizes escalation and minimizes everything else. **Use it to route, not to label.**

---

## 5. Operating-point comparison (the three cascade reports)

| | Escalate | Light | Auto-clear (vol) | Auto-clear NPV | abn auto-cleared as normal | abn not escalated |
|---|--:|--:|--:|--:|--:|--:|
| **95% / 90%** | 73% | 20% | 103 (7%) | 78% | 9 | 35 |
| **95% / 95%** | 73% | 22% | 67 (4%) | 78% | 8 | 35 |
| **98% / 98%** | 83% | 15% | 29 (2%) | **83%** | **1** | **14** |

Tightening both stages to 98% is the **safest** (1 abnormal in auto-clear, 14 unescalated, NPV 83%) but the **lowest-yield** (auto-clears 2% of the worklist, escalates 83%). The 95/90 point is the highest-volume but leaks 9 abnormals into auto-clear.

---

## 6. Findings

| Item | Finding |
|---|---|
| **Structure** | Sound and clinically aligned; re-normalization is real (Stage-2 AUC 0.75 → 0.84); Pos=near_normal is correct. |
| ⚠️ **Near-total escalation** | 98% abnormal-sensitivity forces T1=0.012 → escalate **83% of the worklist**; auto-clear is only 2%. |
| ⚠️ **Stage-2 transfer gap** | 98% near-recall on val → 94.6% on test; the pass-through (74 near patients) is too small to set a stable 98% threshold. |
| ⚠️ **Cascading error** | The 14 abnormals not escalated at Stage 1 can never be recovered downstream. |
| **Calibration** | Label smoothing (α=0.1) compresses probabilities (cliff ≈0.93); T1/T2 sit very low (0.012 / 0.017). Temperature-scale logits + plot per-stage histograms for stable operating points. |
| **Bottleneck** | Both boundaries (abn-vs-rest 0.88, near-vs-normal 0.84), not the inference structure. Same fixes as the flat reports: higher AUC, calibration, the **v5 rule-out head** (`ct-brain-autoreport-strategy`). |

**Bottom line.** At 98%/98% the cascade is the safest configuration (only 1 abnormal auto-cleared, NPV 83%) but auto-clears just 2% of the worklist while sending 83% to a doctor. It is a high-safety, near-zero-relief routing point that fully inherits v3's AUC ceiling.

---
*Computed by `eval_cascade.py --s1-targets 0.98 --s2-target 0.98` over `series_probs_{val,test}.csv` (patient-mean, thresholds set on val → measured on test). No leakage.*
