# v3 Cascaded (Hierarchical) Classification — Stage 1 @95% / Stage 2 @90%

> **Model:** ct_brain v3 (`runs/maxvit384_3class_clinical_v3`), MaxViT-MIL, 3-class. **No retraining** — cascade runs on the existing softmax probabilities.
> **Date:** 2026-06-30
> **Operating point:** Stage 1 = **95%** abnormal-sensitivity, Stage 2 = **90%** near-sensitivity. Abnormal-first: decide the critical class, then split the rest.
> **Scoring unit:** per-patient, mean over series. **Thresholds set on val, applied to test** (no leakage).
> Companions: `v3_cascaded_classification_95_95.md`, `v3_cascaded_classification_98_98.md`, `v3_objective_comparison.md`, the two `v3_production_deployment_pos(*).md` reports.

---

## 1. Method — two thresholded stages on one softmax

One forward pass → `P(normal), P(near), P(abn)` (sum to 1). Two sequential thresholds:

| Stage | Score | Rule | Threshold (val) |
|---|---|---|--:|
| **1 — isolate abnormal** | `s₁ = P(abn)` | `s₁ ≥ T1` → **ABNORMAL** (escalate); else → Stage 2 | **T1 = 0.025** (95% target) |
| **2 — split the rest** | `s₂ = Padj(near) = P(near) / (P(near)+P(normal))` | `s₂ ≥ T2` → **NEAR_NORMAL** (light review); else → **NORMAL** (auto-clear) | **T2 = 0.047** (90% target) |

- **Re-normalize in Stage 2** (divide by `P(near)+P(normal)`): Stage 1 consumed `P(abn)` mass, so raw `P(near)` is deflated. Re-normalization sharpens the boundary — **near-vs-normal AUC 0.75 (raw) → 0.84 (re-normalized)**.
- **Stage 2 polarity: Positive = near_normal, Negative = normal** — protect near_normal recall.
- **Why Stage 2 = 90% (vs 95%):** relaxing the near-recall target lifts `T2` (0.029 → 0.047), which lets *more true normals* clear the bar — trading a little near_normal recall for more auto-clear volume.

---

## 2. Stage results (test)

**Stage 1 — abnormal vs (normal + near), T1 = 0.025** *(identical to the 95/95 report — Stage 1 is unchanged)*

| metric | value |
|---|--:|
| abnormal sensitivity | **94.9% (654/689)** |
| escalated to doctor | **1,163 (73% of 1,583)** |
| — of which false positives | 509 (102 normal + 407 near) |
| passed to Stage 2 | 420 (234 normal / 151 near / **35 abnormal**) |
| abnormals lost to Stage 2 | **35** (permanently mislabeled) |

**Stage 2 — near vs normal on the 420 pass-through, T2 = 0.047**

| metric | @90% (this report) | @95% (for reference) |
|---|--:|--:|
| near_normal sensitivity | **90.7% (137/151)** | 95.4% (144/151) |
| normal specificity | **34.2% (80/234)** | 22.2% (52/234) |

Dropping the near target to 90% raises normal specificity from 22% → **34%** — i.e. more true normals are correctly let through to auto-clear — at the cost of 7 more missed near_normals (137 vs 144 caught). Specificity is still low because the normal↔near boundary is weak (AUC 0.75 raw / 0.84 re-normalized).

---

## 3. End-to-end triage (95% / 90%)

Three buckets, counts as **normal / near / abnormal** (true class):

| Bucket | Total (% worklist) | normal | near | abnormal |
|---|--:|--:|--:|--:|
| **ESCALATE → doctor** | 1,163 (73%) | 102 | 407 | **654** ✓ |
| **LIGHT review** | 317 (20%) | 154 | 137 | 26 |
| **AUTO-CLEAR normal** | 103 (7%) | 80 | 14 | **9** ⚠️ |

- **Abnormal safety:** 654/689 escalated; **35 not escalated** (26 → light review, **9 auto-cleared as normal** — the dangerous misses).
- **Auto-clear yield:** **103 studies** (7% of worklist), of which **80 truly normal, 14 near, 9 abnormal**. Negative precision = 80/103 = **78%**.
- **vs 95/95:** auto-clear grows **67 → 103 (+54%)** and NPV holds at ~78%, but dangerous abn-as-normal ticks up **8 → 9**. The extra volume is mostly true normals plus a few more near_normals slipping through.
- **Workload:** still 73% to radiologist, 20% to light review.

---

## 4. As a labeler (95/90 vs flat argmax)

| | normal recall | near recall | abn recall | balanced acc |
|---|--:|--:|--:|--:|
| **Cascade 95/90** | 23.8% | 24.6% | **94.9%** | 47.8% |
| Flat argmax | 69.9% | 65.8% | 70.4% | **68.7%** |

Weaker *as a labeler* — by design; the cascade is tuned for abnormal safety, not balanced accuracy. **Use it to route, not to label.**

---

## 5. Findings

| Item | Finding |
|---|---|
| **Structure** | Sound and clinically aligned; re-normalization is real (Stage-2 AUC 0.75 → 0.84); Pos=near_normal is correct. |
| **90% vs 95% Stage 2** | Auto-clear volume **67 → 103 (+54%)**, normal specificity **22% → 34%**, near recall **95% → 91%**, dangerous abn-as-normal **8 → 9**. A modest volume-for-safety trade. |
| ⚠️ **Low relief at 95% Stage 1** | T1=0.025 still escalates **73% of the worklist**; auto-clear is only 7%. |
| ⚠️ **Auto-clear unsafe** | 9 of 103 auto-cleared are abnormal (NPV 78%); 26 more abnormals sit in light review. |
| ⚠️ **Cascading error** | The 35 abnormals not escalated at Stage 1 can never be recovered downstream. |
| **Calibration** | Label smoothing (α=0.1) compresses probabilities (cliff ≈0.93); T1/T2 sit low (0.025 / 0.047). Temperature-scale logits + plot per-stage histograms for stable operating points. |
| **Bottleneck** | Both boundaries (abn-vs-rest 0.88, near-vs-normal 0.84), not the inference structure. Same fixes as the flat reports: higher AUC, calibration, the **v5 rule-out head** (`ct-brain-autoreport-strategy`). |

**Bottom line.** At 95%/90% the cascade auto-clears 7% of the worklist (up from 4% at 95/95) at the same ~78% negative precision, with one extra abnormal leaking into the cleared bucket. It remains a routing layer that inherits v3's AUC ceiling — useful for triage, not yet a safe high-volume auto-clear.

---
*Computed by `eval_cascade.py --s1-targets 0.95 --s2-target 0.90` over `series_probs_{val,test}.csv` (patient-mean, thresholds set on val → measured on test). No leakage.*
