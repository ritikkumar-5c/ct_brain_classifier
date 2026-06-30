# v3 Cascaded (Hierarchical) Classification — Stage 1 @95% / Stage 2 @95%

> **Model:** ct_brain v3 (`runs/maxvit384_3class_clinical_v3`), MaxViT-MIL, 3-class. **No retraining** — cascade runs on the existing softmax probabilities.
> **Date:** 2026-06-30
> **Operating point:** both stages set to **95% target sensitivity**. Abnormal-first: decide the critical class, then split the rest.
> **Scoring unit:** per-patient, mean over series. **Thresholds set on val, applied to test** (no leakage).
> Companions: `v3_cascaded_classification.md` (target sweep), `v3_objective_comparison.md`, the two `v3_production_deployment_pos(*).md` reports.

---

## 1. Method — two thresholded stages on one softmax

One forward pass → `P(normal), P(near), P(abn)` (sum to 1). Two sequential thresholds:

| Stage | Score | Rule | Threshold (val, 95% target) |
|---|---|---|--:|
| **1 — isolate abnormal** | `s₁ = P(abn)` | `s₁ ≥ T1` → **ABNORMAL** (escalate); else → Stage 2 | **T1 = 0.025** |
| **2 — split the rest** | `s₂ = Padj(near) = P(near) / (P(near)+P(normal))` | `s₂ ≥ T2` → **NEAR_NORMAL** (light review); else → **NORMAL** (auto-clear) | **T2 = 0.029** |

- **Re-normalize in Stage 2** (divide by `P(near)+P(normal)`): Stage 1 consumed `P(abn)` mass, so raw `P(near)` is deflated. The re-normalization sharpens the boundary — **near-vs-normal AUC 0.75 (raw) → 0.84 (re-normalized)**.
- **Stage 2 polarity: Positive = near_normal, Negative = normal** — protect near_normal recall (calling early-disease "Normal" is the costly error).

---

## 2. Stage results (test)

**Stage 1 — abnormal vs (normal + near), T1 = 0.025**

| metric | value |
|---|--:|
| abnormal sensitivity | **94.9% (654/689)** |
| escalated to doctor | **1,163 (73% of 1,583)** |
| — of which false positives | 509 (102 normal + 407 near) |
| passed to Stage 2 | 420 (234 normal / 151 near / **35 abnormal**) |
| abnormals lost to Stage 2 | **35** (permanently mislabeled) |

**Stage 2 — near vs normal on the 420 pass-through, T2 = 0.029**

| metric | value |
|---|--:|
| near_normal sensitivity | **95.4% (144/151)** |
| normal specificity | **22.2% (52/234)** |

Stage 2 hits its 95% near-recall target, but normal specificity is only 22% — to catch 95% of near_normal it calls **78% of true normals "near_normal"** too. That is the weak normal↔near boundary (AUC 0.75 raw / 0.84 re-normalized) capping the auto-clear yield.

---

## 3. End-to-end triage (95% / 95%)

Three buckets, counts as **normal / near / abnormal** (true class):

| Bucket | Total (% worklist) | normal | near | abnormal |
|---|--:|--:|--:|--:|
| **ESCALATE → doctor** | 1,163 (73%) | 102 | 407 | **654** ✓ |
| **LIGHT review** | 353 (22%) | 182 | 144 | 27 |
| **AUTO-CLEAR normal** | 67 (4%) | 52 | 7 | **8** ⚠️ |

- **Abnormal safety:** 654/689 abnormals escalated; **35 not escalated** (27 → light review, **8 auto-cleared as normal** — the dangerous misses).
- **Auto-clear yield:** only **67 studies** (4% of worklist) auto-cleared, of which **52 truly normal, 7 near, 8 abnormal**. Negative precision of the auto-clear bucket = 52/67 = **78%** — i.e. ~22% of what's auto-cleared is *not* normal.
- **Workload:** 73% of studies still go to a radiologist; another 22% to light review. The cascade relieves little at this safety level.

---

## 4. As a labeler (95/95 vs flat argmax)

| | normal recall | near recall | abn recall | balanced acc |
|---|--:|--:|--:|--:|
| **Cascade 95/95** | 15.5% | 25.8% | **94.9%** | 45.4% |
| Flat argmax | 69.9% | 65.8% | 70.4% | **68.7%** |

The cascade is far weaker *as a labeler* — by design. It is tuned to not miss abnormal and trades away normal/near accuracy. **Use it to route, not to label.**

---

## 5. Findings

| Item | Finding |
|---|---|
| **Structure** | Sound and clinically aligned; re-normalization is real (Stage-2 AUC 0.75 → 0.84); Pos=near_normal is the correct polarity. |
| ⚠️ **Low relief at 95%** | T1=0.025 still escalates **73% of the worklist**; auto-clear bucket is only 67 studies (4%). |
| ⚠️ **Auto-clear unsafe** | 8 of 67 auto-cleared are abnormal (NPV 78%); plus 27 more abnormals sit in light-review. |
| ⚠️ **Cascading error** | The 35 abnormals not escalated at Stage 1 can never be recovered downstream. |
| **Calibration** | Label smoothing (α=0.1) compresses probabilities (cliff ≈0.93), so T1/T2 sit low (0.025 / 0.029) — expected. Temperature-scale logits + plot per-stage histograms for stable operating points. |
| **Bottleneck** | Both boundaries (abn-vs-rest 0.88, near-vs-normal 0.84), not the inference structure. Same fixes as the flat reports: higher AUC, calibration, the **v5 rule-out head** (`ct-brain-autoreport-strategy`). |

**Bottom line.** At 95%/95% the cascade is safe-leaning (655/689 abnormals escalated or light-reviewed) but low-yield: it auto-clears only 4% of the worklist and that bucket is 22% impure. It is a routing layer that inherits v3's AUC ceiling, not a way around it.

---
*Computed by `eval_cascade.py --s1-targets 0.95 --s2-target 0.95` over `series_probs_{val,test}.csv` (patient-mean, thresholds set on val → measured on test). No leakage.*
