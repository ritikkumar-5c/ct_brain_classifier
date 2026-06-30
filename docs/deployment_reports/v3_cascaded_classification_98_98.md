# v3 Cascaded (Hierarchical) Classification — abnormal-first triage on the single 3-class softmax

> **Model:** ct_brain v3 (`runs/maxvit384_3class_clinical_v3`), MaxViT-MIL, 3-class. **No retraining** — cascade runs on the existing softmax probabilities.
> **Date:** 2026-06-30
> **Idea (from the hierarchical-classification design):** the model separates **abnormal** from the rest better than it separates normal↔near_normal. So decide the *critical* class first (Stage 1), then resolve the fuzzy boundary on what's left (Stage 2). Mirrors clinical triage: rule out the severe case first.
> **Scoring unit:** per-patient, mean over series. **Thresholds set on val, applied to test** (no leakage).
> Companions: `v3_production_deployment_pos(abnormal).md`, `v3_production_deployment_pos(near-normal_abnormal).md`, `v3_objective_comparison.md`.

---

## 1. Method — two thresholded stages on one softmax

Run a single forward pass → `P(normal), P(near), P(abn)` (sum to 1). Apply two sequential thresholds:

| Stage | Score | Rule | Threshold from |
|---|---|---|---|
| **1 — isolate abnormal** | `s₁ = P(abn)` | `s₁ ≥ T1` → **ABNORMAL** (escalate); else → Stage 2 | val ROC of **abnormal vs (normal+near)** at target abnormal-sensitivity |
| **2 — split the rest** | `s₂ = Padj(near) = P(near) / (P(near)+P(normal))` | `s₂ ≥ T2` → **NEAR_NORMAL** (light review); else → **NORMAL** (auto-clear) | val ROC of **near vs normal** (Pos = near_normal) on the Stage-1 pass-through, at target near-sensitivity |

Two design points from the source discussion, both confirmed important here:

- **Re-normalize in Stage 2.** Stage 1 consumed `P(abn)` probability mass, so raw `P(near)` is deflated and unstable as a threshold. Dividing by `P(near)+P(normal)` restores a true binary score. **It measurably sharpens the boundary: near-vs-normal AUC 0.75 (raw `P(near)`) → 0.84 (re-normalized).**
- **Stage 2 polarity: Positive = near_normal, Negative = normal** (never the reverse). Missing an early-disease (near_normal) study by calling it Normal is the costly error; the threshold must protect *near_normal* recall.

---

## 2. Stage 1 — isolating abnormal (test, val-set thresholds)

This is exactly the abnormal-vs-rest boundary (AUC **0.88**). "Escalated" = predicted abnormal → radiologist.

| Target abn-sens | T1 | Achieved abn-sens (TP/Pos) | Escalated (% of worklist) | of which FP (normal+near) | Passed to Stage 2 | Abnormals lost to Stage 2 |
|--:|--:|--:|--:|--:|--:|--:|
| 95% | 0.025 | 94.9% (654/689) | 1,163 (**73%**) | 509 | 420 | 35 |
| 98% | 0.012 | 98.0% (675/689) | 1,321 (**83%**) | 646 | 262 | 14 |
| 99% | 0.004 | 99.1% (683/689) | 1,451 (**92%**) | 768 | 132 | 6 |

**The threshold collapses.** To reach 98% abnormal-sensitivity on v3, `T1` falls to **0.012** (far below 0.5 — partly the label-smoothing compression, §5). At that point Stage 1 escalates **83% of the entire worklist** to a doctor; even at 95% it escalates 73%. The abnormal-vs-rest AUC (0.88) simply isn't sharp enough to isolate abnormals at high sensitivity without sweeping most negatives along.

---

## 3. End-to-end triage (Stage 2 target near-sens = 95%)

Three buckets per patient. Counts shown as **normal / near / abnormal** (true class). The dangerous cell is **abnormal landing in AUTO-CLEAR**.

| S1 target | T1 / T2 | ESCALATE → doctor (n/ne/ab) | LIGHT review (n/ne/ab) | AUTO-CLEAR normal (n/ne/ab) | abn auto-cleared as normal | abn never escalated |
|--:|--:|--:|--:|--:|--:|--:|
| 95% | 0.025 / 0.029 | 1,163 — 102/407/654 | 353 — 182/144/27 | **67 — 52/7/8** | **8** | 35 |
| 98% | 0.012 / 0.021 | 1,321 — 162/484/675 | 218 — 138/69/11 | **44 — 36/5/3** | **3** | 14 |
| 99% | 0.004 / 0.017 | 1,451 — 236/532/683 | 106 — 79/22/5 | **26 — 21/4/1** | **1** | 6 |

**The auto-clear bucket is tiny and still unsafe.** At S1=98% only **44 studies** (2.8% of the worklist) are auto-cleared as normal, of which **36 are truly normal, 5 near, 3 abnormal**. Tightening to 99% shrinks it to 26. So the cascade clears *fewer* normals than the direct normal-rule-out report (which cleared 164 at 95% — see §4) **and** still leaks abnormals into the cleared pile — because abnormal-first routing pulls most studies into "escalate," starving the auto-clear bucket.

---

## 4. As a labeler, and vs the flat approaches

**As a hard 3-class labeler** (cascade S1=98% / S2=95% vs plain argmax on the same probs):

| | normal recall | near recall | abn recall | balanced acc |
|---|--:|--:|--:|--:|
| **Cascade** | 10.7% | 12.4% | **98.0%** | 40.3% |
| Flat argmax | 69.9% | 65.8% | 70.4% | **68.7%** |

The cascade is far worse *as a labeler* — by construction. It is tuned for one thing (don't miss abnormal) and trades away normal/near accuracy to get it. **Deploy it to route, not to label.**

**Versus the flat rule-out reports** (same model, same goal of taking studies off the worklist):

| | auto-cleared as normal (true normals) | abnormal leaked into cleared | escalated to doctor |
|---|--:|--:|--:|
| **Cascade** (S1=98/S2=95) | 36 | 3 | 83% of worklist |
| **pos(not-normal)** rule-out @95% | 164 | 19 | — (rules out normals directly) |
| **pos(abnormal)** rule-out @95% | 234 (+151 near) | 35 | — |

For the *auto-report-normals* objective, the cascade is **strictly worse on volume** than the direct normal-rule-out: abnormal-first routing escalates most studies before any normal can be cleared. The cascade's only distinct value is producing a **3-way triage** (escalate / light-review / auto-clear) in a single pass.

---

## 5. Findings & caveats

| Item | Finding |
|---|---|
| **Structure** | Sound and clinically aligned; the re-normalization step is real (sharpens Stage 2: near-vs-normal AUC 0.75 → **0.84**), and Pos=near_normal in Stage 2 is the correct polarity. |
| ⚠️ **High-sensitivity collapse** | 98% abnormal-sensitivity forces T1=0.012 → **escalate 83% of the worklist**. The cascade does *not* escape v3's AUC ceiling; it relocates it to Stage 1. |
| ⚠️ **Auto-clear yield** | Only 26–67 studies auto-cleared (2–4% of worklist), still containing **1–8 missed abnormals** — small *and* unsafe. The direct normal-rule-out clears far more (§4). |
| ⚠️ **Cascading error** | The 6–35 abnormals not escalated at Stage 1 are **permanently** mislabeled (Stage 2 can only call them normal/near). Irreversible by design. |
| **Calibration / label smoothing (α=0.1)** | Probabilities are compressed (cliff ≈0.93), so thresholds sit central (T1≈0.01–0.03) — expected, not a bug. For stable operating points, **temperature-scale the logits** and **plot per-stage score histograms** before fixing T1/T2. |
| **Bottleneck** | Model discrimination on *both* boundaries (abn-vs-rest 0.88, near-vs-normal 0.84), not the inference structure. The fixes are the same as for the flat approach: higher AUC, calibration, the **v5 rule-out head** (`ct-brain-autoreport-strategy`). |

**Bottom line.** The cascade is the right *shape* for a triage product and worth keeping as the inference framing, but on v3's current probabilities it gives no free lunch: high abnormal-sensitivity escalates almost everything, and the auto-clear tail stays small and contaminated. It is a routing layer, not an accuracy or volume win — improve the model, then the cascade improves with it.

---
*Computed by `eval_cascade.py` over `series_probs_{val,test}.csv` (patient-mean, thresholds set on val → measured on test). No leakage.*
