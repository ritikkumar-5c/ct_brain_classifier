# ct_brain v2 — Detailed Training Report (MaxViT-MIL, 3-class)

> **Run:** `runs/maxvit384_3class_clinical_v2`
> **Date:** 2026-06-29 (held-out test evaluated post-hoc via `eval_operating_points.py`)
> **Status:** STOPPED at epoch 6 (manually, to launch the warm-started v3), best checkpoint = **epoch 4**.
> **Task:** study-level 3-class CT-brain — normal / near_normal / abnormal.
> **Backbone:** `maxvit_tiny_tf_384.in1k` + gated-attention MIL pooling. **Data:** train 13,114 / val 2,623 / test 2,623; image_size 384; 3 HU windows.

---

## 1. Headline

| metric | val (best, ep4) | **held-out TEST** |
|---|--:|--:|
| balanced_acc | 0.647 | **0.623** |
| AUC (macro-OVR) | 0.837 | **0.816** |
| accuracy | 0.645 | 0.618 |

v2 eased v1's cost matrix and added light regularization, but **confirmed the ceiling rather than breaking it** — it hit ~0.65 balanced_acc 4× faster than v1 (ep4 vs ep15), then overfit sharply. The weakest held-out AUC of the three (0.816). Its best checkpoint (ep4) became the warm-start seed for v3.

---

## 2. Configuration

From-ImageNet init. `loss=cost_sensitive`, `monitor=balanced_acc`, `batch_size=16`, `image_size=384`, `grad_checkpoint=true`, `use_amp=true`, gated-attention MIL, 3 HU windows.

| hyperparameter | v2 value | change vs v1 |
|---|--:|---|
| `cost_miss_abnormal / near` | 3.0 / 2.0 | **eased** (was 5/3) |
| `cost_ce_lambda` | 0.4 | **↑** (damp oscillation) |
| `lr` (peak) | 2e-4 | **↓** (was 3e-4) |
| `epochs` (cosine T_max) | 50 | — (LR barely decayed) |
| `warmup_epochs` | 3 | **↑** |
| `weight_decay` | 5e-4 | **↑** (was 1e-4) |
| `dropout` (head/drop_rate) | 0.1 | **↑** (was 0.04) |
| `drop_path` (stochastic depth) | **0.0** | still OFF — backbone unregularized |
| `label_smoothing` | 0.05 | — |
| `window_jitter` | 0.1 | **↑** |
| `train_slices_per_study` / eval | 96 / 96 | — |
| `early_stop_patience` | 12 | **↑** |

---

## 3. Training dynamics (per-epoch, ep0–6)

| ep | trBal | vaBal | trL | vaL | trAUC | vaAUC | rNorm | rNear | rAbn | nnSens | nSpec |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 0 | 0.433 | 0.592 | 1.15 | 1.09 | 0.675 | 0.781 | 0.62 | 0.70 | 0.46 | 0.85 | 0.62 |
| 1 | 0.535 | 0.520 | 1.00 | 0.95 | 0.775 | 0.796 | 0.27 | 0.44 | 0.86 | 0.98 | 0.27 |
| 2 | 0.578 | 0.577 | 0.96 | 0.89 | 0.796 | 0.815 | 0.33 | 0.68 | 0.73 | 0.97 | 0.33 |
| 3 | 0.607 | 0.596 | 0.90 | **0.87** | 0.820 | 0.828 | 0.38 | 0.72 | 0.69 | 0.96 | 0.38 |
| **4** | 0.641 | **0.647** | 0.86 | 0.91 | 0.838 | 0.837 | 0.62 | 0.74 | 0.58 | 0.90 | 0.62 |
| 5 | 0.671 | 0.590 | 0.82 | 0.91 | 0.854 | 0.820 | 0.50 | 0.51 | 0.75 | 0.92 | 0.50 |
| 6 | 0.683 | 0.598 | 0.79 | **0.93** | 0.862 | 0.819 | 0.41 | 0.68 | 0.70 | 0.93 | 0.41 |

- **Fast rise to 0.647 @ep4, then regression.** Best at ep4; ep5–6 dropped.
- **Overfit onset by ep3** — val_loss bottomed at ep3 (0.87) then rose every epoch; train balanced_acc kept climbing (→0.683) while val fell. **Earlier/sharper than v1** despite more regularization → the v2 regularizers (head dropout, wd) were the wrong ones for a MaxViT (backbone still unregularized, `drop_path=0`).
- **Oscillation eased vs v1 but persisted** (`recall_normal` 0.27–0.62).
- **LR never decayed** — cosine `T_max=50`, so it trained near peak (1.96e-4) straight through the overfit zone.

These diagnoses motivated v3 (stochastic depth, slice subsampling, real LR decay, warm-start from this run's ep4).

---

## 4. Held-out TEST results (best.pt = ep4)

| metric | value |
|---|--:|
| accuracy | 0.618 |
| balanced_acc | 0.623 |
| AUC (macro-OVR) | 0.816 |
| f1 (macro) | 0.616 |
| recall normal / near_normal / abnormal | 0.619 / 0.701 / 0.548 |
| not_normal_sensitivity (argmax) | 0.876 |
| normal_specificity (argmax) | 0.619 |

Per-class recall more balanced than v1 (eased cost matrix), but overall discrimination is the lowest of the three on test (AUC 0.816).

---

## 5. Screening operating points — normal vs not-normal (threshold on VAL → applied to TEST)

| target sens | threshold | TEST sensitivity | specificity | precision | FAR (1−spec) | TP / FP / FN / TN |
|--:|--:|--:|--:|--:|--:|--|
| 0.95 | 0.209 | 0.945 | 0.368 | 0.850 | 0.632 | 1962 / 345 / 115 / 201 |
| 0.96 | 0.177 | 0.952 | 0.326 | 0.843 | 0.674 | 1978 / 368 / 99 / 178 |
| 0.97 | 0.147 | 0.961 | 0.282 | 0.836 | 0.718 | 1995 / 392 / 82 / 154 |
| 0.98 | 0.112 | 0.973 | 0.198 | 0.822 | 0.802 | 2020 / 438 / 57 / 108 |
| 0.99 | 0.078 | 0.987 | 0.117 | 0.810 | 0.883 | 2050 / 482 / 27 / 64 |

⚠️ Precision inflated by the enriched 79% not-normal test prevalence.

---

## 6. Verdict
v2 was a refinement that **confirmed the 3-class AUC ceiling rather than lifting it**, and it overfit faster than v1 because the added regularization didn't reach the backbone. Its lasting value: the ep4 checkpoint seeded v3's warm start, and its failure modes (LR never decaying, backbone unregularized) directly defined v3's winning changes. Weakest on held-out test (AUC 0.816). See `v3_detailed_report.md` and `run_comparison_v1_v2_v3.md`.

---
*Metrics from TensorBoard event files (training) and `eval_operating_points.py` (held-out test, computed post-hoc — v2 did not run `test()` at training time).*
