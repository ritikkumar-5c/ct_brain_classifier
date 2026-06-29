# ct_brain v1 — Detailed Training Report (MaxViT-MIL, 3-class)

> **Run:** `runs/maxvit384_3class_clinical`
> **Date:** 2026-06-29 (held-out test evaluated post-hoc via `eval_operating_points.py`)
> **Status:** COMPLETE — stopped at epoch 17, best checkpoint = **epoch 15**.
> **Task:** study-level 3-class CT-brain — normal / near_normal / abnormal.
> **Backbone:** `maxvit_tiny_tf_384.in1k` + gated-attention MIL pooling. **Data:** train 13,114 / val 2,623 / test 2,623; image_size 384; 3 HU windows.

---

## 1. Headline

| metric | val (best, ep15) | **held-out TEST** |
|---|--:|--:|
| balanced_acc | 0.665 | **0.632** |
| AUC (macro-OVR) | 0.852 | **0.837** |
| accuracy | 0.678 | 0.653 |

The first 3-class run. Plateaued at val balanced_acc ~0.66 with AUC capped ~0.85; **val→test drop is notable (balAcc −0.033, AUC −0.015)** — larger than v3's, reflecting more overfitting/val-tuning. Baseline against which v2/v3 were tuned.

---

## 2. Configuration

From-ImageNet init. `loss=cost_sensitive`, `monitor=balanced_acc`, `batch_size=16`, `image_size=384`, `grad_checkpoint=true`, `use_amp=true`, gated-attention MIL, 3 HU windows.

| hyperparameter | v1 value | note |
|---|--:|---|
| `cost_miss_abnormal / near` | 5.0 / 3.0 | heavy asymmetry (crushed normal recall) |
| `cost_ce_lambda` | 0.3 | CE blend |
| `lr` (peak) | 3e-4 | highest of the three runs |
| `epochs` (cosine T_max) | 50 | |
| `warmup_epochs` | 2 | |
| `weight_decay` | 1e-4 | light |
| `dropout` (head/drop_rate) | 0.04 | paper value, very light |
| `drop_path` (stochastic depth) | **0.0** | OFF — backbone unregularized |
| `label_smoothing` | 0.05 | mild |
| `window_jitter` | 0.05 | |
| `train_slices_per_study` / eval | 96 / 96 | no slice subsampling |
| `early_stop_patience` | 10 | |

---

## 3. Training dynamics (per-epoch)

| ep | trBal | vaBal | trL | vaL | trAUC | vaAUC | rNorm | rNear | rAbn | nnSens | nSpec |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 5 | 0.573 | 0.620 | 0.86 | 0.79 | 0.819 | 0.844 | 0.39 | 0.71 | 0.75 | 0.96 | 0.39 |
| 8 | 0.624 | 0.649 | 0.80 | 0.82 | 0.847 | 0.846 | 0.51 | 0.74 | 0.70 | 0.94 | 0.51 |
| 10 | 0.667 | 0.661 | 0.75 | 0.88 | 0.866 | 0.847 | 0.66 | 0.56 | 0.77 | 0.90 | 0.66 |
| 12 | 0.692 | 0.659 | 0.71 | 0.86 | 0.882 | 0.847 | 0.59 | 0.73 | 0.66 | 0.93 | 0.59 |
| **15** | 0.745 | **0.665** | 0.61 | 0.84 | 0.910 | 0.852 | 0.57 | 0.76 | 0.66 | 0.93 | 0.57 |
| 17 | 0.774 | 0.659 | 0.56 | **0.94** | 0.926 | 0.841 | 0.65 | 0.59 | 0.74 | 0.89 | 0.65 |

- **Slow climb to a ~0.66 plateau** by ep8–10; best at ep15 but barely above ep10.
- **Violent per-class oscillation** — `recall_normal` swings 0.04 → 0.66 (e.g. 0.04 @ep2, 0.09 @ep7). The heavy 5/3 cost matrix sacrifices the normal class to avoid missing pathology, capping balanced_acc and destabilizing recall.
- **Gradual overfit** — train balanced_acc climbs to 0.774 while val stalls ~0.66; val_loss rises to 0.94 by ep17.

---

## 4. Held-out TEST results (best.pt = ep15)

| metric | value |
|---|--:|
| accuracy | 0.653 |
| balanced_acc | 0.632 |
| AUC (macro-OVR) | 0.837 |
| f1 (macro) | 0.640 |
| recall normal / near_normal / abnormal | 0.507 / 0.727 / 0.662 |
| not_normal_sensitivity (argmax) | 0.932 |
| normal_specificity (argmax) | 0.507 |

Low normal recall (0.507) on test — the cost-matrix bias against the normal class carries through to held-out data.

---

## 5. Screening operating points — normal vs not-normal (threshold on VAL → applied to TEST)

| target sens | threshold | TEST sensitivity | specificity | precision | FAR (1−spec) | TP / FP / FN / TN |
|--:|--:|--:|--:|--:|--:|--|
| 0.95 | 0.284 | 0.949 | 0.425 | 0.863 | 0.575 | 1972 / 314 / 105 / 232 |
| 0.96 | 0.175 | 0.958 | 0.368 | 0.852 | 0.632 | 1989 / 345 / 88 / 201 |
| 0.97 | 0.111 | 0.967 | 0.310 | 0.842 | 0.690 | 2009 / 377 / 68 / 169 |
| 0.98 | 0.048 | 0.974 | 0.194 | 0.821 | 0.806 | 2023 / 440 / 54 / 106 |
| 0.99 | 0.019 | 0.988 | 0.117 | 0.810 | 0.883 | 2053 / 482 / 24 / 64 |

⚠️ Precision is inflated by the enriched 79% not-normal test prevalence; at realistic screening prevalence it would be much lower.

---

## 6. Verdict
v1 established the task and the **~0.85 AUC ceiling**, but is the weakest of the three on held-out test (balAcc 0.632) and suffers violent class oscillation from the heavy 5/3 cost matrix and an unregularized backbone (`drop_path=0`). Superseded by v2 (eased cost + light reg) and v3 (proper backbone regularization, best overall). See `run_comparison_v1_v2_v3.md` and `v3_detailed_report.md`.

---
*Metrics from TensorBoard event files (training) and `eval_operating_points.py` (held-out test, computed post-hoc — v1 did not run `test()` at training time).*
