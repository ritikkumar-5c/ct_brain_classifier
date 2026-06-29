# ct_brain v3 — Detailed Training Report (MaxViT-MIL, 3-class)

> **Run:** `runs/maxvit384_3class_clinical_v3`
> **Date:** 2026-06-29
> **Status:** COMPLETE — early-stopped at epoch 8, best checkpoint = **epoch 3**, held-out test evaluated.
> **Task:** study-level 3-class CT-brain — normal / near_normal / abnormal.
> **Backbone:** `maxvit_tiny_tf_384.in1k` + gated-attention MIL pooling (Ilse et al.).
> **Data:** patient-grouped split — train 13,114 / val 2,623 / test 2,623 series; image_size 384; 3 HU windows (brain/subdural/bone).

---

## 1. Headline

| metric | val (best, ep3) | **held-out TEST** |
|---|--:|--:|
| balanced_acc | 0.693 | **0.673** |
| AUC (macro-OVR) | 0.859 | **0.846** |
| accuracy | — | 0.675 |

v3 is the **best of all three runs** (v1 best 0.665, v2 best 0.647) and generalizes honestly — the val→test drop is small (balAcc −0.02, AUC −0.013). The 3-class **AUC ceiling (~0.85) is confirmed on truly held-out data**.

---

## 2. Configuration

Warm-started from v2's best checkpoint (ep4, weights-only → fresh optimizer/scheduler). All else: `loss=cost_sensitive`, `monitor=balanced_acc`, `target_sensitivity=0.95`, `batch_size=16`, `image_size=384`, `grad_checkpoint=true`, `use_amp=true`, gated-attention MIL, 3 HU windows.

| hyperparameter | v3 value | role |
|---|--:|---|
| init | **warm-start ← v2 best ep4 (weights-only)** | keep learned features, fresh schedule |
| `lr` (peak) | 1.5e-4 | lower peak |
| `epochs` (cosine T_max) | 18 | so cosine actually decays to ~0 |
| `warmup_epochs` | 1 | warm weights need no long ramp |
| `weight_decay` | 1e-3 | shrink weights (capacity) |
| `dropout` (head/drop_rate) | 0.1 | zero head activations |
| **`drop_path` (stochastic depth)** | **0.15** | drop residual blocks — KEY backbone regularizer (was OFF in v1/v2) |
| `label_smoothing` | 0.1 | soften targets on noisy normal↔near boundary |
| `window_jitter` | 0.1 | CT-correct intensity aug |
| `train_slices_per_study` | **48** (random/epoch) | MIL slice-dropout regularizer + ~2× faster |
| `max_slices_per_study` (eval) | 96 | full bag at val/test |
| `cost_miss_abnormal / near` | 3.0 / 2.0 | asymmetric cost (ease vs v1's 5/3) |
| `cost_ce_lambda` | 0.4 | CE blend (stability) |
| `early_stop_patience` | 5 | stop early; don't grind into overfit |

**Throughput:** ~5–6 s/it (~1h20m/epoch); K48 subsampling ~halved activation cost vs v1/v2. GPU ~47 GB active (~75 GB reserved high-water-mark from the K96 val pass).
**Code change required:** new `drop_path` Config field wired into `timm.create_model` in `models/build.py` (DropPath has no params → old checkpoints still load `strict=True`).

---

## 3. Training dynamics (full per-epoch)

| ep | trBal | vaBal | trL | vaL | trAUC | vaAUC | rNorm | rNear | rAbn | nnSens | nSpec |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 0 | 0.686 | 0.687 | 0.82 | 0.93 | 0.867 | 0.852 | 0.77 | 0.65 | 0.64 | 0.87 | 0.77 |
| 1 | 0.681 | 0.675 | 0.83 | 0.90 | 0.863 | 0.849 | 0.66 | 0.76 | 0.60 | 0.92 | 0.66 |
| 2 | 0.703 | 0.680 | 0.80 | 0.92 | 0.875 | 0.849 | 0.71 | 0.69 | 0.64 | 0.90 | 0.71 |
| **3** | 0.714 | **0.693** | 0.77 | 0.88 | 0.887 | **0.859** | 0.72 | 0.68 | 0.68 | 0.90 | 0.72 |
| 4 | 0.738 | 0.668 | 0.73 | 0.95 | 0.898 | 0.835 | 0.69 | 0.61 | 0.70 | 0.89 | 0.69 |
| 5 | 0.755 | 0.637 | 0.70 | 0.91 | 0.906 | 0.835 | 0.59 | 0.57 | 0.75 | 0.92 | 0.59 |
| 6 | 0.772 | 0.674 | 0.66 | 0.86 | 0.919 | 0.856 | 0.61 | 0.67 | 0.74 | 0.93 | 0.61 |
| 7 | 0.805 | 0.650 | 0.61 | 0.91 | 0.933 | 0.846 | 0.54 | 0.73 | 0.68 | 0.94 | 0.54 |
| 8 | 0.823 | 0.688 | 0.56 | 0.99 | 0.945 | 0.842 | 0.77 | 0.63 | 0.67 | 0.88 | 0.77 |

**Two phases:**
- **ep0–3 (productive):** both train_loss and val_loss fall, val balanced_acc climbs to the 0.693 peak. Well-regularized — train AUC only ~0.89 (not memorizing), per-class recall balanced & stable.
- **ep4–8 (overfitting):** train_balanced_acc climbs 0.71→0.82 and train_loss falls 0.77→0.56, while **val_loss rises to 0.99** and val balanced_acc drifts down (train–val gap widens to +0.135 by ep8). Per-class recall destabilizes again (rNorm swings 0.54–0.77).

**Takeaway:** the v3 regularizers (stochastic depth, slice subsampling, decaying LR, stronger wd/label-smoothing) **pushed the peak higher (0.693 vs v2's 0.647) and held off overfit a bit**, but did not eliminate it — overfitting still set in ~ep4. Checkpoint selection on val (best = ep3) means the deployed weights are from the point of best generalization; the later overfit never reaches `best.pt`. AUC oscillated 0.835–0.859 around the ~0.85 wall — the ep3 0.859 was a noise high, not a real break.

---

## 4. Held-out TEST results (best.pt = ep3, no leakage)

**3-class (argmax):**
| metric | value |
|---|--:|
| accuracy | 0.675 |
| balanced_acc | 0.673 |
| AUC (macro-OVR) | 0.846 |
| f1 (macro) | 0.669 |
| recall normal / near_normal / abnormal | 0.670 / 0.659 / 0.690 |
| not_normal_sensitivity (argmax) | 0.890 |
| normal_specificity (argmax) | 0.670 |

Per-class recall is balanced (no class sacrificed); val→test generalization is honest.

**Test class distribution:** normal 546 / near_normal 942 / abnormal 1135 → not-normal prevalence **79.2%** (enriched, NOT screening-realistic).

---

## 5. Screening operating points — SERIES-level (threshold set on VAL, applied to TEST)

Score = `1 − P(normal)`; threshold maximizes specificity subject to the sensitivity floor. Units = 2,623 test series.

| target sens | threshold | Not-Normal Sensitivity | Not-Normal Specificity | Not-Normal Precision | Not-Normal FAR (1−spec) | TP / FP / FN / TN |
|--:|--:|--:|--:|--:|--:|--|
| 0.95 | 0.179 | 0.941 | 0.452 | 0.867 | 0.548 | 1955 / 299 / 122 / 247 |
| 0.96 | 0.128 | 0.951 | 0.399 | 0.858 | 0.601 | 1976 / 328 / 101 / 218 |
| 0.97 | 0.090 | 0.962 | 0.342 | 0.848 | 0.658 | 1999 / 359 / 78 / 187 |
| **0.98** | 0.052 | **0.977** | **0.244** | **0.831** | **0.756** | 2030 / 413 / 47 / 133 |
| 0.99 | 0.030 | 0.986 | 0.143 | 0.814 | 0.857 | 2048 / 468 / 29 / 78 |

*(Threshold chosen on val to meet each sensitivity floor, applied to the held-out test set. Counts exact; TEST = 2,077 not-normal / 546 normal. Generated by `eval_operating_points.py`.)*

**Sensitivity↔specificity trade-off is steep:** raising the floor 95%→98% nearly **halves specificity** (0.452 → 0.244) — at 98% sensitivity the model false-flags 76% of normals (only 133 of 546 normals pass clean). This is the practical consequence of the ~0.85 AUC ceiling: no threshold yields both high sensitivity and a tolerable false-alarm load.

⚠️ **Prevalence caveat:** precision 0.87 is inflated by the enriched 79% not-normal test prevalence. At a realistic screening prevalence (~20% not-normal), the same sensitivity/specificity gives **precision ≈ 0.30** — most alarms would be false. For deployment scores, evaluate at the true clinical prevalence.

---

## 6. Patient-level evaluation (series → patient aggregation)

The test set is **2,623 series from 1,583 patients** (926 patients have ≥2 series, up to 8); all series of a patient share the same label. Series-level metrics therefore over-weight multi-series patients and treat correlated series as independent. Aggregating each patient's series to one decision is the honest deployment unit. Two aggregators compared (`eval_patient_level.py`):
- **mean** — average the softmax across a patient's series.
- **max** — take the most-pathological series (lowest `P(normal)`) = "flag the patient if *any* series is suspicious".

### 3-class metrics (held-out TEST)
| eval level | units | acc | balAcc | AUC | rNorm | rNear | rAbn |
|---|--:|--:|--:|--:|--:|--:|--:|
| series | 2,623 | 0.675 | 0.673 | 0.846 | 0.670 | 0.659 | 0.690 |
| **patient [mean]** | 1,583 | **0.687** | **0.687** | **0.854** | 0.699 | 0.658 | 0.704 |
| patient [max] | 1,583 | 0.680 | 0.671 | 0.852 | 0.631 | 0.665 | 0.717 |

**Patient-level is slightly better, and `mean` wins** — aggregation averages out per-series noise (a mini-ensemble), so the per-series numbers were *not* inflated by correlated samples. `max` shifts toward pathology (normal recall 0.699→0.631), so `mean` is the better aggregator. **Recommendation: deploy with per-patient mean aggregation.**

### Screening operating points — PATIENT-level [mean] (threshold on VAL → TEST; 1,583 patients)
| target sens | threshold | Not-Normal Sensitivity | Not-Normal Specificity | Not-Normal Precision | Not-Normal FAR (1−spec) | TP / FP / FN / TN |
|--:|--:|--:|--:|--:|--:|--|
| 0.95 | 0.208 | 0.945 | 0.488 | 0.873 | 0.512 | 1178 / 172 / 69 / 164 |
| 0.96 | 0.152 | 0.953 | 0.420 | 0.859 | 0.580 | 1188 / 195 / 59 / 141 |
| 0.97 | 0.105 | 0.966 | 0.357 | 0.848 | 0.643 | 1204 / 216 / 43 / 120 |
| 0.98 | 0.066 | 0.978 | 0.289 | 0.836 | 0.711 | 1219 / 239 / 28 / 97 |
| 0.99 | 0.038 | 0.986 | 0.196 | 0.820 | 0.804 | 1230 / 270 / 17 / 66 |

### Screening operating points — PATIENT-level [max] (threshold on VAL → TEST; 1,583 patients)
| target sens | threshold | Not-Normal Sensitivity | Not-Normal Specificity | Not-Normal Precision | Not-Normal FAR (1−spec) | TP / FP / FN / TN |
|--:|--:|--:|--:|--:|--:|--|
| 0.95 | 0.235 | 0.947 | 0.458 | 0.866 | 0.542 | 1181 / 182 / 66 / 154 |
| 0.96 | 0.175 | 0.957 | 0.414 | 0.858 | 0.586 | 1194 / 197 / 53 / 139 |
| 0.97 | 0.116 | 0.968 | 0.351 | 0.847 | 0.649 | 1207 / 218 / 40 / 118 |
| 0.98 | 0.072 | 0.979 | 0.286 | 0.836 | 0.714 | 1221 / 240 / 26 / 96 |
| 0.99 | 0.039 | 0.987 | 0.182 | 0.817 | 0.818 | 1231 / 275 / 16 / 61 |

**Across all three granularities the picture holds:** specificity at clinical sensitivity stays low (patient-mean: 49% @95%, 29% @98%) — the ~0.85 AUC ceiling means there's no operating point with both high sensitivity and a tolerable false-alarm load. Per-series probabilities dumped to `series_probs_{val,test}.csv` for free re-aggregation.

---

## 7. Verdict & next steps

**v3 is the strongest 3-class model and is deployment-*candidate* quality for a triage/assist role**, validated on held-out test (balanced_acc 0.673 / AUC 0.846). But:
- At a clinical high-sensitivity operating point (95% sens), **specificity is only ~45%** → heavy false-alarm load.
- The **3-class AUC ceiling (~0.85) is confirmed real** and reproduces across all hyperparameter regimes → it's a label/boundary limit (noisy normal↔near_normal), not an optimization problem. More 3-class tuning will not move it.

**Recommended next moves (highest leverage first):**
1. **2-class reframe (v4)** — normal vs not-normal. Removes the noisy boundary capping AUC; also the more deployable screening framing.
2. **Ensemble + TTA** — cheap, reliable few-point AUC bump using existing checkpoints.
3. **Near_normal label cleaning** (confident learning) — addresses the root cause of the ceiling.
4. Skip further LR/dropout/cost tuning — direct evidence shows it doesn't move AUC.

---
*Metrics from the run's TensorBoard event files and `test()` output. Operating-point sweep via `eval_operating_points.py`.*
