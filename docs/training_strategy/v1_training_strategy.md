# v1 Training Strategy Report — MaxViT-MIL, 3-class CT Brain

> **Run:** `runs/maxvit384_3class_clinical`
> **Task:** study-level 3-class classification — `normal` / `near_normal` / `abnormal`
> **Launcher:** `run_watchdog.sh` · **Config:** `config.py` (paper defaults)
> **Init:** from-ImageNet (fresh fine-tune)
> **Status:** COMPLETE — stopped at ep17, best checkpoint = **ep15**
> **Role in the lineage:** the **baseline** — establishes the task, the recipe, and the ~0.85 AUC ceiling against which v2/v3/v5 were tuned

---

## 1. Strategic Objective

v1 is the **first 3-class run** — a faithful adaptation of the paper's MaxViT recipe
(Qari & Thafar, Table 3) to a *study-level, 3-class* CT-brain task on DICOM input.

**Goal:** establish an honest baseline — does a MaxViT + MIL model discriminate
normal / near_normal / abnormal at the study level, and where does it plateau?

There is no prior version to compare against; every later run is tuned relative to what
this baseline revealed:
- a **~0.85 macro-AUC ceiling**, and
- **violent per-class recall oscillation** driven by an aggressive cost matrix.

---

## 2. Data & Bag Construction

Patient-grouped 70/15/15 split — every series of a patient stays in one split (no leakage).

| split | normal | near_normal | abnormal | total series |
|---|--:|--:|--:|--:|
| train | 2,727 | 4,712 | 5,675 | 13,114 |
| val | 546 | 942 | 1,135 | 2,623 |
| test | 546 | 942 | 1,135 | 2,623 |

| Parameter | Value | Objective |
|---|--:|---|
| `image_size` | 384 | native `maxvit_tiny_tf_384` input |
| HU windows | brain / subdural / bone | 3 clinical windows → 3 channels (vs replicated grayscale) |
| `train_slices_per_study` | 96 | full bag while training (no subsampling) |
| `max_slices_per_study` | 96 | full bag at val/test |

---

## 3. Model

| Component | Value | Objective |
|---|--:|---|
| backbone | `maxvit_tiny_tf_384.in1k` | pretrained hybrid conv+attention encoder (paper primary) |
| MIL pooling | gated-attention (Ilse et al.) | slices → one study embedding; learns which slices matter |
| `mil_attn_dim` | 256 | attention hidden dim |
| `num_classes` | 3 | normal / near_normal / abnormal |
| `pretrained` | true | ImageNet init, full fine-tune |

---

## 4. Regularization — minimal (paper defaults)

v1 uses only the paper's light defaults. Crucially the **backbone is unregularized**
(`drop_path=0`) — a gap that only v3 closes.

| Parameter | Value | Objective |
|---|--:|---|
| `dropout` (head) | 0.04 | paper value — very light head reg |
| `weight_decay` | 1e-4 | light L2 |
| `label_smoothing` | 0.05 | mild target softening on noisy boundary |
| `window_jitter` | 0.05 | light CT-correct intensity aug |
| **`drop_path`** (stochastic depth) | **0.0 (OFF)** | ⚠️ backbone unregularized |

Standard augmentations on: `aug_hflip`, `aug_rotation_deg=15`, `aug_crop_scale_min=0.85`.

---

## 5. Optimization Schedule

| Parameter | Value | Objective |
|---|--:|---|
| optimizer | Adam | paper Table 3 |
| `lr` (peak) | 3e-4 | highest of all runs (paper mid-grid) |
| `epochs` (cosine `T_max`) | 50 | long horizon — LR barely decays over the actual ~17 epochs |
| `warmup_epochs` | 2 | short LR ramp |
| `weight_decay` | 1e-4 | (see §4) |
| `early_stop_patience` | 10 | epochs w/o monitor improvement |
| `batch_size` | 16 | studies-per-batch |

**Efficiency:** `grad_checkpoint=true`, `use_amp=true`, `length_bucketing=true`.

---

## 6. Loss — `cost_sensitive`

**Objective:** minimize expected *clinical* cost — under-calling pathology as `normal`
is the dangerous error, so it is priced highest. v1 uses the **heaviest asymmetry** of all runs.

```
L = E_j[ p_j · C[true, j] ]  +  λ_ce · CE(logits, target)
```

**Cost matrix C[true, pred]:**

| true ↓ / pred → | normal | near_normal | abnormal |
|---|--:|--:|--:|
| normal | 0 | 1 | 1 |
| near_normal | **3.0** | 0 | 1 |
| abnormal | **5.0** | 1 | 0 |

| Parameter | Value | Objective |
|---|--:|---|
| `cost_miss_abnormal` | **5.0** | penalize abnormal→normal (worst miss) |
| `cost_miss_near_normal` | **3.0** | penalize near_normal→normal |
| `cost_ce_lambda` | 0.3 | CE stabilizer weight |
| `label_smoothing` | 0.05 | inside both terms |

⚠️ This heavy 5.0/3.0 asymmetry is what **sacrificed the normal class** (see §8) and motivated v2's easing.

---

## 7. Metrics & Selection

| Metric | Role | Objective |
|---|---|---|
| **`balanced_acc`** | **monitor / checkpoint select** | mean per-class recall — can't be gamed by over-calling |
| AUC (macro-OVR) | discrimination quality | class-threshold-independent separability |
| per-class recall | balance check | expose the normal-class sacrifice |
| `not_normal_sensitivity` | screening safety | fraction of pathology flagged |
| `target_sensitivity` = 0.95 | operating-point floor | threshold on val → applied to test |

**Selection rule:** highest val `balanced_acc` → `best.pt` (= ep15).

---

## 8. Outcome

| metric | val (best, ep15) | held-out TEST |
|---|--:|--:|
| balanced_acc | 0.665 | **0.632** |
| AUC (macro-OVR) | 0.852 | **0.837** |
| accuracy | 0.678 | 0.653 |

- **Slow climb to a ~0.66 plateau** by ep8–10; best at ep15 barely above ep10.
- **Violent per-class oscillation** — `recall_normal` swings 0.04→0.66. The heavy 5.0/3.0 cost
  matrix sacrifices the normal class to avoid missing pathology → caps balanced_acc, destabilizes recall.
- **Gradual overfit** — train balanced_acc climbs to 0.774 while val stalls; val_loss rises to 0.94 by ep17.
- **Notable val→test drop** (balAcc −0.033, AUC −0.015) — larger than v3's, reflecting more val-tuning.

**Verdict:** established the task and the **~0.85 AUC ceiling**, but weakest of the family on held-out
test. Its two failure modes — **heavy cost matrix** (class oscillation) and **unregularized backbone**
(overfit) — directly set the agenda for v2 (ease costs) and v3 (regularize the backbone).

---
*Parameters sourced from `config.py` defaults, `run_watchdog.sh`, and `engine/losses.py`.
Results from TensorBoard event files (training) and `eval_operating_points.py` (held-out test, post-hoc).*
