# v2 Training Strategy Report — MaxViT-MIL, 3-class CT Brain

> **Run:** `runs/maxvit384_3class_clinical_v2`
> **Task:** study-level 3-class classification — `normal` / `near_normal` / `abnormal`
> **Launcher:** `run_watchdog_v2.sh` · **Config:** `config.py`
> **Init:** from-ImageNet (fresh fine-tune)
> **Status:** STOPPED at ep6 (manually, to launch warm-started v3), best checkpoint = **ep4**
> **Role in the lineage:** refinement of **v1** → its ep4 checkpoint **seeds the v3 warm-start**

---

## 1. Strategic Objective

v1 hit a **~0.66 val balanced_acc plateau** and its `recall_normal` oscillated violently
(0.04–0.66): the heavy cost matrix (5.0/3.0) penalized calling pathology "normal" so hard that
the **normal class was sacrificed**, capping balanced_acc and destabilizing recall.

**v2's goal:** ease that asymmetry and add light regularization to stabilize per-class recall
and close a mild train/val gap — *without* touching the backbone, data, or MIL head.

| Diagnosis (v1) | v2 Response |
|---|---|
| Cost matrix crushes normal recall | Ease costs 5.0/3.0 → **3.0/2.0** (#1 lever) |
| Per-class recall oscillates | Stronger CE blend `cost_ce_lambda` 0.3→0.4 |
| Mild early overfit (train/val gap) | `weight_decay` 1e-4→5e-4, `dropout` 0.04→0.1 |
| Early collapse episodes | Lower `lr` peak 3e-4→2e-4, longer warmup 2→3 |

---

## 2. Data & Bag Construction

Same patient-grouped 70/15/15 split as v1 (no leakage).

| split | normal | near_normal | abnormal | total series |
|---|--:|--:|--:|--:|
| train | 2,727 | 4,712 | 5,675 | 13,114 |
| val | 546 | 942 | 1,135 | 2,623 |
| test | 546 | 942 | 1,135 | 2,623 |

| Parameter | Value | Objective |
|---|--:|---|
| `image_size` | 384 | native `maxvit_tiny_tf_384` input |
| HU windows | brain / subdural / bone | 3 clinical windows → 3 channels |
| `train_slices_per_study` | 96 | full bag (NOT yet subsampled — a v3 change) |
| `max_slices_per_study` | 96 | full bag at val/test |

---

## 3. Model

Unchanged from v1.

| Component | Value | Objective |
|---|--:|---|
| backbone | `maxvit_tiny_tf_384.in1k` | pretrained hybrid conv+attention encoder |
| MIL pooling | gated-attention (Ilse et al.) | slices → one study embedding |
| `mil_attn_dim` | 256 | attention hidden dim |
| `num_classes` | 3 | normal / near_normal / abnormal |

---

## 4. Regularization

v2 added light regularization — but it **targeted the head, not the backbone**, which turned
out to be its central weakness (see §8).

| Parameter | v1 → v2 | Mechanism / Objective |
|---|--:|---|
| `weight_decay` | 1e-4 → **5e-4** | L2 to close the mild train/val gap |
| `dropout` (head) | 0.04 → **0.1** | regularize the gated-attention MIL head |
| `window_jitter` | 0.05 → **0.1** | stronger CT-correct intensity aug (±10% window jitter) |
| **`drop_path`** (stochastic depth) | **0.0 (still OFF)** | ⚠️ backbone left **unregularized** — the gap v3 closes |
| `label_smoothing` | 0.05 | (unchanged) softens noisy normal↔near boundary |

Standard augmentations on: `aug_hflip`, `aug_rotation_deg=15`, `aug_crop_scale_min=0.85`.

---

## 5. Optimization Schedule

| Parameter | Value | Objective |
|---|--:|---|
| optimizer | Adam | paper Table 3 |
| `lr` (peak) | **2e-4** | lowered from v1's 3e-4 → fewer early-collapse episodes |
| `epochs` (cosine `T_max`) | 50 | ⚠️ **LR barely decays** — trains near peak through the overfit zone |
| `warmup_epochs` | **3** | smoother ramp into the lower peak |
| `weight_decay` | 5e-4 | (see §4) |
| `early_stop_patience` | **12** | give the smoother schedule room to converge |
| `batch_size` | 16 | studies-per-batch |

**Efficiency:** `grad_checkpoint=true`, `use_amp=true`, `length_bucketing=true`.

---

## 6. Loss — `cost_sensitive`

**Objective:** minimize expected *clinical* cost. v2's key change is the **eased cost matrix** —
v1's 5.0/3.0 was over-penalizing missed pathology so hard it sacrificed the normal class.

```
L = E_j[ p_j · C[true, j] ]  +  λ_ce · CE(logits, target)
```

**Cost matrix C[true, pred]** (eased vs v1):

| true ↓ / pred → | normal | near_normal | abnormal |
|---|--:|--:|--:|
| normal | 0 | 1 | 1 |
| near_normal | **2.0** | 0 | 1 |
| abnormal | **3.0** | 1 | 0 |

| Parameter | v1 → v2 | Objective |
|---|--:|---|
| `cost_miss_abnormal` | 5.0 → **3.0** | ease asymmetry crushing normal recall |
| `cost_miss_near_normal` | 3.0 → **2.0** | same, for near_normal→normal |
| `cost_ce_lambda` | 0.3 → **0.4** | stronger CE blend → damp per-class oscillation |
| `label_smoothing` | 0.05 | inside both terms |

---

## 7. Metrics & Selection

| Metric | Role | Objective |
|---|---|---|
| **`balanced_acc`** | **monitor / checkpoint select** | mean per-class recall — can't be gamed by over-calling |
| AUC (macro-OVR) | discrimination quality | class-threshold-independent separability |
| per-class recall | balance check | confirm normal isn't sacrificed (the v1 failure) |
| `not_normal_sensitivity` | screening safety | fraction of pathology flagged |
| `target_sensitivity` = 0.95 | operating-point floor | threshold on val → applied to test |

**Selection rule:** highest val `balanced_acc` → `best.pt` (= ep4).

---

## 8. Outcome

| metric | val (best, ep4) | held-out TEST |
|---|--:|--:|
| balanced_acc | 0.647 | **0.623** |
| AUC (macro-OVR) | 0.837 | **0.816** |
| accuracy | 0.645 | 0.618 |

- **Reached ~0.65 balanced_acc 4× faster than v1** (ep4 vs ep15) — the eased cost matrix worked.
- **But overfit earlier/sharper than v1** (val_loss bottomed ep3, rose every epoch after) despite
  *more* regularization → the added dropout/WD were the **wrong regularizers for a MaxViT**: the
  backbone stayed unregularized (`drop_path=0`) and the LR never decayed.
- **Weakest held-out AUC of the family** (0.816) — confirmed the ~0.85 3-class ceiling, didn't lift it.

**Verdict:** a refinement that **confirmed the ceiling rather than breaking it**. Its lasting value
is diagnostic — the failure modes here (unregularized backbone, LR never decaying) **directly defined
v3's winning changes**, and its ep4 checkpoint became the v3 warm-start seed.

---

## 9. Difference vs v1

| Dimension | v1 | v2 | Why |
|---|---|---|---|
| **Cost matrix** | 5.0 / 3.0 | **3.0 / 2.0** | ease asymmetry crushing normal recall (#1 lever) |
| **`cost_ce_lambda`** | 0.3 | **0.4** | stronger CE blend → damp per-class oscillation |
| **`lr` peak** | 3e-4 | **2e-4** | fewer early-collapse episodes |
| **`warmup_epochs`** | 2 | **3** | smoother ramp into the lower peak |
| **`weight_decay`** | 1e-4 | **5e-4** | close the mild train/val gap |
| **`dropout`** (head) | 0.04 | **0.1** | regularize the MIL head |
| **`window_jitter`** | 0.05 | **0.1** | stronger CT-correct intensity aug |
| **`early_stop_patience`** | 10 | **12** | room for the smoother schedule to converge |
| **`drop_path`** | 0.0 | 0.0 | *(unchanged — the miss that defined v3)* |
| **Init** | ImageNet | ImageNet | *(unchanged)* |
| **Best balanced_acc (val)** | 0.665 | 0.647 | faster but lower peak; then overfit |

**The through-line:** v2 fixed v1's *loss* problem (eased costs → balanced recall, 4× faster
convergence) but exposed two new ones — the **backbone was never regularized** and the **LR never
decayed**. Those two diagnoses are exactly what v3 targets next.

---
*Parameters sourced from `config.py`, `run_watchdog_v2.sh`, and `engine/losses.py`.
Results from TensorBoard event files (training) and `eval_operating_points.py` (held-out test, post-hoc).*
