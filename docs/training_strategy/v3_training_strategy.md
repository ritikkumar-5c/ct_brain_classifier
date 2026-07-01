# v3 Training Strategy Report — MaxViT-MIL, 3-class CT Brain

> **Run:** `runs/maxvit384_3class_clinical_v3`
> **Task:** study-level 3-class classification — `normal` / `near_normal` / `abnormal`
> **Launcher:** `run_watchdog_v3.sh` · **Config:** `config.py`
> **Init:** warm-start ← **v2** best (epoch 4, weights-only → fresh optimizer/scheduler)
> **Status:** COMPLETE — early-stopped at ep8, best checkpoint = **ep3**
> **Role in the lineage:** next iteration of **v2** — the **best 3-class model of the family**

---

## 1. Strategic Objective

v3 is an **anti-overfit re-training** of the v2 model. v2 peaked at ep3–4 then overfit:
`val_loss` rose every epoch while `train_loss` fell, and the LR stayed near peak because
cosine `T_max=50` barely decayed over the run.

**v3's single goal:** raise and *hold* the validation peak by adding the **right** regularizers
(backbone-level) and a properly decaying LR schedule — *without* touching the parts that already
worked (loss, cost matrix, monitor, backbone architecture).

| Diagnosis (v2) | v3 Response |
|---|---|
| Overfit onset at ep3–4 | Add stochastic depth + slice-dropout + stronger WD |
| LR stuck near peak (cosine too flat) | Cut `epochs` 50→18 so cosine decays to ~0 |
| Train/val gap widens | Stronger `weight_decay`, `label_smoothing`, `dropout` |
| Backbone never regularized (`drop_path=0`) | Turn on **stochastic depth** `drop_path`=0.15 |
| Wasted compute past the peak | `early_stop_patience` 12→5 |
| AUC capped ~0.85 | *Accepted as a label-boundary ceiling, not chased* |

---

## 2. Data & Bag Construction

Same patient-grouped 70/15/15 split as v1/v2 (no leakage).

| split | normal | near_normal | abnormal | total series |
|---|--:|--:|--:|--:|
| train | 2,727 | 4,712 | 5,675 | 13,114 |
| val | 546 | 942 | 1,135 | 2,623 |
| test | 546 | 942 | 1,135 | 2,623 |

| Parameter | Value | Objective |
|---|--:|---|
| `image_size` | 384 | native `maxvit_tiny_tf_384` input |
| HU windows | brain / subdural / bone | 3 clinical windows → 3 channels |
| `train_slices_per_study` | **48** (random/epoch) | MIL slice-dropout regularizer + ~2× faster |
| `max_slices_per_study` | 96 | full bag at val/test (no eval subsampling) |

---

## 3. Model

Architecture unchanged from v2; the only code change is wiring `drop_path` into the backbone.

| Component | Value | Objective |
|---|--:|---|
| backbone | `maxvit_tiny_tf_384.in1k` | pretrained hybrid conv+attention encoder |
| MIL pooling | gated-attention (Ilse et al.) | slices → one study embedding |
| `mil_attn_dim` | 256 | attention hidden dim |
| `num_classes` | 3 | normal / near_normal / abnormal |

---

## 4. Regularization — the core of v3

These four knobs are what separate v3 from v2.

| Parameter | v2 → v3 | Mechanism / Objective |
|---|--:|---|
| **`drop_path`** (stochastic depth) | 0.0 → **0.15** | Randomly drops residual blocks (50 DropPath layers, ramped 0.015→0.15). The **key** MaxViT regularizer, OFF in v1/v2. |
| **`train_slices_per_study`** | 96 → **48** | Bag-level dropout: random 48-slice subset per epoch → can't memorize fixed slice patterns. |
| `weight_decay` | 5e-4 → **1e-3** | Stronger L2 to close the train/val gap. |
| `dropout` (head) | 0.1 | Zeroes head/MIL activations (kept from v2). |
| `label_smoothing` | 0.05 → **0.1** | Softens targets on the noisy `normal↔near_normal` boundary. |
| `window_jitter` | 0.1 | CT-correct intensity aug (kept from v2). |

Standard augmentations on: `aug_hflip`, `aug_rotation_deg=15`, `aug_crop_scale_min=0.85`.

---

## 5. Optimization Schedule

| Parameter | Value | Objective |
|---|--:|---|
| optimizer | Adam | paper Table 3 |
| `lr` (peak) | **1.5e-4** | lowered from v2's 2e-4 — warm weights need a gentler peak |
| `epochs` (cosine `T_max`) | **18** | short horizon so cosine **actually decays to ~0** |
| `warmup_epochs` | **1** | warm-started weights don't need a long ramp |
| `weight_decay` | 1e-3 | (see §4) |
| `early_stop_patience` | **5** | stop before grinding into overfit |
| `batch_size` | 16 | studies-per-batch |

**Efficiency:** `grad_checkpoint=true`, `use_amp=true`, `length_bucketing=true`. K48 subsampling
~halved activation cost vs v1/v2 (~5–6 s/it, ~1h20m/epoch; GPU ~47 GB active).

---

## 6. Loss — `cost_sensitive` (unchanged from v2)

**Objective:** minimize expected *clinical* cost — under-calling pathology is priced highest.

```
L = E_j[ p_j · C[true, j] ]  +  λ_ce · CE(logits, target)
```

**Cost matrix C[true, pred]** (same eased matrix as v2):

| true ↓ / pred → | normal | near_normal | abnormal |
|---|--:|--:|--:|
| normal | 0 | 1 | 1 |
| near_normal | **2.0** | 0 | 1 |
| abnormal | **3.0** | 1 | 0 |

| Parameter | Value | Objective |
|---|--:|---|
| `cost_miss_abnormal` | 3.0 | penalize abnormal→normal (worst miss) |
| `cost_miss_near_normal` | 2.0 | penalize near_normal→normal |
| `cost_ce_lambda` | 0.4 | CE stabilizer weight |
| `label_smoothing` | 0.1 | inside both terms (raised from v2) |

*(Rule-out multi-task head is OFF — `multitask_ruleout=false`. That is a v5 feature.)*

---

## 7. Metrics & Selection

| Metric | Role | Objective |
|---|---|---|
| **`balanced_acc`** | **monitor / checkpoint select** | mean per-class recall — can't be gamed by over-calling |
| AUC (macro-OVR) | discrimination quality | class-threshold-independent separability |
| per-class recall | balance check | ensure no class is sacrificed |
| `not_normal_sensitivity` | screening safety | fraction of pathology flagged |
| `target_sensitivity` = 0.95 | operating-point floor | threshold on val → applied to test |

**Selection rule:** highest val `balanced_acc` → `best.pt` (= ep3). Later overfit epochs never reach the saved checkpoint.

---

## 8. Outcome

| metric | val (best, ep3) | held-out TEST |
|---|--:|--:|
| balanced_acc | 0.693 | **0.673** |
| AUC (macro-OVR) | 0.859 | **0.846** |
| accuracy | — | 0.675 |

- **Best of the family** (v1 0.665, v2 0.647); honest val→test drop (balAcc −0.02).
- Regularizers **raised the peak and delayed overfit to ~ep4**, but did not eliminate it.
- **AUC ceiling ~0.85 confirmed on held-out data** — reproduces across every hyperparameter regime
  → it's a label/boundary limit (noisy normal↔near_normal), not an optimization problem.

**Verdict:** deployment-*candidate* for triage/assist. Further LR/dropout/cost tuning will not move
AUC. Highest-leverage next moves → **2-class / rule-out reframe (v5)**, ensemble+TTA, near_normal label cleaning.

---

## 9. Difference vs v2

| Dimension | v2 | v3 | Why |
|---|---|---|---|
| **Init** | ImageNet | **warm-start ← v2 best ep4** | keep learned features, fresh schedule |
| **`drop_path`** | 0.0 | **0.15** | ⭐ the missing MaxViT backbone regularizer |
| **`train_slices_per_study`** | 96 | **48** | MIL slice-dropout + ~2× faster |
| **`weight_decay`** | 5e-4 | **1e-3** | stronger L2 to close the gap |
| **`label_smoothing`** | 0.05 | **0.1** | softer targets on the noisy boundary |
| **`lr` peak** | 2e-4 | **1.5e-4** | warm weights need a gentler peak |
| **`epochs` (T_max)** | 50 | **18** | cosine actually decays to ~0 |
| **`warmup_epochs`** | 3 | **1** | warm weights need no long ramp |
| **`early_stop_patience`** | 12 | **5** | don't grind into overfit |
| **Loss / cost matrix** | cost_sensitive 3.0/2.0 | *(unchanged)* | the part that worked |
| **Best balanced_acc (val)** | 0.647 | **0.693** | higher peak, later overfit onset |

**The through-line:** v2 fixed the *loss*; v3 fixes the *regularization + schedule*. By turning on
stochastic depth (the backbone regularizer v1/v2 lacked), adding slice-dropout, and shortening the
cosine horizon so the LR truly decays, v3 pushes the peak to 0.693 and holds generalization honest —
then confirms the ~0.85 AUC wall is a data/label limit, redirecting effort to the v5 rule-out reframe.

---
*Parameters sourced from `config.py`, `run_watchdog_v3.sh`, and `engine/losses.py`.
Results from the run's TensorBoard event files and `test()` output.*
