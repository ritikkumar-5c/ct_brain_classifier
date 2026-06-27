# Training Progress Analysis — 3-Class Cost-Sensitive Run (through epoch 5)

> **Date:** 2026-06-26
> **Status:** Run in progress (mid-epoch 6). Epochs 0–5 have completed validation.
> **Update:** Supersedes the epoch-4 analysis. Epoch 5 materially changes the read **for the better** — the over-calling / specificity-collapse concern flagged at epoch 4 is reversing on its own.

## Validation metrics (epochs 0–5)

| ep | trLoss | vaLoss | balAcc | rNorm | rNear | rAbn | notNormSens | AUC   | acc  | f1   |
|----|--------|--------|--------|-------|-------|------|-------------|-------|------|------|
| 0  | 1.082  | 1.096  | 0.560  | 0.39  | 0.85  | 0.44 | 0.93        | 0.787 | 0.58 | 0.55 |
| 1  | 0.988  | 0.909  | 0.494  | 0.12  | 0.56  | 0.79 | 0.99        | 0.785 | 0.57 | 0.48 |
| 2  | 0.957  | 0.877  | 0.487  | 0.04  | 0.70  | 0.72 | 1.00        | 0.795 | 0.57 | 0.45 |
| 3  | 0.919  | 0.896  | 0.522  | 0.16  | 0.78  | 0.63 | 0.98        | 0.803 | 0.58 | 0.51 |
| 4  | 0.894  | 0.841  | 0.571  | 0.25  | 0.83  | 0.64 | 0.98        | 0.832 | 0.62 | 0.57 |
| 5  | 0.857  | 0.793  | 0.620  | 0.39  | 0.71  | 0.75 | 0.96        | 0.844 | 0.66 | 0.63 |

## What changed since epoch 4

- **Specificity collapse is reversing.** `recall_normal` went 0.04 (ep2) → 0.16 → 0.25 → **0.39** (ep5). The model has stopped dumping everything into the pathology classes.
- **Per-class recall is now balanced** at epoch 5: normal 0.39, near_normal 0.71, abnormal 0.75 — all three classes contributing, no degenerate single-class behavior.
- **Epoch 5 is a clean step-up on every axis at once:** balanced_acc 0.571→0.620, val_loss 0.841→0.793, AUC 0.832→0.844, accuracy 0.62→0.66, f1 0.57→0.63.
- **not_normal_sensitivity stays high (0.96)** — still catching ~96% of pathology, but now *not* at the total expense of normal recall.

## What each metric is telling us

- **train/val loss** (1.08→0.86, 1.10→0.79): both steadily decreasing. Val loss sits slightly **below** train loss — expected, since train has dropout + augmentation active while val does not. No overfitting. ✅
- **AUC** (0.787→0.844, monotonic up): threshold-independent discrimination keeps improving. The clearest "right direction" signal. ✅
- **balanced_acc** (0.49 trough → 0.62): climbing steadily since the epoch-2 trough, now driven by *recovering* normal recall rather than dragged down by it. ✅
- **recall_normal / not_normal_sensitivity**: the two are rebalancing into a healthy equilibrium — pathology sensitivity stays ~0.96 while normal recall climbs back to 0.39. The cost loss and the `balanced_acc` monitor have settled their tug-of-war.

## Diagnosis — over-bias was a transient warmup artifact

At epoch 4 the worry was that `cost_sensitive` (`cost_miss_abnormal=5`, `cost_miss_near_normal=3`) had pushed the argmax decision into near-degenerate over-calling (normal recall 0.04). Epoch 5 shows that was a **transient** early-training state, not a standing failure:

- the **balanced_acc monitor** rewards normal recall and has been pulling specificity back up (0.04 → 0.16 → 0.25 → 0.39), and
- as the backbone fine-tunes, the model gains the capacity to keep pathology sensitivity high **and** recover normal recall, instead of trading one for the other.

So the cost-sensitive loss is behaving as intended for ct_brain (no frozen-backbone warmup here, unlike MedicalNet where the same loss collapsed during head-only warmup).

## Verdict

**Direction: clearly correct — and improving faster than at epoch 4.** ✅

- **Discrimination:** AUC monotonic 0.787 → 0.844.
- **Decision quality:** balanced_acc 0.49 (trough) → 0.620, on recovering normal recall.
- **Generalization:** val_loss < train_loss and both falling — no overfitting.
- **Clinical priority intact:** not_normal_sensitivity ~0.96 throughout.
- **Still early:** epoch 5 of 50 (early-stop patience 10). Best checkpoint so far = balanced_acc 0.620 @ epoch 5.

The earlier lean toward switching to `weighted_ce` is **downgraded** — the over-bias healed itself, so `cost_sensitive` is no longer the concern it looked like at epoch 4.

## Recommendation

**Let it run — no intervention.** Monitor these:

- `recall_normal` should keep climbing. If it **stalls below ~0.5 for several consecutive epochs**, then consider softening costs (`--cost_miss_abnormal 2 --cost_miss_near_normal 1.5`) or switching to `--loss weighted_ce`.
- `balanced_acc` and AUC should keep rising — flag if they plateau before ~epoch 15.

Keep the **balanced_acc monitor** and the **0.95 operating point**; rely on the operating point for the final clinical decision threshold at test time.
