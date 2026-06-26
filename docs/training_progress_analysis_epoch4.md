# Training Progress Analysis — 3-Class Cost-Sensitive Run (through epoch 4)

> **Date:** 2026-06-26
> **Status:** Run in progress (mid-epoch 5). Epochs 0–4 have completed validation.
> **Note on logs:** The `nan` rows in the metrics log are spurious — they come from the per-step LR scalar creating extra entries, not from failed validation. The real per-epoch data is below.

## Validation metrics (epochs 0–4)

| ep | trLoss | vaLoss | balAcc | rNorm | rNear | rAbn | normSpec | notNormSens | AUC |
|----|--------|--------|--------|-------|-------|------|----------|-------------|-------|
| 0  | 1.082  | 1.096  | 0.560  | 0.39  | 0.85  | 0.44 | 0.39     | 0.93        | 0.787 |
| 1  | 0.988  | 0.909  | 0.494  | 0.12  | 0.56  | 0.79 | 0.12     | 0.99        | 0.785 |
| 2  | 0.957  | 0.877  | 0.487  | 0.04  | 0.70  | 0.72 | 0.04     | 1.00        | 0.795 |
| 3  | 0.919  | 0.896  | 0.522  | 0.16  | 0.78  | 0.63 | 0.16     | 0.98        | 0.803 |
| 4  | 0.894  | 0.841  | 0.571  | 0.25  | 0.83  | 0.64 | 0.25     | 0.98        | 0.832 |

## What each metric is telling us

- **train/val loss** (1.08→0.89, 1.10→0.84): both steadily decreasing → the model is fitting **and** generalizing. Val tracks train, so no overfitting yet. ✅
- **AUC** (0.787→0.832, monotonic up): the threshold-independent discrimination metric. The model is genuinely getting better at ranking the three classes. This is the clearest "right direction" signal. ✅
- **not_normal_sensitivity** (~0.98–1.0): catches essentially all pathology. ✅ (the cost loss's intended effect)
- **normal_specificity = recall_normal** (0.39→0.12→0.04→0.16→0.25): ⚠️ **this is the problem.** The model is barely ever predicting "normal" — it's flagging 75–96% of truly-normal scans as pathology. Massive over-calling.
- **balanced_acc** (0.49–0.57): dragged down almost entirely by that collapsed normal recall (near/abnormal recalls are healthy at 0.6–0.85).

## Diagnosis — learning correctly, but the loss is over-tuned

Two things are simultaneously true:

1. **The representation is improving** — AUC up, losses down, near/abnormal recall healthy. At the level of "can the model tell these apart," it's going the right way.
2. **The decision behavior is near-degenerate** — `cost_miss_abnormal=5` / `cost_miss_near_normal=3` are so high that the argmax decision almost never picks "normal." That's why specificity sits at 0.04–0.39. A model that calls 90% of normals abnormal isn't clinically usable as-is.

This is the **over-calling failure mode** flagged earlier: the cost-sensitive loss is double-applying the sensitivity priority that the operating-point threshold already handles. There's a tug-of-war:

- the **cost loss** pushes away from "normal," while
- the **balanced_acc monitor** rewards normal recall —

and you can see the monitor slowly winning (normal recall recovering 0.04→0.25 over epochs 2→4).

## Verdict

**Direction: half right.**

- Discrimination (AUC) is improving correctly, so the run isn't wasted — and because AUC is decent (0.83), the test operating point would still recover a usable sens/spec tradeoff even from this model.
- But the cost loss is **mis-tuned**: it's sacrificing nearly all specificity, which drags balanced_acc down and makes the raw argmax predictions impractical.

## Recommendation

Two clean paths:

### Option 1 — Let it run and watch
Let it run to ~epoch 10–12 and watch `normal_specificity`. It's recovering (0.04→0.25) as the backbone fine-tunes and the monitor pulls it up. If it climbs past ~0.5, the run is fine and you rely on the operating point for the final threshold.

### Option 2 — Restart with a less aggressive setup *(preferred)*
Either soften costs:
```
--cost_miss_abnormal 2 --cost_miss_near_normal 1.5   # and/or --cost_ce_lambda 0.5
```
or, cleaner, switch to:
```
--loss weighted_ce
```
and let the operating point set sensitivity. That's the principled design — **train a balanced, well-calibrated model, then choose the threshold** — rather than baking the bias into both the loss and the threshold.

### Lean: Option 2 with `weighted_ce`

This run is demonstrating that `cost_sensitive` + operating-point is **redundant** and **over-biases**. A balanced model + 0.95 operating point gives the same "don't miss pathology" guarantee without crushing specificity. We've only spent ~5 epochs, and full-state checkpoints mean nothing is lost.

Keep the **balanced_acc monitor** and the **0.95 operating point** when restarting.
