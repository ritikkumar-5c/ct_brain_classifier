# Metrics, Losses & Decision/Selection — Reference

> 3-class CT-brain classifier (`0=normal, 1=near_normal, 2=abnormal`).
> Code: `engine/metrics.py`, `engine/losses.py`, `engine/trainer.py`, `config.py`.
> Design goal (AI-radiology): **do not miss `near_normal` / `abnormal`.**

Division of labour (the three levers are deliberately separate):

| Lever | Job | Where |
|---|---|---|
| **Monitor** | pick the best-discriminating *model* | `trainer.fit` + `--monitor` |
| **Loss** | shape the *error profile* during training | `engine/losses.py` + `--loss` |
| **Operating point** | set the *decision threshold* to the required sensitivity | `metrics.pathology_operating_point` (at test) |

> ### ✅ Currently active in the live run (`runs/maxvit384_3class_clinical`)
> | Setting | Active value |
> |---|---|
> | **Loss** | **`cost_sensitive`** — expected-cost + 0.3·CE; costs `abnormal→normal=5`, `near_normal→normal=3`; CE term class-weighted (`use_class_weights=True`) + `label_smoothing=0.05` |
> | **Monitor** (best/early-stop) | **`balanced_acc`** (early-stop patience 10) |
> | **Operating point** | **`target_sensitivity=0.95`** (val-chosen threshold, applied at test) |
> | Task | 3-class, `normal_index=0` |
>
> Launched via: `--loss cost_sensitive --monitor balanced_acc --target_sensitivity 0.95`.
> Everything else below documents the full menu of options; the rows tagged **[ACTIVE]** are what this run uses.

---

## 1. Metrics (`engine/metrics.py::compute_metrics`)

All training/val metrics use **argmax** of the 3 softmax probabilities. Verified against hand-computed values.

| Metric | Definition | Notes |
|---|---|---|
| `accuracy` | correct / total | imbalance-blind; can mislead |
| `precision` | macro mean of per-class precision | unweighted over the 3 classes |
| `recall` | macro mean of per-class recall | **identical to `balanced_acc`** |
| `f1` | macro mean of per-class F1 | |
| `balanced_acc` | mean(recall₀, recall₁, recall₂) | **default monitor — [ACTIVE]**; chance = 0.333; can't be gamed by over-calling |
| `recall_normal` / `recall_near_normal` / `recall_abnormal` | per-class recall = per-class **sensitivity** | where the `near_normal` boundary shows up |
| `not_normal_sensitivity` | P(pred ≠ normal \| true ∈ {near_normal, abnormal}) | **caught-pathology rate**, argmax. The clinical "don't miss" number |
| `normal_specificity` | P(pred = normal \| true = normal) | = `recall_normal` (one-vs-rest specificity of normal). True-normal cleared |
| `auc` | macro one-vs-rest ROC-AUC | threshold-independent; `nan` if only one class present in the split |
| `loss` | mean batch loss for the epoch | |

Notes / gotchas:
- `recall` (macro) and `balanced_acc` are the **same number** — `balanced_acc` is exposed under an explicit name for the monitor.
- `normal_specificity` and `recall_normal` are the **same number** under argmax (kept for the normal-vs-not-normal screening framing).
- `balanced_acc` averages over **all** `num_classes`; an absent class contributes recall 0 (deflating it). Fine on full splits (all 3 classes present); matters only on tiny/filtered splits.
- `not_normal_sensitivity` / `normal_specificity` are the **argmax** screening view. The **operating-point** values (§3) are the threshold-tuned versions, reported only at test.

---

## 2. Losses (`engine/losses.py`, selected with `--loss`)

| `--loss` | Formula | Use when |
|---|---|---|
| `weighted_ce` *(config default)* | `CrossEntropy(weight=class_weights, label_smoothing)` | frequency imbalance, symmetric costs |
| `focal` | `(1 − p_t)^γ · CE` | hard-example focus (γ = `focal_gamma`); risky here — amplifies noisy `near_normal` |
| **`cost_sensitive`** **[ACTIVE]** | `E_j[ p_j · C[true, j] ] + λ · CE` | asymmetric clinical cost — **used in the current run** |

**Class weights** (`use_class_weights=True`): inverse class frequency, applied to the CE term.

**`label_smoothing`** (default 0.05): softens targets; helps the noisy `normal/near_normal` boundary.

### Cost-sensitive loss in detail

- Minimizes **expected misclassification cost** under the predicted distribution: `loss = mean_i Σ_j p_ij · C[true_i, j]`, plus a `cost_ce_lambda · CE` term for gradient stability / calibration.
- Cost matrix `C[true, pred]` (`build_cost_matrix`): 0 on the diagonal, 1 for generic errors, higher for **under-calling pathology to `normal`**:

```
              pred:  normal   near_normal   abnormal
true normal             0          1            1
true near_normal        3 ←        0            1        # cost_miss_near_normal
true abnormal           5 ←        1            0        # cost_miss_abnormal
```

- Knobs: `--cost_miss_abnormal` (default 5), `--cost_miss_near_normal` (default 3), `--cost_ce_lambda` (default 0.3).
- **Known failure mode:** the expected-cost objective can be minimized by collapsing to a *constant non-normal* prediction (dodging the costly "normal" column) when the model lacks capacity — e.g. during a **frozen-backbone warmup** (this is what made MedicalNet collapse early). ct_brain does not freeze, so it avoids this; observed behaviour is high pathology sensitivity with specificity recovering as the backbone fine-tunes. Soften the costs or raise `cost_ce_lambda` if over-calling persists.

---

## 3. Decision / best-checkpoint selection (`engine/trainer.py`)

### Checkpoint + early stop — `--monitor`

After each epoch's validation:

```python
score = val_metrics[cfg.monitor]          # higher = better
if score > best:  best = score; save best.pt;  reset patience
else:             patience += 1; stop if patience >= early_stop_patience (10)
save last.pt every epoch                  # full state (resume)
```

- **`--monitor` (default `balanced_acc`).** Options: `balanced_acc | f1 | not_normal_sensitivity | accuracy | auc`.
- **Why `balanced_acc`:** rewards recall on all three classes and **cannot be gamed by over-calling** (over-calling tanks `normal` recall). Recommended.
- **Why not `not_normal_sensitivity` as the monitor:** sensitivity is a *threshold choice*, not model quality — a flag-everything model maxes it. Select the best-discriminating model, then set the threshold via the operating point.
- `best.pt` = best-monitor checkpoint; `last.pt` = most recent. Both store **full training state** (model + optimizer + scheduler + AMP scaler + epoch + best score + early-stop counter + RNG) → resume with `--resume <ckpt>`.

### Clinical operating point (test only — `pathology_operating_point`)

Argmax uses a symmetric threshold. For "don't miss pathology", the decision threshold is tuned instead:

1. Pathology score `s = 1 − P(normal)`.
2. On **val**, choose the threshold that **maximizes specificity subject to `not_normal_sensitivity ≥ --target_sensitivity`** (default 0.95).
3. Apply that fixed threshold to **test** → report `op_test_sensitivity`, `op_test_specificity` (no leakage; val-chosen, test-applied).

This yields a defensible "at 95% pathology sensitivity, here is the false-alarm rate" statement, and is the proper home for the sensitivity priority (rather than baking it into the loss).

---

## 4. Config knobs (`config.py`)

"Active" = value in effect for the current `runs/maxvit384_3class_clinical` run (CLI override or config default).

| Field | Active value | Meaning |
|---|---|---|
| `num_classes` / `class_names` | 3 / (normal, near_normal, abnormal) | task definition; `normal_index = 0` |
| `loss` | **`cost_sensitive`** (CLI) | overrides config default `weighted_ce` |
| `monitor` | **`balanced_acc`** (CLI = default) | checkpoint / early-stop metric |
| `target_sensitivity` | **0.95** (CLI = default) | not-normal sensitivity floor for the test operating point |
| `cost_miss_abnormal` | **5.0** | C[abnormal, normal] (active — `cost_sensitive`) |
| `cost_miss_near_normal` | **3.0** | C[near_normal, normal] (active — `cost_sensitive`) |
| `cost_ce_lambda` | **0.3** | CE blend weight in cost-sensitive loss (active) |
| `use_class_weights` | **True** | inverse-frequency weights on the CE term |
| `label_smoothing` | **0.05** | target smoothing |
| `early_stop_patience` | **10** | epochs without monitor improvement before stopping |
| `focal_gamma` | 2.0 | focal focusing — **inactive** (only if `loss=focal`) |
| `resume` | "" | checkpoint path for full-state resume — **inactive** (fresh run) |

### Current clinical run
```
--loss cost_sensitive --monitor balanced_acc --target_sensitivity 0.95
```
Judge it by **`not_normal_sensitivity` + operating-point specificity**, not `balanced_acc` alone (the cost loss deliberately trades some specificity for sensitivity).
