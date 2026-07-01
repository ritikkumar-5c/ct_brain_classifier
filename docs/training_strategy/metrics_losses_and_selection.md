# Metrics, Losses & Selection — General Reference

> A **version-agnostic catalogue** of every metric, loss, and selection criterion used
> across ct_brain training runs (v1 · v2 · v3 · v5), each with its objective.
> This is a living document — append new methods as they are introduced; do **not** tie it
> to any single run's hyperparameters (those live in the per-version strategy reports).
>
> **Task:** study-level 3-class CT-brain — `0=normal`, `1=near_normal`, `2=abnormal`.
> **Design principle (AI-radiology):** *do not miss `near_normal` / `abnormal`.*
> **Code:** `engine/metrics.py`, `engine/losses.py`, `engine/trainer.py`, `config.py`.

---

## 0. The three separate levers

The pipeline deliberately keeps three concerns decoupled — each is tuned independently.

| Lever | Job | Where |
|---|---|---|
| **Loss** | shape the *error profile* during training | `engine/losses.py` + `--loss` |
| **Monitor** | select the best *model* (checkpoint + early stop) | `trainer.fit` + `--monitor` |
| **Operating point** | set the *decision threshold* to a required sensitivity | `metrics.py` (at eval/test) |

> **Golden rule:** never bake the sensitivity target into the monitor. Model quality
> (discrimination) and the deployed threshold are different things — pick the best model
> first, *then* choose the threshold on a held-out split.

---

## 1. Metrics (`engine/metrics.py`)

### 1a. Multi-class metrics — `compute_metrics` (argmax of softmax)

| Metric | Definition | Objective / when it matters | Seen in |
|---|---|---|---|
| `accuracy` | correct / total | overall correctness; imbalance-blind — can mislead | all |
| `precision` (macro) | mean per-class precision | over-calling penalty | all |
| `recall` (macro) | mean per-class recall | **identical to `balanced_acc`** | all |
| `f1` (macro) | mean per-class F1 | precision/recall balance | all |
| **`balanced_acc`** | mean(recall₀, recall₁, recall₂) | primary **monitor** for v1–v3; chance=0.333; can't be gamed by over-calling | v1·v2·v3 |
| `recall_{class}` | per-class recall = per-class **sensitivity** | exposes the noisy `near_normal` boundary & normal-class sacrifice | all |
| `not_normal_sensitivity` | P(pred≠normal \| true∈{near,abnormal}) | **caught-pathology rate** (argmax) — the clinical "don't miss" number | all |
| `normal_specificity` | P(pred=normal \| true=normal) | true-normal cleared (= `recall_normal` under argmax) | all |
| `auc` | macro one-vs-rest ROC-AUC | threshold-independent discrimination; `nan` if a split has one class | all |
| `loss` | mean epoch batch loss | convergence / overfit tracking | all |

**Gotchas:**
- `recall` (macro) and `balanced_acc` are the **same number**; `balanced_acc` is exposed under an explicit name for the monitor.
- `normal_specificity` and `recall_normal` are the **same number** under argmax (kept for the normal-vs-not-normal framing).
- `balanced_acc` averages over **all** `num_classes`; an absent class contributes recall 0. Fine on full splits; matters only on tiny/filtered ones.
- `not_normal_sensitivity` / `normal_specificity` here are the **argmax** view; the threshold-tuned versions come from the operating point (§3).

### 1b. Binary rule-out metrics — `ruleout_metrics` (score = P(not-normal))

Introduced with the v5 rule-out head. Computed on a 1-D score, not argmax.

| Metric | Definition | Objective | Seen in |
|---|---|---|---|
| `ruleout_auc` | ROC-AUC of the binary normal-vs-not-normal score | overall rule-out discrimination (clears the 3-class AUC wall) | v5 |
| `ruleout_spec_at_sens` | max specificity at sensitivity ≥ target | the single **auto-report** operating point | v5 |
| **`ruleout_pauc`** | mean specificity over a sensitivity grid `[target, ~1]` | smooth "how good is the whole high-sensitivity region" — v5 **monitor** | v5 |

**Objective of the tail focus:** auto-reporting normals safely depends on *specificity at very
high sensitivity*. `ruleout_pauc` is smoother than a single point → the preferred selection metric
for the tail.

---

## 2. Losses (`engine/losses.py`, selected with `--loss`)

| `--loss` | Formula | Objective / when to use | Seen in |
|---|---|---|---|
| `weighted_ce` *(config default)* | `CE(weight=class_weights, label_smoothing)` | frequency imbalance, symmetric costs | (baseline default) |
| `focal` | `(1 − p_t)^γ · CE` | hard-example focus (γ=`focal_gamma`); risky here — amplifies noisy `near_normal` | available, unused |
| **`cost_sensitive`** | `E_j[ p_j · C[true,j] ] + λ·CE` | asymmetric **clinical** cost — the 3-class workhorse | v1·v2·v3·v5 |
| **`RuleOutLoss`** (auxiliary) | `BCE + λ_pauc · pAUC-hinge` | grow/calibrate the auto-report tail (added to the 3-class loss) | v5 |

**Shared modifiers:**
- **Class weights** (`use_class_weights=True`): inverse class frequency, applied to the CE term.
- **`label_smoothing`**: softens targets on the noisy `normal/near_normal` boundary.

### 2a. Cost-sensitive loss (detail)

**Objective:** minimize **expected misclassification cost** under the predicted distribution, so
under-calling pathology as `normal` is explicitly expensive.

```
L = mean_i Σ_j p_ij · C[true_i, j]   +   cost_ce_lambda · CE
```

Cost matrix `C[true, pred]` (`build_cost_matrix`): 0 on diagonal, 1 for generic errors, higher for
under-calling pathology to `normal`:

```
              pred:  normal        near_normal   abnormal
true normal             0               1            1
true near_normal   cost_miss_near       0            1
true abnormal      cost_miss_abnormal   1            0
```

- **Knobs:** `cost_miss_abnormal`, `cost_miss_near_normal` (the two off-diagonal costs), `cost_ce_lambda` (CE blend for gradient stability / calibration).
- **Failure mode:** with too-heavy costs the objective can collapse to a *constant non-normal* prediction (dodging the costly "normal" column) — especially under a frozen backbone. ct_brain fine-tunes fully, so this is avoided; if over-calling persists, soften costs or raise `cost_ce_lambda`.

### 2b. Rule-out loss (detail)

**Objective:** calibrate the binary bag score **and** concentrate gradient on the deploy corner
(high sensitivity / low false-clear). Added to the 3-class loss weighted by `ruleout_weight`.

```
L_ruleout = BCE(logit, is_not_normal)  +  λ_pauc · Σ max(0, margin − (s_hard_pos − s_hard_neg))²
```

- **BCE term** — calibrates P(not-normal).
- **Partial-AUC term** — pushes the *hardest positives* (lowest-scoring pathology) above the *hardest negatives* (highest-scoring normals = would-be false clears) → raises specificity **exactly** at the target sensitivity.
- **Knobs:** `ruleout_weight`, `ruleout_topk` (top-k pooling), `ruleout_pauc_lambda`, `ruleout_pos_frac`/`neg_frac` (hard-subset selection), `ruleout_margin`, `ruleout_bce_pos_weight`.

---

## 3. Selection & decision (`engine/trainer.py`, `engine/metrics.py`)

### 3a. Checkpoint + early stop — `--monitor`

After each validation epoch:

```
score = val_metrics[monitor]              # higher = better
if score > best:  best = score; save best.pt; reset patience
else:             patience += 1; stop if patience >= early_stop_patience
save last.pt every epoch                  # full state → resume
```

| `--monitor` option | Selects for | Objective / notes | Seen in |
|---|---|---|---|
| **`balanced_acc`** | balanced 3-class discrimination | can't be gamed by over-calling; default for triage models | v1·v2·v3 |
| `f1` | precision/recall balance | alternative symmetric metric | available |
| `accuracy` | raw correctness | imbalance-blind — discouraged | available |
| `auc` | threshold-free discrimination | good when threshold is set separately | available |
| `not_normal_sensitivity` | caught-pathology rate | ⚠️ a threshold choice, not model quality — flag-everything maxes it; **avoid as monitor** | available |
| **`ruleout_pauc`** | auto-report tail (mean spec @ high sens) | v5 tail-growth selection | v5 |
| `ruleout_spec_at_sens` / `ruleout_auc` | tail point / binary AUC | rule-out alternatives | v5 |

- `best.pt` = best-monitor checkpoint; `last.pt` = most recent. Both store **full training state** (model + optimizer + scheduler + AMP scaler + epoch + best score + early-stop counter + RNG) → `--resume <ckpt>`.

### 3b. Clinical operating point (eval/test only)

Argmax uses a symmetric threshold; for "don't miss pathology" the threshold is tuned instead. Two
implementations, same principle (choose on val, apply to test → no leakage):

| Function | Score | Objective | Seen in |
|---|---|---|---|
| `pathology_operating_point` | `s = 1 − P(normal)` (from 3-class head) | max specificity s.t. `not_normal_sensitivity ≥ target_sensitivity` | v1·v2·v3 |
| `score_operating_point` / `apply_score_threshold` | any 1-D score (e.g. `p_ruleout`) | same, on the rule-out head's score | v5 |

**Procedure:** (1) compute the score; (2) on **val** pick the threshold maximizing specificity
subject to the sensitivity floor; (3) apply that fixed threshold to **test**. Yields a defensible
"at X% sensitivity, the false-alarm rate is Y%" statement.

**`target_sensitivity`** (typically 0.95) = the not-normal sensitivity floor. It belongs *here*, at
the operating point — not in the monitor or (ideally) the loss.

---

## 4. Config knobs (`config.py`) — the menu

Values below are **defaults / ranges**; the value in effect for any given run lives in that run's
strategy report and watchdog script.

| Field | Role | Objective |
|---|---|---|
| `num_classes` / `class_names` | task definition | 3-class; `normal_index=0` |
| `loss` | which loss | `weighted_ce` \| `focal` \| `cost_sensitive` |
| `monitor` | checkpoint / early-stop metric | see §3a |
| `target_sensitivity` | operating-point floor | not-normal sensitivity target at test |
| `cost_miss_abnormal` / `cost_miss_near_normal` | cost matrix off-diagonals | asymmetric clinical cost (cost_sensitive) |
| `cost_ce_lambda` | CE blend weight | stability / calibration of cost_sensitive |
| `use_class_weights` | inverse-freq CE weights | frequency-imbalance correction |
| `label_smoothing` | target softening | eases the noisy normal/near boundary |
| `focal_gamma` | focal focusing | only if `loss=focal` |
| `early_stop_patience` | early-stop window | epochs without monitor improvement |
| `multitask_ruleout` | enable rule-out head | `false` → v1–v4 behavior; `true` → v5 |
| `ruleout_*` | rule-out head/loss knobs | see §2b (topk, weight, pauc_lambda, pos/neg_frac, margin, bce_pos_weight) |
| `resume` | full-state resume path | "" = fresh run |

---

## 5. Quick map — what each run used

| | Loss | Monitor | Operating point | New method introduced |
|---|---|---|---|---|
| **v1** | cost_sensitive | balanced_acc | pathology_operating_point | baseline recipe |
| **v2** | cost_sensitive | balanced_acc | pathology_operating_point | (tuning only) |
| **v3** | cost_sensitive | balanced_acc | pathology_operating_point | (tuning only) |
| **v5** | cost_sensitive **+ RuleOutLoss** | **ruleout_pauc** | score_operating_point (`p_ruleout`) | rule-out head, pAUC loss, tail metrics |

---

## 6. Changelog (append new methods here)

- **v1** — established metrics/loss/selection baseline: `compute_metrics`, `cost_sensitive`, `balanced_acc` monitor, `pathology_operating_point`.
- **v5** — added binary rule-out track: `ruleout_metrics` (`ruleout_auc`, `ruleout_spec_at_sens`, `ruleout_pauc`), `RuleOutLoss` (BCE + partial-AUC), top-k pooling, and `score_operating_point` on `p_ruleout`.
- *(add future methods here — new losses, monitors, calibration/conformal steps, ensembling, etc.)*
