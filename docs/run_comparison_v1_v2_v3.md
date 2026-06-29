# ct_brain (MaxViT-MIL, 3-class) — Run Comparison: v1 vs v2 vs v3

> **Date:** 2026-06-28 (updated 2026-06-29 — v3 complete)
> **Task:** study-level 3-class CT-brain classification — normal / near_normal / abnormal
> **Backbone:** `maxvit_tiny_tf_384.in1k` + gated-attention MIL pooling (Ilse et al.)
> **Data (all runs identical):** patient-grouped split — train 13,114 / val 2,623 / test 2,623 series; image_size 384; 3 HU windows (brain/subdural/bone) → 3-channel.

---

## 1. Results summary

| run | dir | epochs run | **best balanced_acc** | AUC @ best | max AUC | status |
|---|---|---|--:|--:|--:|---|
| **v1** | `runs/maxvit384_3class_clinical` | 7–17* | **0.6645** @ep15 | 0.852 | 0.852 | stopped (plateau + overfit) |
| **v2** | `runs/maxvit384_3class_clinical_v2` | 0–6 | **0.6469** @ep4 | 0.837 | 0.837 | stopped (same ceiling, sharper overfit) |
| **v3** | `runs/maxvit384_3class_clinical_v3` | 0–8 | **0.6931** @ep3 | 0.859 | 0.859 | **complete** ✅ (early-stop ep8; test done) |

\* v1's TensorBoard event file retains epochs 7–17; chance = 0.333.

**Headline:** v3 is the **best of all three runs** — val balanced_acc **0.6931 @ep3** (vs v1 0.6645, v2 0.6469) and the top AUC (0.859) — and is the only run with **held-out TEST** numbers: **balanced_acc 0.673 / AUC 0.846 / acc 0.675**, per-class recall balanced (no class sacrificed), small val→test drop (balAcc −0.02, AUC −0.013) → it generalizes honestly. The **AUC ceiling (~0.85) is consistent across all three runs and now confirmed on truly held-out data** → it is a data/label limit (noisy normal↔near_normal boundary), not an optimization problem.

### Per-epoch trajectories

**v1** (cost 5/3, lr 3e-4) — high, oscillating, slowly overfitting:
| ep | balAcc | AUC | trL | vaL | rNorm | rNear | rAbn |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 8 | 0.649 | 0.846 | 0.80 | 0.82 | 0.51 | 0.74 | 0.70 |
| 10 | 0.661 | 0.847 | 0.75 | 0.88 | 0.66 | 0.56 | 0.77 |
| 15 | **0.665** | 0.852 | 0.61 | 0.84 | 0.57 | 0.76 | 0.66 |
| 17 | 0.659 | 0.841 | 0.56 | **0.94** | 0.65 | 0.59 | 0.74 |

**v2** (eased cost 3/2, lr 2e-4, more reg) — same ceiling 4× faster, overfit by ep4:
| ep | balAcc | AUC | trL | vaL | rNorm | rNear | rAbn |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 3 | 0.596 | 0.828 | 0.90 | **0.87** | 0.38 | 0.72 | 0.69 |
| 4 | **0.647** | 0.837 | 0.86 | 0.91 | 0.62 | 0.74 | 0.58 |
| 6 | 0.598 | 0.819 | 0.79 | **0.93** | 0.41 | 0.68 | 0.70 |

**v3** (stochastic depth + slice-subsampling + decaying LR, warm-started) — peaks ep3, overfits ep4+ (val_loss climbs to 0.99, train–val gap +0.135 by ep8), early-stopped ep8:
| ep | balAcc | AUC | trL | vaL | rNorm | rNear | rAbn |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 0 | 0.687 | 0.852 | 0.82 | 0.93 | 0.77 | 0.65 | 0.64 |
| 1 | 0.675 | 0.849 | 0.83 | 0.90 | 0.66 | 0.76 | 0.60 |
| 2 | 0.680 | 0.849 | 0.80 | 0.92 | 0.71 | 0.69 | 0.64 |
| **3** | **0.693** | **0.859** | 0.77 | 0.88 | 0.72 | 0.68 | 0.68 |
| 4 | 0.668 | 0.835 | 0.73 | 0.95 | 0.69 | 0.61 | 0.70 |
| 5 | 0.637 | 0.835 | 0.70 | 0.91 | 0.59 | 0.57 | 0.75 |
| 6 | 0.674 | 0.856 | 0.66 | 0.86 | 0.61 | 0.67 | 0.74 |
| 7 | 0.650 | 0.846 | 0.61 | 0.91 | 0.54 | 0.73 | 0.68 |
| 8 | 0.688 | 0.842 | 0.56 | **0.99** | 0.77 | 0.63 | 0.67 |

**v3 held-out TEST (best.pt = ep3):** balanced_acc 0.673 · AUC 0.846 · acc 0.675 · f1 0.669 · recall normal/near/abn 0.670/0.659/0.690. Screening op-point @95% sens (thr set on val): not-normal sens 0.941 / spec 0.452 — high false-alarm load; precision 0.87 is inflated by the enriched 79% not-normal test prevalence (≈0.30 at realistic ~20% prevalence).

---

## 2. Configuration per run

Bold = changed from the previous version. All else constant: `loss=cost_sensitive`, `monitor=balanced_acc`,
`target_sensitivity=0.95`, `batch_size=16`, `image_size=384`, `grad_checkpoint=true`, `use_amp=true`,
gated-attention MIL, 3 HU windows.

| hyperparameter | v1 | v2 | v3 |
|---|--:|--:|--:|
| init | from ImageNet | from ImageNet | **warm-start ← v2 best ep4 (weights-only)** |
| `cost_miss_abnormal` | 5.0 | **3.0** | 3.0 |
| `cost_miss_near_normal` | 3.0 | **2.0** | 2.0 |
| `cost_ce_lambda` | 0.3 | **0.4** | 0.4 |
| `lr` (peak) | 3e-4 | **2e-4** | **1.5e-4** |
| `epochs` (cosine T_max) | 50 | 50 | **18** |
| `warmup_epochs` | 2 | **3** | **1** |
| `weight_decay` | 1e-4 | **5e-4** | **1e-3** |
| `dropout` (head/drop_rate) | 0.04 | **0.1** | 0.1 |
| **`drop_path` (stochastic depth)** | 0.0 | 0.0 | **0.15** |
| `label_smoothing` | 0.05 | 0.05 | **0.1** |
| `window_jitter` | 0.05 | **0.1** | 0.1 |
| `train_slices_per_study` | 96 | 96 | **48** (random subset/epoch) |
| `max_slices_per_study` (eval) | 96 | 96 | 96 |
| `early_stop_patience` | 10 | **12** | **5** |

**Throughput:** v1/v2 ≈ 8–12 s/it (~2 h/epoch, K96 train); v3 ≈ 6 s/it (~1h20m/epoch) — the K48 slice subsampling roughly halves activation cost. GPU: v1/v2 peak ~64 GB; v3 ~47 GB active (≈75 GB reserved high-water-mark from the K96 val pass).

---

## 3. Reason for each switch

### v1 → v2 (2026-06-27)
v1 **plateaued** at val balanced_acc ~0.66 (best 0.6645 @ep15) with AUC capped ~0.85 (representation saturated). Two failure modes:
- **Violent per-class oscillation** — `recall_normal` swung 0.09→0.66 epoch-to-epoch. The heavy cost matrix (5/3) penalized calling pathology "normal" so hard that the **normal class was sacrificed**, capping balanced_acc and destabilizing recall.
- **Accelerating overfit** — train_loss fell to 0.56 while val_loss rose to 0.94 by ep17.

**v2 response:** ease the cost asymmetry (5/3→3/2), strengthen the CE blend (λ 0.3→0.4) to damp oscillation, lower peak LR (3e-4→2e-4) + longer warmup, and add mild regularization (wd, dropout, window_jitter).

### v2 → v3 (2026-06-28)
v2 **confirmed rather than broke the ceiling**: best balanced_acc 0.6469 @ep4 — *below* v1, reached 4× faster — then regressed. Diagnosis:
- **Overfit returned earlier and sharper** than v1 despite the reg bump: val_loss bottomed at **ep3** then rose every epoch; train–val balanced_acc gap blew out to +0.08 by ep6. The v2 regularizers (classifier dropout, wd) were the *wrong* ones for a MaxViT.
- **LR never decayed** — cosine `T_max=epochs=50`, so LR sat near peak (1.96e-4 @ep6), training at full LR straight through the overfit zone.
- **Oscillation eased but persisted.**

**v3 response:** attack overfitting with the levers that actually fit this architecture, warm-started from v2's best epoch so we keep the learned features and iterate cheaply:
- **`drop_path=0.15` (stochastic depth)** — the standard, most-effective regularizer for ViT/MaxViT; it was **OFF** in v1/v2 (only `drop_rate` was wired). *(Required a code change: new `drop_path` config field wired into `timm.create_model` in `models/build.py`; DropPath has no params so old checkpoints still load `strict=True`.)*
- **`train_slices_per_study` 96→48** — random slice subsampling per epoch = strong MIL bag-level regularizer **and** ~2× faster (eval still uses 96).
- **Real LR decay** — `epochs` 50→18 so the cosine actually anneals to ~0; lower peak (1.5e-4); shorter warmup (warm weights don't need it).
- **Stronger generic reg** — wd 5e-4→1e-3, label_smoothing 0.05→0.1.
- **`early_stop_patience` 12→5** — best arrives early; stop burning compute on overfit.

---

## 4. What changes for next (v4 candidates)

**The ceiling is the AUC ~0.85, and it is a label problem, not a tuning problem** — it reproduces across all three runs and all hyperparameter regimes. Ranked by leverage:

1. **2-class reframe (highest leverage).** Collapse to normal vs not-normal (or drop/merge `near_normal`). The AUC cap is dominated by the noisy normal↔near_normal boundary; removing that boundary is the only change expected to lift discrimination rather than re-balance it. Clinically aligned (screening = "is there pathology?"). 2-class data pipeline already staged.
2. **Model-weight EMA (decay ~0.999)** — deferred from v3. The single best remaining lever against the per-class oscillation, and a near-free regularizer (+0.5–1% typical). Needs a small trainer change (shadow weights, swap for eval/checkpoint). Add if v3's recalls still wobble.
3. **Ordinal head (CORN/CORAL).** The classes are ordered (normal < near_normal < abnormal); nominal cross-entropy throws that away and punishes adjacent confusions like far ones. An ordinal/cumulative-link head encodes the order and softens the fuzzy middle boundary.
4. **Label cleaning of `near_normal`.** Confident-learning (e.g. cleanlab) to surface mislabeled near_normal series, or consensus relabel. Most expensive, but directly addresses the root cause of the AUC cap.

**Recommendation:** v3 ran to its early stop and **confirmed the anti-overfit changes held** — peak pushed to 0.693 and overfit delayed ~1 epoch, though not eliminated (val_loss still climbs from ep4). Make **v4 = the 2-class reframe** rather than a 5th round of 3-class tuning. Keep v3's regularization recipe (stochastic depth + slice subsampling + decaying LR + wd 1e-3 / label_smoothing 0.1) as the new baseline config, and add EMA (item 2) since v3's per-class recalls still wobbled (rNorm 0.54–0.77 across epochs).

---
*Metrics read from each run's TensorBoard event files and v3's `test()` output. v3 complete — early-stopped at epoch 8, best checkpoint = epoch 3, held-out test evaluated. See [v3_detailed_report.md](v3_detailed_report.md) for the full per-epoch table and operating-point analysis.*
