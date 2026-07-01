# v5 Training Strategy Report — MaxViT-MIL Rule-Out, 3-class + Auto-Report Gate

> **Run:** `runs/maxvit384_3class_clinical_v5`
> **Task:** study-level 3-class triage (`normal`/`near_normal`/`abnormal`) **+** binary rule-out gate (normal vs not-normal)
> **Launcher:** `run_watchdog_v5.sh` · **Config:** `config.py`
> **Init:** fresh (ImageNet pretrained — new heads co-adapt from start)
> **Data:** same old v3 splits (train 13,114 / val 2,623 / test 2,623 series; ~20% normal)
> **Status:** IN PROGRESS — through ep9, best (by `ruleout_pauc`) = **ep8**
> **Role in the lineage:** next iteration of **v3** — attacks the auto-report tail v3 couldn't grow

---

## 1. Strategic Objective

v3 was the best 3-class model but hit a **~0.85 AUC wall**, and its high-confidence-normal
**tail was tiny and miscalibrated** (val→test false-clear 0% → 13.6%). That tail is exactly what a
screening deployment needs: the region where a normal study can be **auto-reported** at a very low
miss rate.

**v5's single goal:** grow and calibrate that tail — **maximize specificity at very high
sensitivity** — by adding a second head purpose-built for it, while keeping the proven v3 3-class
head for triage.

| v3 limitation | v5 response |
|---|---|
| Auto-clear tail tiny + miscalibrated | Dedicated binary **rule-out head** |
| Attention-mean dilutes 1–2 slice findings | **Top-k pooling** (k=8) on the rule-out head |
| Objective (balanced_acc / CE) spends capacity on ROC middle | **Partial-AUC** objective focuses on the deploy corner |
| Checkpoint selected on symmetric balanced_acc | Select on **`ruleout_pauc`** (tail metric) |

---

## 2. Architecture — the new capability

| Component | v3 | v5 |
|---|---|---|
| heads | single 3-class head | **two heads:** 3-class triage + binary rule-out |
| rule-out pooling | — | **`TopKPool`** — bag score = mean of the `k=8` most-pathological slice scores |
| backbone | shared by 1 head | shared by both — pAUC gradient shapes features for the tail |
| test outputs | metrics only | dumps `series_probs_{val,test}.csv` with a **`p_ruleout`** column + rule-out operating point |

**Why top-k, not attention-mean:** the 3-class head averages attention weights across the bag,
diluting a finding on only 1–2 slices. Top-k pooling keeps the most pathological slices → **subtle
pathology can't masquerade as a confident normal**.

Files touched: `config.py` (8 `ruleout_*` fields), `models/maxvit_mil.py` (`TopKPool` + rule-out head),
`engine/losses.py` (`RuleOutLoss`), `engine/metrics.py` (`ruleout_metrics`), `engine/trainer.py`
(composite loss + dump). **`multitask_ruleout=false` → byte-for-byte v1–v4 behavior.**

---

## 3. Kept from the v3 recipe (the part that earned bal_acc 0.688)

3-class gated-attention MIL head · `loss=cost_sensitive` (miss costs 3.0/2.0, `cost_ce_lambda` 0.4)
· `drop_path` 0.15 · `dropout` 0.1 · `label_smoothing` 0.1 · `weight_decay` 1e-3 · `window_jitter` 0.1
· `train_slices_per_study` 48 (eval 96) · `batch_size` 16 · `grad_checkpoint` · `use_amp` ·
length-bucketing · backbone `maxvit_tiny_tf_384` · `target_sensitivity` 0.95 · same patient-grouped splits.

---

## 4. Rule-Out Loss — `RuleOutLoss`

**Objective:** calibrate the bag not-normal score **and** grow the tail at the deploy corner.
Added to the 3-class loss with weight `ruleout_weight`.

```
L_ruleout = BCE(logit, is_not_normal)  +  λ_pauc · Σ max(0, margin − (s_hard_pos − s_hard_neg))²
```

- **BCE term** — calibrates the bag not-normal probability.
- **Partial-AUC term** — within each batch, take the **hardest positives** (lowest-scoring pathology
  — where a high-sensitivity threshold must sit) and the **hardest negatives** (highest-scoring
  normals — would-be false clears), and push them apart with a squared-hinge rank penalty. This raises
  specificity **exactly** at the target sensitivity, where a plain BCE/CE spreads effort across the whole range.

| Parameter | Value | Objective |
|---|--:|---|
| `ruleout_weight` | 0.5 | rule-out loss vs. 3-class loss |
| `ruleout_topk` | 8 | top-k slices pooled for the bag score |
| `ruleout_pauc_lambda` | 1.0 | weight of the pAUC rank term |
| `ruleout_pos_frac` | 0.5 | focus on the harder half of pathology |
| `ruleout_neg_frac` | 1.0 | use all normals (few per batch) |
| `ruleout_margin` | 1.0 | squared-hinge margin |
| `ruleout_bce_pos_weight` | 1.0 | BCE class balance (>1 = penalize false-clears more) |

*(3-class loss unchanged: `cost_sensitive`, costs 3.0/2.0, `cost_ce_lambda` 0.4 — see v3 report §6.)*

---

## 5. Optimization Schedule

| Parameter | Value | Objective |
|---|--:|---|
| optimizer | Adam | — |
| `lr` (peak) | **2.5e-4** | fresh fine-tune needs a higher peak (vs v3's 1.5e-4) |
| `epochs` (cosine `T_max`) | **20** | fresh start on the small old data (~1.3 h/epoch at K48) |
| `warmup_epochs` | **2** | random heads + fresh optimizer need a ramp |
| `weight_decay` | 1e-3 | (kept from v3) |
| `early_stop_patience` | 5 | stop once the tail metric plateaus |
| `slice_chunk` | **48** | cap GPU mem (~24 GB); capacity unaffected — pooling spans all slices |
| `batch_size` | 16 | studies-per-batch |

Efficiency: `grad_checkpoint=true`, `use_amp=true`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
(reduce allocator fragmentation → lower reserved GPU memory).

---

## 6. Metrics & Selection

| Metric | Role | Objective |
|---|---|---|
| **`ruleout_pauc`** | **monitor / checkpoint select** | mean specificity over the high-sensitivity region — the tail-growth metric |
| `ruleout_spec@sens` | tail readout | specificity at `target_sensitivity`=0.95 |
| `ruleout_auc` | rule-out separability | binary normal-vs-not-normal AUC |
| `balanced_acc` | triage health | mean per-class recall (3-class head) |
| AUC (macro-OVR) | triage discrimination | 3-class separability |
| `not_normal_sensitivity` | clinical priority | fraction of pathology caught |

**Selection rule:** highest val `ruleout_pauc` → `best.pt`. This deliberately optimizes the
auto-clear corner, not the symmetric balanced_acc v3 used.

---

## 7. Progress So Far (through ep9, best = ep8)

| metric | ep8 (best) |
|---|--:|
| `ruleout_pauc` (selection) | **0.278** |
| `ruleout_spec@sens` | 0.480 |
| `ruleout_auc` | 0.901 |
| balanced_acc | 0.674 |
| AUC (3-class) | 0.844 |
| not_normal_sensitivity | 0.91 |

- **Rule-out AUC 0.90 >> 3-class AUC 0.85** — the binary framing clears the 3-class wall, as intended.
- Tail metric (`ruleout_pauc`) trending up through ep8; no overfitting signal (train/val loss both falling).
- Per-epoch `recall_normal` still swings (0.14–0.80) — the 3-class head remains sensitive to the noisy boundary.

**After the run:** feed `series_probs_{val,test}.csv` (`p_ruleout`) into split-conformal at the
**≤2% false-clear** bar to read off the *real* achievable auto-clear volume, and compare directly
against the v3 baseline (oracle 0.4% vol, naïve val→test 13.6% miss — uncertifiable).

---

## 8. Difference vs v3

| Dimension | v3 | v5 | Why |
|---|---|---|---|
| **Purpose** | best 3-class triage model | grow the auto-report **normal tail** | v3 confirmed the 3-class AUC wall |
| **Heads** | 1 (3-class) | **2** (3-class + rule-out) | dedicated tail head |
| **Rule-out pooling** | — | **top-k (k=8)** | don't dilute sparse findings |
| **Extra loss** | — | **`RuleOutLoss`** (BCE + pAUC) | optimize the deploy corner |
| **Monitor** | `balanced_acc` | **`ruleout_pauc`** | select on the tail metric |
| **Init** | warm-start ← v2 | **fresh (ImageNet)** | new heads must co-adapt from start |
| **`lr` peak** | 1.5e-4 | **2.5e-4** | fresh fine-tune needs a higher peak |
| **`warmup_epochs`** | 1 | **2** | random heads + fresh optimizer need a ramp |
| **`epochs`** | 18 | **20** | fresh start on small data |
| **`slice_chunk`** | 96 | **48** | cap GPU mem (~24 GB) |
| **Test outputs** | metrics | **metrics + `p_ruleout` CSVs** → split-conformal | certify auto-clear volume |
| **Unchanged** | — | cost loss 3.0/2.0, `cost_ce_lambda` 0.4, drop_path 0.15, dropout 0.1, label_smoothing 0.1, WD 1e-3, window_jitter 0.1, K48/eval96, batch 16, data | the proven v3 recipe |

**The through-line:** v3 maxed out the *3-class* framing and proved the ~0.85 AUC ceiling is a
label/boundary limit. v5 stops fighting that wall and instead adds a **binary rule-out head** trained
on a **partial-AUC objective** with **top-k pooling** — concentrating learning on specificity at high
sensitivity, the exact quantity that sets how many normals can be safely auto-cleared. The 3-class head
is retained, so triage (auto-report normal / review near_normal / priority abnormal) stays intact.

---
*Parameters sourced from `config.py`, `run_watchdog_v5.sh`, `engine/losses.py`, and `v5_vs_v3_strategy.md`.
Progress metrics from the report_daemon TensorBoard export (through ep9).*
