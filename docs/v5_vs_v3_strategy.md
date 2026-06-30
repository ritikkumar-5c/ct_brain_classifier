# v5 vs v3 — training-strategy diff

> **v5 goal:** grow the high-confidence-normal **tail** so normal studies can be
> **auto-reported** at a very low miss rate (≤2% false-clear among cleared). v3 was
> the best 3-class model but hit a ~0.85 AUC wall and its auto-clear tail was tiny
> and miscalibrated (val→test false-clear 0%→13.6%). v5 attacks the tail directly.
> **Same data as v3** (old splits, ~20% normal): `train_data/csvs/splits/{train,val,test}.csv`.

## 1. What's the same (kept from the v3 recipe that earned bal_acc 0.688)
3-class gated-attention MIL head, `loss=cost_sensitive` (miss costs 3.0/2.0, `cost_ce_lambda` 0.4),
`drop_path` 0.15, `dropout` 0.1, `label_smoothing` 0.1, `weight_decay` 1e-3, `window_jitter` 0.1,
`train_slices_per_study` 48 (eval 96), `batch_size` 16, `grad_checkpoint`, `use_amp`, length-bucketing,
backbone `maxvit_tiny_tf_384`, `target_sensitivity` 0.95, same patient-grouped old splits + held-out test.

## 2. Architecture / code changes (NEW capability)
| component | v3 | v5 |
|---|---|---|
| heads | single 3-class head | **two heads (multi-task):** 3-class triage head **+** binary normal-vs-not-normal **rule-out head** |
| rule-out pooling | — | **top-k pooling** (`TopKPool`): bag score = mean of the `k=8` most-pathological slice scores. Unlike the 3-class head's attention-**mean**, it does NOT dilute a 1–2 slice finding → subtle pathology can't masquerade as a confident normal |
| rule-out loss | — | **`RuleOutLoss` = BCE + partial-AUC pairwise surrogate**. The pAUC term pushes the hardest positives (low-scoring pathology) above the hardest normals (high-scoring would-be false-clears) → raises specificity exactly at the target sensitivity |
| backbone | shared by 1 head | shared by both heads — the pAUC gradient shapes features for the tail |
| test outputs | metrics only | also dumps `series_probs_{val,test}.csv` with a **`p_ruleout`** column + prints the rule-out operating point (threshold-on-val → applied-to-test) → feeds straight into split-conformal |

Files touched: `config.py` (8 new `ruleout_*` fields), `models/maxvit_mil.py` (`TopKPool` + rule-out head + `return_ruleout`), `engine/losses.py` (`RuleOutLoss`/`build_ruleout_loss`), `engine/metrics.py` (`ruleout_metrics` = AUC + spec@sens + pAUC; `score_operating_point`/`apply_score_threshold`), `engine/trainer.py` (composite loss, rule-out logging, dump). **`multitask_ruleout=False` → byte-for-byte v1–v4 behavior** (verified).

## 3. Hyperparameter / objective changes
| knob | v3 | v5 | why |
|---|---|---|---|
| `multitask_ruleout` | false | **true** | add the auto-report gate |
| `ruleout_topk` | — | **8** | top-k pooling for sparse findings |
| `ruleout_weight` | — | **0.5** | rule-out loss vs 3-class loss |
| `ruleout_pauc_lambda` / `pos_frac` | — | **1.0 / 0.5** | pAUC rank term, focused on the harder half of pathology |
| **`monitor`** | balanced_acc | **`ruleout_pauc`** | select the checkpoint that best grows the tail (mean specificity over the high-sensitivity region), not the symmetric balanced_acc |
| `init` | warm-start ← v2 | **fresh (ImageNet)** | new heads must co-adapt from start; avoids strict-load mismatch on v3 ckpts |
| `lr` | 1.5e-4 | **2.5e-4** | fresh fine-tune needs a higher peak |
| `warmup_epochs` | 1 | **2** | random heads + fresh optimizer need a ramp |
| `epochs` | 18 | **20** | fresh start on the small old data (~1.3 h/epoch at K48) |
| `slice_chunk` | 96 | **48** | cap GPU mem (~24 GB); capacity unaffected (pools over all slices) |

## 4. Why this should move the needle where v3 couldn't
- v3's auto-clear tail was capped by the noisy normal↔near_normal boundary **and** by an objective
  (balanced_acc / symmetric CE) that spent capacity on the middle of the ROC, not the corner we deploy in.
- v5's pAUC objective + top-k pooling concentrate learning on **specificity at high sensitivity** — the
  exact quantity that determines how many normals we can safely auto-clear.
- v5 keeps the 3-class head, so the **three-bucket triage** (auto-report normal / review near_normal /
  priority abnormal) is intact; only the auto-report gate is new.

## 5. After the run
`series_probs_{val,test}.csv` (with `p_ruleout`) → split-conformal at the ≤2% false-clear bar to read
off the **real** achievable auto-clear volume, and compare directly against the v3 baseline (oracle 0.4%
vol, naïve val→test 13.6% miss, uncertifiable).
