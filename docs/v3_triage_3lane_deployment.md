# v3 Production Deployment — 3-Lane Triage (normal / near_normal / abnormal)

> **Model:** ct_brain v3 (`runs/maxvit384_3class_clinical_v3`, best.pt = ep3), MaxViT-MIL, 3-class.
> **Date:** 2026-06-29
> **Scoring unit:** per-patient, **mean** aggregation over the patient's series.
> **Evaluated on:** held-out **test set** (1,583 patients, 21% normal). ⚠️ enriched prevalence — see §7.

## 1. The proposed workflow

Route each study by the model's **predicted class**:

| predicted | → destination | human effort |
|---|---|---|
| **normal** | AI auto rule-out + auto-generated report | **none (fully autonomous)** |
| **near_normal** | internal pre-read team | junior/cheaper human |
| **abnormal** | radiologist (direct) | full radiologist read |

The single safety-critical point is the **normal lane** — it's the only place a human never looks. near_normal and abnormal both keep a human in the loop, so the pre-read team is a **safety net** for the model's uncertainty.

---

## 2. Confusion matrix (held-out test, patient-level mean, argmax)

| | pred normal | pred near_normal | pred abnormal | total (true) |
|---|--:|--:|--:|--:|
| **true normal** | 235 | 94 | 7 | 336 |
| **true near_normal** | 99 | 367 | 92 | 558 |
| **true abnormal** | 31 | 173 | 485 | 689 |
| **total (pred)** | 365 | 634 | 584 | 1,583 |

Per-class recall: normal 70%, near_normal 66%, abnormal 70%. AUC 0.85 (the model's ceiling).

---

## 3. Routing volumes & lane composition (argmax)

| lane (predicted) | destination | volume | % of all | actually contains |
|---|---|--:|--:|---|
| normal | AI auto + report | 365 | 23.1% | **235 normal, 99 near_normal, 31 abnormal** |
| near_normal | pre-read team | 634 | 40.1% | 94 normal, 367 near_normal, 173 abnormal |
| abnormal | radiologist | 584 | 36.9% | 7 normal, 92 near_normal, 485 abnormal |

---

## 4. Workload & savings (argmax routing)

Radiologist reads **only the abnormal lane**; the pre-read team absorbs the near lane; AI handles the normal lane.

| resource | studies | % of all |
|---|--:|--:|
| AI auto (no human) | 365 | 23.1% |
| pre-read team | 634 | 40.1% |
| radiologist | 584 | **36.9%** |

- **Radiologist workload reduction: 63.1%** (reads 584 instead of 1,583) — the headline prize.
- **Fully automated (no human at all): 23.1%.**

**Cost saving depends on pre-read cost** (`f` = pre-read cost ÷ radiologist cost). Total reading cost vs all-to-radiologist = `(584 + 634·f) / 1583`:

| pre-read cost f | total reading cost | **cost saving** |
|---|--:|--:|
| 0.3 | 48.9% | **51.1%** |
| 0.5 | 56.9% | **43.1%** |

---

## 5. Safety — the auto-normal lane is the problem (argmax)

The auto-normal lane is the only autonomous decision, and at argmax it is **not safe**:

- **Lane purity (NPV) = 235/365 = 64.4%** → **35.6% of "auto-normal" reports are actually pathology.**
- **31 truly-abnormal patients (4.5% of all abnormals) would be auto-reported as normal** — missed diagnoses with no human review. **Unacceptable.**
- The 99 near_normal auto-cleared are less critical but still suboptimal.

The near/abnormal lanes are recoverable (a human sees them): the 173 abnormals routed to pre-read (25% of abnormals) are *under-triaged*, not missed — **provided the pre-read team escalates**. So the deployment risk concentrates entirely in the auto-normal lane.

---

## 6. The fix — confidence-gate the auto-normal lane

Only auto-report when `P(normal)` is high; send low-confidence "normal" predictions to the pre-read team instead. **Radiologist load is unchanged (584, 63.1% reduction) — gating only moves cases between the AUTO and PRE-READ lanes:**

| auto-lane rule | AUTO vol (%) | auto-lane NPV | abnormal mis-auto-cleared | pre-read vol | radiologist (fixed) |
|---|--:|--:|--:|--:|--:|
| argmax (P(normal) is max) | 365 (23.1%) | 64.4% | **31** | 634 | 584 (63.1% ↓) |
| P(normal) ≥ 0.90 | 156 (9.9%) | 75.0% | 12 | 843 | 584 (63.1% ↓) |
| P(normal) ≥ 0.95 | 102 (6.4%) | 79.4% | 8 | 897 | 584 (63.1% ↓) |
| **P(normal) ≥ 0.977** (0.995-sens) | **40 (2.5%)** | **85.0%** | **2** | 959 | 584 (63.1% ↓) |

**Trade-off:** tightening the auto lane shrinks full automation (23% → 2.5%) but cuts abnormal misses (31 → 2) and lifts NPV (64% → 85%). The radiologist saving is unaffected — it comes from the abnormal lane, not the auto lane.

---

## 7. Feasibility verdict

**The 3-lane design is materially more deployable than pure binary auto-rule-out**, for one structural reason: the **pre-read team absorbs the model's uncertainty**, so the AI only acts autonomously on confidently-normal studies. Concretely:

- **Radiologist workload reduction ≈ 63%** on this test set — large, and it's really a *shift* of ~40% of volume onto a cheaper pre-read team (net cost saving ~43–51% depending on pre-read cost).
- **Safe full automation is small but real:** gate the auto lane at `P(normal) ≥ 0.977` → 2.5% fully automated at 85% lane-NPV with only **2/689 abnormals mis-cleared (0.3%)**. Argmax routing (23% automated) is **not** safe (31 abnormals mis-cleared).

**Required conditions for go-live:**
1. **Gate the auto lane** to a clinically-agreed `P(normal)` threshold (don't use raw argmax) + **calibrate** the probability first (v3 is overconfident).
2. **Pre-read escalation protocol** — the lane's value depends on the pre-read team catching the 25% of abnormals routed to it; define escalation + audit.
3. **Random audit** of the auto lane; **monitor** prevalence/drift with auto-revert.

⚠️ **Prevalence caveat (critical).** These numbers are on the **enriched 21%-normal test set**. In a pathology-heavy production population (e.g. a recent 1-week sample was ~6% normal / ~94% not-normal):
- the **abnormal lane grows** → radiologist reduction shrinks well below 63%;
- the **auto-normal lane NPV falls** (fewer true normals to find) → even gated automation may drop below the safety bar.
- **Re-measure the full table on the real production distribution before committing.**

**Bottom line:** feasible as a **radiologist-offloading triage** (≈60% fewer radiologist reads by shifting near-normals to pre-read), with a **small, confidence-gated, audited** fully-autonomous normal lane — *not* as broad autonomous reporting. The biggest lever to widen safe automation remains a higher-AUC model (2-class reframe / label cleaning).

---
*Confusion matrix, routing and gating from the dumped `series_probs_test.csv` (patient-level mean). Radiologist reduction = 1 − pred_abnormal/total. Cost saving model assumes radiologist reads only the abnormal lane.*
