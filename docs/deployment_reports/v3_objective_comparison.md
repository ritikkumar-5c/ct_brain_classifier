# v3 Deployment — Objective Comparison: pos(abnormal) vs pos(near_normal + abnormal)

Which rule-out objective to deploy on v3. Both are computed identically — same model (`runs/maxvit384_3class_clinical_v3`), patient-mean aggregation, threshold set on val → measured on the held-out test set. Full per-objective detail in each report:

- **pos(near_normal + abnormal)** — auto-clear = "truly **normal**" → `v3_production_deployment_pos(near-normal_abnormal).md`
- **pos(abnormal)** — auto-clear = "no **abnormal** pathology (minor findings OK)" → `v3_production_deployment_pos(abnormal).md`

---

## Side-by-side

| @95% target | NOT-NORMAL objective | ABNORMAL objective |
|---|--:|--:|
| dangerous-miss class | near_normal **+** abnormal | abnormal **only** |
| miss rate | 5.5% (69/1,247) | 5.1% (35/689) |
| Negative Precision (NPV) | 70.4% | **91.7%** |
| % of negatives auto-cleared | 48.8% (164/336 normal) | 43.1% (385/894 non-abn) |
| workload saved (of 1,583) | 14.7% (233) | **26.5% (420)** |
| what's auto-reported | normal only | normal **+ near_normal** |

| @99.5% target | NOT-NORMAL objective | ABNORMAL objective |
|---|--:|--:|
| miss rate | 0.48% (6/1,247) | 0.58% (4/689) |
| Negative Precision (NPV) | 85.0% | **96.0%** |
| workload saved (of 1,583) | 2.5% (40) | **6.4% (101)** |
| min achievable miss | 0.24% (3 cases) | 0.58% floor (4 cases) |

**Trade-off in one line:** the abnormal objective **clears ~1.8× the volume at much higher NPV** (91.7% vs 70.4% at 95%) — *but only by auto-reporting near_normal studies and no longer counting near_normal as a miss.*

---

## Which to deploy

| If the requirement is… | Use | near_normal handling |
|---|---|---|
| "auto-reported studies must be **normal**" | **pos(near_normal + abnormal)** | → doctor |
| "auto-reported studies must have **no abnormal pathology** (minor findings OK)" | **pos(abnormal)** | auto-reported |

Neither is yet safe-and-useful on v3 (abnormal floors at 0.58% miss; not-normal collapses to ~10% volume at the ≤0.5% bar). The model improvements that lift both are the same — see each report's Caveats section.

---
*All numbers measured on the held-out test set (patient-mean aggregation, threshold-on-val → test), from `eval_autorule.py` / `eval_autorule_abnormal.py` over `series_probs_{val,test}.csv`.*
