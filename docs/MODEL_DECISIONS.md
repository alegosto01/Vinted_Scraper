# Model Decisions

## 2026-07-04 — basic_5_giant_model pruned from 9 → 4 models

### Context

Live scorer (`apply_to_live_collector.py`) ran 9 models simultaneously since Jul 1.
Analysis on run `live_scoring_20260704_065826` (30,435 items scored, 19,600 evaluated@72h, base sell rate 21.7%).

### Models dropped

| model | reason |
|---|---|
| `logistic_v1_baseline` | r=1.000 with `logistic_snapshot_v2` — exact duplicate, 0 unique passes, prec@72h=0.447 |
| `logistic_snapshot_v2` | r=1.000 with `logistic_v1_baseline` — exact duplicate, 0 unique passes, prec@72h=0.447 |
| `sgd_text_numeric_v1` | r=0.96 with logistic pair, only 10 unique passes, prec@72h=0.442 |
| `linear_svm_calibrated_v1` | worst AUC of remaining models (0.551), prec@72h=0.537, 23 unique passes |
| `rules_price_v1` | pure rule/price filter (no ML), prec@72h=0.512 — lowest of non-logistic models; high volume (283 passed, 221 unique) but poor quality |

### Models kept

| model | passed | prec@72h | auc@72h | rationale |
|---|---:|---:|---:|---|
| `xgboost_basic_v1` | 117 | 0.720 | 0.671 | best precision |
| `random_forest_basic_v1` | 91 | 0.690 | 0.624 | good precision, distinct signal from xgb |
| `hist_gradient_basic_numeric_v1` | 60 | 0.625 | 0.675 | best AUC overall, complements xgb |
| `numeric_tree_v1` | 161 | 0.600 | 0.636 | highest volume at decent quality |

### Notes

- `xgboost_basic_v1` and `hist_gradient_basic_numeric_v1` are correlated (r=0.891) but kept because both have top-tier precision+AUC and the ensemble effect is worth it.
- `rules_price_v1` may be worth revisiting as a **pre-filter** (not a model vote) if coverage drops too much.
- Scorer needs restart to apply changes.
