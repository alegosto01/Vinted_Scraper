# Benchmark Basic To Full

## Goal

This experiment tests a two-stage deal ranking cascade:

1. Run the existing basic model first.
2. Fully collect item/seller details only for items that pass the basic model.
3. Add photo-arbitrage visual features.
4. Run the full-scrape-plus-visual model as the final precision filter.

The experiment is isolated under:

```text
data/experiments/benchmark_basic_to_full/
```

It does not write to production `old_df.csv`, `sold_df.csv`, `unsold_df.csv`, or normal full-scrape files.

## Models

The default model source is:

```text
data/experiments/full_scrape_model/offline_runs/sold_status_feature_modalities_20260515_full_visual/
```

The cascade uses:

- Stage 1: best `basic_5` model per search.
- Stage 2: best `full_scrape_plus_visual` model per search.

Current default stage plan:

| Search | Stage 1 | Stage 1 Threshold | Stage 2 | Stage 2 Threshold |
| --- | --- | ---: | --- | ---: |
| `griffati_donna_all` | `linear_svm_calibrated_v1` | `0.6216` | `numeric_tree_v1` | `0.9728` |
| `griffati_uomo_all` | `logistic_v1_baseline` | `0.8866` | `numeric_tree_v1` | `0.9680` |
| `gucci` | `linear_svm_calibrated_v1` | `0.7984` | `logistic_v1_baseline` | `0.9643` |
| `nike` | `sgd_text_numeric_v1` | `0.9500` | `numeric_tree_v1` | `0.9917` |
| `prada` | `logistic_v1_baseline` | `0.9445` | `linear_svm_calibrated_v1` | `0.9955` |
| `ps4` | `sgd_text_numeric_v1` | `0.9815` | `linear_svm_calibrated_v1` | `0.9542` |

## Recheck Schedule

Tracked items are rechecked with this schedule:

- First 24 hours: every 1 hour.
- From 24 to 48 hours: every 3 hours.
- After 48 hours: every 12 hours until 7 days.

Precision is reported for checkpoint windows:

```text
1h, 2h, 3h, ..., 24h,
27h, 30h, 33h, ..., 48h,
60h, 72h, ..., 168h
```

## Metrics

The main metric is final precision among items that passed both stages.

The report also tracks:

- Stage-1 pass count.
- Stage-2 pass count.
- Full item collection success/failure count.
- Precision by checkpoint window.
- False positives: stage-2 pass items that did not sell by an evaluated window.
- False negatives: stage-1 pass items rejected by stage 2 that later sold.

## Commands

Dry-run the plan:

```bash
/home/ale/miniconda3/envs/vinted_scraper/bin/python scripts/experiments/benchmark_basic_to_full/cascade_runner.py collect-once --dry-run --out-dir data/experiments/benchmark_basic_to_full/live_runs/dry_run_plan
```

Run one collection pass:

```bash
/home/ale/miniconda3/envs/vinted_scraper/bin/python scripts/experiments/benchmark_basic_to_full/cascade_runner.py collect-once --out-dir data/experiments/benchmark_basic_to_full/live_runs/cascade_live
```

Recheck due items:

```bash
/home/ale/miniconda3/envs/vinted_scraper/bin/python scripts/experiments/benchmark_basic_to_full/cascade_runner.py recheck-due --out-dir data/experiments/benchmark_basic_to_full/live_runs/cascade_live
```

Generate report:

```bash
/home/ale/miniconda3/envs/vinted_scraper/bin/python scripts/experiments/benchmark_basic_to_full/cascade_runner.py report --out-dir data/experiments/benchmark_basic_to_full/live_runs/cascade_live
```

Run scheduled loop:

```bash
/home/ale/miniconda3/envs/vinted_scraper/bin/python scripts/experiments/benchmark_basic_to_full/cascade_runner.py run-loop --out-dir data/experiments/benchmark_basic_to_full/live_runs/cascade_live --collect-every-hours 1
```

## Notes

- All stage-1 pass items are tracked, not only final stage-2 pass items. This is needed for false-negative analysis.
- The full+visual stage can be slower because it collects item pages and runs photo-quality methods.
- DINO features are computed from the primary local image for the live item.
- The final proof should come from live checkpoint precision, not only offline sold-vs-unsold metrics.

