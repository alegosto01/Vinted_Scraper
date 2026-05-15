# Full Scrape Model

## Purpose

This experiment family trains the same offline model approaches used by the existing deal-finder sweep, but on the newer full-scrape-derived datasets.

The main goal is to evaluate sold vs not-sold prediction quality on full-scraped items, using per-search training and the same model families already used in deal-finder benchmarking.

At this point, the only task that matters is `sold_status`.

The alternative `extreme_score` task was wired up initially for `score_low` vs `score_high`, but the user explicitly said they do not care about that path now, and its run outputs were deleted.

## Current Decision State

- Keep and use only the `sold_status` task.
- Ignore `extreme_score` for now.
- Exclude `Borse_Griffate` and `Scarpe_Griffate` from user-facing result summaries.
- Current comparison baseline for the old approach is the deal-finder offline sweep in [data/experiments/deal_finder/offline_runs/sweep_20260510_222252](/home/ale/Desktop/Vinted_New_Version/data/experiments/deal_finder/offline_runs/sweep_20260510_222252).

## Code Layout

The new experiment package lives in [scripts/experiments/full_scrape_model](/home/ale/Desktop/Vinted_New_Version/scripts/experiments/full_scrape_model).

- [scripts/experiments/full_scrape_model/dataset.py](/home/ale/Desktop/Vinted_New_Version/scripts/experiments/full_scrape_model/dataset.py)
  Builds per-search datasets from merged full-scrape CSV exports.
- [scripts/experiments/full_scrape_model/model_sweep.py](/home/ale/Desktop/Vinted_New_Version/scripts/experiments/full_scrape_model/model_sweep.py)
  Reuses the deal-finder sweep machinery and model definitions, but writes outputs into the full-scrape-model experiment root.
- [scripts/experiments/full_scrape_model/paths.py](/home/ale/Desktop/Vinted_New_Version/scripts/experiments/full_scrape_model/paths.py)
  Defines the experiment root, model output root, and helper manifest/json writers.
- [scripts/experiments/full_scrape_model/build_dataset.py](/home/ale/Desktop/Vinted_New_Version/scripts/experiments/full_scrape_model/build_dataset.py)
  Thin CLI wrapper around `dataset.py`.
- [scripts/experiments/full_scrape_model/train_offline.py](/home/ale/Desktop/Vinted_New_Version/scripts/experiments/full_scrape_model/train_offline.py)
  Thin CLI wrapper around `model_sweep.py`.

## Reused Logic

The sweep intentionally reuses the existing deal-finder approaches from [scripts/experiments/deal_finder/model_sweep.py](/home/ale/Desktop/Vinted_New_Version/scripts/experiments/deal_finder/model_sweep.py).

Current approach set:

- `logistic_v1_baseline`
- `logistic_snapshot_v2`
- `sgd_text_numeric_v1`
- `linear_svm_calibrated_v1`
- `numeric_tree_v1`
- `rules_price_v1`
- `visual_basic_v1`

The full-scrape sweep patches the reused deal-finder sweep so that:

- artifacts are written under the full-scrape-model experiment root instead of the deal-finder root
- metadata is rewritten with `experiment_family = full_scrape_model`
- task-specific label names are recorded in model metadata

## Data Sources

### Primary Sold/Not-Sold Training Source

The active source is:

- [data/simple_scrape/full_scrape_merged_stage_resume_20260513/backfill_sold_unsold_all_searches.csv](/home/ale/Desktop/Vinted_New_Version/data/simple_scrape/full_scrape_merged_stage_resume_20260513/backfill_sold_unsold_all_searches.csv)

Important columns used by the dataset builder:

- `SourceSearch`
- `MergedStatusBinary`
- `Dataid` and fallback `Link` for identity

For `sold_status`, the dataset builder maps:

- `MergedStatusBinary == sold` to `offline_sold_label = 1`
- all rows to `offline_label_eligible = True`
- `label_quality = exact`
- `label_source = full_scrape_stage_resume_20260513`

### Deprecated Low/High Score Source

This source exists in code, but is not currently relevant to the user:

- [data/simple_scrape/full_scrape_merged_stage_resume_20260513/main_full_scrape_verylow_veryhigh_all_searches.csv](/home/ale/Desktop/Vinted_New_Version/data/simple_scrape/full_scrape_merged_stage_resume_20260513/main_full_scrape_verylow_veryhigh_all_searches.csv)

The related outputs were deleted from the full-scrape-model run folder.

## Search Coverage

The sold-status merged full-scrape export currently contains these searches:

- `Borse_Griffate`
- `griffati_donna_all`
- `griffati_uomo_all`
- `gucci`
- `nike`
- `prada`
- `ps4`

There is no `Scarpe_Griffate` entry in the merged full-scrape sold-status export.

For reporting, the user asked to exclude:

- `Borse_Griffate`
- `Scarpe_Griffate`

## Current Run State

The active full-scrape-model sold-status run is:

- [data/experiments/full_scrape_model/offline_runs/sold_status_sweep_20260515_101018](/home/ale/Desktop/Vinted_New_Version/data/experiments/full_scrape_model/offline_runs/sold_status_sweep_20260515_101018)

Key files from that run:

- [manifest.json](/home/ale/Desktop/Vinted_New_Version/data/experiments/full_scrape_model/offline_runs/sold_status_sweep_20260515_101018/manifest.json)
- [dataset_summary.json](/home/ale/Desktop/Vinted_New_Version/data/experiments/full_scrape_model/offline_runs/sold_status_sweep_20260515_101018/dataset_summary.json)
- [metrics_long.csv](/home/ale/Desktop/Vinted_New_Version/data/experiments/full_scrape_model/offline_runs/sold_status_sweep_20260515_101018/metrics_long.csv)
- [best_by_search.csv](/home/ale/Desktop/Vinted_New_Version/data/experiments/full_scrape_model/offline_runs/sold_status_sweep_20260515_101018/best_by_search.csv)
- [best_by_approach.csv](/home/ale/Desktop/Vinted_New_Version/data/experiments/full_scrape_model/offline_runs/sold_status_sweep_20260515_101018/best_by_approach.csv)
- [promotion_candidates.json](/home/ale/Desktop/Vinted_New_Version/data/experiments/full_scrape_model/offline_runs/sold_status_sweep_20260515_101018/promotion_candidates.json)
- [sweep_report.md](/home/ale/Desktop/Vinted_New_Version/data/experiments/full_scrape_model/offline_runs/sold_status_sweep_20260515_101018/sweep_report.md)

Run summary:

- task: `sold_status`
- dataset count: `7`
- trained rows: `42`
- promotion candidates: `25`
- seed set: `42`

The earlier `extreme_score` runs were removed, so only sold-status runs should remain under [data/experiments/full_scrape_model/offline_runs](/home/ale/Desktop/Vinted_New_Version/data/experiments/full_scrape_model/offline_runs).

## Feature Modality Comparison

The feature-modality comparison run is:

- [data/experiments/full_scrape_model/offline_runs/sold_status_feature_modalities_20260515_full_visual](/home/ale/Desktop/Vinted_New_Version/data/experiments/full_scrape_model/offline_runs/sold_status_feature_modalities_20260515_full_visual)

This run trains the same sold-status model families across three modes:

- `basic_5`: only `Title`, `Brand`, `Size`, `Price`, and `Likes` as source inputs.
- `full_scrape`: basic fields plus full item/seller metadata, without photo-arbitrage visual features.
- `full_scrape_plus_visual`: full-scrape fields plus photo-arbitrage visual features from `sold_unsold_visuals_20260514_full`, including DINO embedding dimensions where available.

Main files:

- [feature_modality_report.md](/home/ale/Desktop/Vinted_New_Version/data/experiments/full_scrape_model/offline_runs/sold_status_feature_modalities_20260515_full_visual/feature_modality_report.md)
- [metrics_long.csv](/home/ale/Desktop/Vinted_New_Version/data/experiments/full_scrape_model/offline_runs/sold_status_feature_modalities_20260515_full_visual/metrics_long.csv)
- [modality_comparison_by_search.csv](/home/ale/Desktop/Vinted_New_Version/data/experiments/full_scrape_model/offline_runs/sold_status_feature_modalities_20260515_full_visual/modality_comparison_by_search.csv)
- [approach_mode_lift.csv](/home/ale/Desktop/Vinted_New_Version/data/experiments/full_scrape_model/offline_runs/sold_status_feature_modalities_20260515_full_visual/approach_mode_lift.csv)

Headline result:

- The large improvement comes from full item/seller fields, not from visual features.
- Visual features add small extra value for some model/search pairs, especially `nike`, `prada`, and a few `ps4` rows by AUC.
- Visual features did not improve the best full-scrape row for every search; in several cases they reduced the selected threshold count while keeping precision high.
- This is still an offline sold-vs-unsold test, not a final live fast-sale result.

## Current Best Models Per Search

From [best_by_search.csv](/home/ale/Desktop/Vinted_New_Version/data/experiments/full_scrape_model/offline_runs/sold_status_sweep_20260515_101018/best_by_search.csv), excluding `Borse_Griffate`:

- `griffati_donna_all`: `numeric_tree_v1`
- `griffati_uomo_all`: `linear_svm_calibrated_v1`
- `gucci`: `linear_svm_calibrated_v1`
- `nike`: `visual_basic_v1`
- `prada`: `numeric_tree_v1`
- `ps4`: `numeric_tree_v1`

Headline metrics for those winners:

- `griffati_donna_all`: test precision `0.9355`, test p@10 `1.0`, test count `31`
- `griffati_uomo_all`: test precision `0.8333`, test p@10 `0.8`, test count `24`
- `gucci`: test precision `0.9524`, test p@10 `1.0`, test count `21`
- `nike`: test precision `0.8333`, test p@10 `0.9`, test count `30`
- `prada`: test precision `1.0`, test p@10 `1.0`, test count `18`
- `ps4`: test precision `1.0`, test p@10 `1.0`, test count `25`

## Comparison Against Older Deal-Finder Models

Comparison baseline:

- [data/experiments/deal_finder/offline_runs/sweep_20260510_222252](/home/ale/Desktop/Vinted_New_Version/data/experiments/deal_finder/offline_runs/sweep_20260510_222252)

Important result summary:

- Full-scrape won clearly on `griffati_donna_all`, `gucci`, `prada`, and `ps4`.
- `griffati_uomo_all` looks less extreme on raw precision than the old deal-finder best row, but the new full-scrape winner is better supported and promotion-qualified, while the old row was not.
- `nike` is the one search where the old deal-finder best model still looks stronger.

More detailed interpretation:

- `griffati_donna_all`: full-scrape `numeric_tree_v1` beat the old `numeric_tree_v1` by a large margin.
- `griffati_uomo_all`: full-scrape `linear_svm_calibrated_v1` is more conservative but more credible than the old `logistic_snapshot_v2` result.
- `gucci`: full-scrape switched the winner from `visual_basic_v1` to `linear_svm_calibrated_v1` and improved precision.
- `nike`: old `linear_svm_calibrated_v1` beat full-scrape `visual_basic_v1`.
- `prada`: full-scrape switched the winner from `visual_basic_v1` to `numeric_tree_v1` while keeping precision at `1.0` and improving PR AUC.
- `ps4`: full-scrape kept `numeric_tree_v1` and improved both threshold precision and p@10.

Important correction to earlier intuition:

The older deal-finder sweep did not use less labeled data. In the compared searches, it actually had more eligible labeled rows than the full-scrape sold-status sweep. The likely explanation for many of the full-scrape improvements is cleaner labels and better class balance, not larger sample size.

## Dataset Sizes Used In The Full-Scrape Sold-Status Run

From [dataset_summary.json](/home/ale/Desktop/Vinted_New_Version/data/experiments/full_scrape_model/offline_runs/sold_status_sweep_20260515_101018/dataset_summary.json):

- `Borse_Griffate`: eligible `17`, positives `17`
- `griffati_donna_all`: eligible `4810`, positives `1804`
- `griffati_uomo_all`: eligible `7894`, positives `4081`
- `gucci`: eligible `8635`, positives `4707`
- `nike`: eligible `4711`, positives `2437`
- `prada`: eligible `3650`, positives `1993`
- `ps4`: eligible `4647`, positives `2471`

`Borse_Griffate` is too small and degenerate for meaningful model selection here.

## How To Rerun

### Build Datasets Only

```bash
/home/ale/miniconda3/envs/vinted_scraper/bin/python scripts/experiments/full_scrape_model/build_dataset.py --task sold_status --all-searches
```

### Run The Sold-Status Sweep For All Searches

```bash
/home/ale/miniconda3/envs/vinted_scraper/bin/python scripts/experiments/full_scrape_model/model_sweep.py --task sold_status --all-searches
```

### Run A Single Search

```bash
/home/ale/miniconda3/envs/vinted_scraper/bin/python scripts/experiments/full_scrape_model/model_sweep.py --task sold_status --search gucci
```

### Restrict To Specific Approaches

```bash
/home/ale/miniconda3/envs/vinted_scraper/bin/python scripts/experiments/full_scrape_model/model_sweep.py --task sold_status --search gucci --approach numeric_tree_v1 --approach linear_svm_calibrated_v1
```

### Robustness Mode

```bash
/home/ale/miniconda3/envs/vinted_scraper/bin/python scripts/experiments/full_scrape_model/model_sweep.py --task sold_status --all-searches --robustness
```

## Known Constraints

- The sweep currently requires at least `50` eligible rows and both label classes to train for a search.
- `visual_basic_v1` can be slower because it derives image features.
- The full-scrape sweep is offline only. It does not trigger paper trading, timers, or live scraping.
- The current output root is fully separate from the deal-finder output root.
- The current user intent is evaluation and comparison, not deployment.

## Recommended Next Steps For Another Agent

If continuing this work, the most useful next steps are:

1. Decide whether to adopt the full-scrape winners as the new default per-search offline candidates.
2. Investigate why `nike` regressed relative to the old deal-finder sweep.
3. If needed, produce a single side-by-side CSV comparing old vs full-scrape best rows for the kept searches.
4. If the user later changes direction, the code can support `extreme_score`, but that path should stay dormant unless they explicitly ask for it again.
