> ⚠️ **Caveman-compressed** — terse/fragment style to save tokens. Technical substance, code, commands, URLs kept verbatim. Original backed up under `~/.local/share/caveman-compress/backups/`.

# Full Scrape Model

## Purpose

Experiment family train same offline model approaches as deal-finder sweep, but on newer full-scrape datasets.

Main goal: evaluate sold vs not-sold prediction quality on full-scraped items, per-search training, same model families as deal-finder benchmark.

Only task that matter now: `sold_status`.

Alternative `extreme_score` task wired up first for `score_low` vs `score_high`, but user said no care, outputs deleted.

## Current Decision State

- Keep only `sold_status` task.
- Ignore `extreme_score`.
- Exclude `Borse_Griffate` and `Scarpe_Griffate` from user-facing summaries.
- Baseline for old approach: deal-finder offline sweep at [data/experiments/deal_finder/offline_runs/sweep_20260510_222252](/home/ale/Desktop/Vinted_New_Version/data/experiments/deal_finder/offline_runs/sweep_20260510_222252).

## Code Layout

New experiment package at [scripts/experiments/current/full_scrape_model](/home/ale/Desktop/vinted/Vinted_New_Version/scripts/experiments/current/full_scrape_model).

- [scripts/experiments/current/full_scrape_model/dataset.py](/home/ale/Desktop/Vinted_New_Version/scripts/experiments/current/full_scrape_model/dataset.py)
  Build per-search datasets from merged full-scrape CSV exports.
- [scripts/experiments/current/full_scrape_model/model_sweep.py](/home/ale/Desktop/Vinted_New_Version/scripts/experiments/current/full_scrape_model/model_sweep.py)
  Reuse deal-finder sweep machinery and model defs, write outputs to full-scrape-model root.
- [scripts/experiments/current/full_scrape_model/paths.py](/home/ale/Desktop/Vinted_New_Version/scripts/experiments/current/full_scrape_model/paths.py)
  Define experiment root, model output root, helper manifest/json writers.
- [scripts/experiments/current/full_scrape_model/build_dataset.py](/home/ale/Desktop/Vinted_New_Version/scripts/experiments/current/full_scrape_model/build_dataset.py)
  Thin CLI wrap of `dataset.py`.
- [scripts/experiments/current/full_scrape_model/train_offline.py](/home/ale/Desktop/Vinted_New_Version/scripts/experiments/current/full_scrape_model/train_offline.py)
  Thin CLI wrap of `model_sweep.py`.

## Reused Logic

Sweep reuse deal-finder approaches from [scripts/experiments/current/deal_finder/model_sweep.py](/home/ale/Desktop/Vinted_New_Version/scripts/experiments/current/deal_finder/model_sweep.py).

Current approach set:

- `logistic_v1_baseline`
- `logistic_snapshot_v2`
- `sgd_text_numeric_v1`
- `linear_svm_calibrated_v1`
- `numeric_tree_v1`
- `rules_price_v1`
- `visual_basic_v1`

Full-scrape sweep patch reused deal-finder sweep so:

- artifacts go under full-scrape-model root not deal-finder root
- metadata rewritten with `experiment_family = full_scrape_model`
- task-specific label names recorded in model metadata

## Data Sources

### Primary Sold/Not-Sold Training Source

Active source:

- [data/simple_scrape/full_scrape_merged_stage_resume_20260513/backfill_sold_unsold_all_searches.csv](/home/ale/Desktop/Vinted_New_Version/data/simple_scrape/full_scrape_merged_stage_resume_20260513/backfill_sold_unsold_all_searches.csv)

Columns used by dataset builder:

- `SourceSearch`
- `MergedStatusBinary`
- `Dataid` and fallback `Link` for identity

For `sold_status`, dataset builder map:

- `MergedStatusBinary == sold` to `offline_sold_label = 1`
- all rows to `offline_label_eligible = True`
- `label_quality = exact`
- `label_source = full_scrape_stage_resume_20260513`

### Deprecated Low/High Score Source

Source in code but not relevant now:

- [data/simple_scrape/full_scrape_merged_stage_resume_20260513/main_full_scrape_verylow_veryhigh_all_searches.csv](/home/ale/Desktop/Vinted_New_Version/data/simple_scrape/full_scrape_merged_stage_resume_20260513/main_full_scrape_verylow_veryhigh_all_searches.csv)

Related outputs deleted from full-scrape-model run folder.

## Search Coverage

Sold-status merged full-scrape export contain these searches:

- `Borse_Griffate`
- `griffati_donna_all`
- `griffati_uomo_all`
- `gucci`
- `nike`
- `prada`
- `ps4`

No `Scarpe_Griffate` entry in merged sold-status export.

User say exclude from reports:

- `Borse_Griffate`
- `Scarpe_Griffate`

## Current Run State

Active full-scrape-model sold-status run:

- [data/experiments/full_scrape_model/offline_runs/sold_status_sweep_20260515_101018](/home/ale/Desktop/Vinted_New_Version/data/experiments/full_scrape_model/offline_runs/sold_status_sweep_20260515_101018)

Key files:

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

Earlier `extreme_score` runs removed, so only sold-status runs remain under [data/experiments/full_scrape_model/offline_runs](/home/ale/Desktop/Vinted_New_Version/data/experiments/full_scrape_model/offline_runs).

## Feature Modality Comparison

Feature-modality run:

- [data/experiments/full_scrape_model/offline_runs/sold_status_feature_modalities_20260515_full_visual](/home/ale/Desktop/Vinted_New_Version/data/experiments/full_scrape_model/offline_runs/sold_status_feature_modalities_20260515_full_visual)

Train same sold-status families in three modes:

- `basic_5`: only `Title`, `Brand`, `Size`, `Price`, `Likes` as inputs.
- `full_scrape`: basic fields plus full item/seller metadata, no photo-arbitrage visual features.
- `full_scrape_plus_visual`: full-scrape fields plus photo-arbitrage visual features from `sold_unsold_visuals_20260514_full`, includes DINO embedding dims where available.

Main files:

- [feature_modality_report.md](/home/ale/Desktop/Vinted_New_Version/data/experiments/full_scrape_model/offline_runs/sold_status_feature_modalities_20260515_full_visual/feature_modality_report.md)
- [metrics_long.csv](/home/ale/Desktop/Vinted_New_Version/data/experiments/full_scrape_model/offline_runs/sold_status_feature_modalities_20260515_full_visual/metrics_long.csv)
- [modality_comparison_by_search.csv](/home/ale/Desktop/Vinted_New_Version/data/experiments/full_scrape_model/offline_runs/sold_status_feature_modalities_20260515_full_visual/modality_comparison_by_search.csv)
- [approach_mode_lift.csv](/home/ale/Desktop/Vinted_New_Version/data/experiments/full_scrape_model/offline_runs/sold_status_feature_modalities_20260515_full_visual/approach_mode_lift.csv)

Headline result:

- Big lift from full item/seller fields, not from visual features.
- Visual features add small extra value for some model/search pairs, especially `nike`, `prada`, few `ps4` rows by AUC.
- Visual features no improve best full-scrape row for every search; sometimes reduce threshold count while keep precision high.
- Still offline sold-vs-unsold test, not final live fast-sale result.

## Current Best Models Per Search

From [best_by_search.csv](/home/ale/Desktop/Vinted_New_Version/data/experiments/full_scrape_model/offline_runs/sold_status_sweep_20260515_101018/best_by_search.csv), exclude `Borse_Griffate`:

- `griffati_donna_all`: `numeric_tree_v1`
- `griffati_uomo_all`: `linear_svm_calibrated_v1`
- `gucci`: `linear_svm_calibrated_v1`
- `nike`: `visual_basic_v1`
- `prada`: `numeric_tree_v1`
- `ps4`: `numeric_tree_v1`

Headline metrics for winners:

- `griffati_donna_all`: test precision `0.9355`, test p@10 `1.0`, test count `31`
- `griffati_uomo_all`: test precision `0.8333`, test p@10 `0.8`, test count `24`
- `gucci`: test precision `0.9524`, test p@10 `1.0`, test count `21`
- `nike`: test precision `0.8333`, test p@10 `0.9`, test count `30`
- `prada`: test precision `1.0`, test p@10 `1.0`, test count `18`
- `ps4`: test precision `1.0`, test p@10 `1.0`, test count `25`

## Comparison Against Older Deal-Finder Models

Baseline:

- [data/experiments/deal_finder/offline_runs/sweep_20260510_222252](/home/ale/Desktop/Vinted_New_Version/data/experiments/deal_finder/offline_runs/sweep_20260510_222252)

Result summary:

- Full-scrape win clear on `griffati_donna_all`, `gucci`, `prada`, `ps4`.
- `griffati_uomo_all` look less extreme on raw precision than old deal-finder best, but new full-scrape winner better supported and promotion-qualified, old row was not.
- `nike` only search where old deal-finder best still stronger.

Detail interpretation:

- `griffati_donna_all`: full-scrape `numeric_tree_v1` beat old `numeric_tree_v1` by big margin.
- `griffati_uomo_all`: full-scrape `linear_svm_calibrated_v1` more conservative but more credible than old `logistic_snapshot_v2`.
- `gucci`: full-scrape switch winner from `visual_basic_v1` to `linear_svm_calibrated_v1`, precision up.
- `nike`: old `linear_svm_calibrated_v1` beat full-scrape `visual_basic_v1`.
- `prada`: full-scrape switch winner from `visual_basic_v1` to `numeric_tree_v1`, precision still `1.0`, PR AUC up.
- `ps4`: full-scrape keep `numeric_tree_v1`, both threshold precision and p@10 up.

Important correction to earlier intuition:

Old deal-finder sweep did not use less labeled data. In compared searches it had MORE eligible labeled rows than full-scrape sold-status sweep. Likely cause of full-scrape gains: cleaner labels and better class balance, not bigger sample.

## Dataset Sizes Used In The Full-Scrape Sold-Status Run

From [dataset_summary.json](/home/ale/Desktop/Vinted_New_Version/data/experiments/full_scrape_model/offline_runs/sold_status_sweep_20260515_101018/dataset_summary.json):

- `Borse_Griffate`: eligible `17`, positives `17`
- `griffati_donna_all`: eligible `4810`, positives `1804`
- `griffati_uomo_all`: eligible `7894`, positives `4081`
- `gucci`: eligible `8635`, positives `4707`
- `nike`: eligible `4711`, positives `2437`
- `prada`: eligible `3650`, positives `1993`
- `ps4`: eligible `4647`, positives `2471`

`Borse_Griffate` too small and degenerate for model selection.

## SHAP / Feature Contribution Analysis

Readable SHAP-style analysis added for best `basic_5` and `full_scrape_plus_visual` models from full visual run.

Latest run:

```text
data/experiments/full_scrape_model/offline_runs/sold_status_feature_modalities_20260515_full_visual/shap_analysis/no_dino_20260515_232153/
```

Main files:

- `shap_analysis_report.md`: compact human-readable summary.
- `shap_model_summary.csv`: one row per explained search/mode/model.
- `shap_feature_importance_long.csv`: feature-level contribution table.
- `shap_group_importance.csv`: grouped importance by text, basic numeric, full-scrape, readable visual features.
- `shap_item_explanations.csv`: per-item top positive and negative drivers.

Raw DINO embedding dims like `DinoEmbedding_0000` excluded from reported SHAP tables. Models still explained as trained, but report hide raw dims for readability. DINO summary features like `DinoEmbeddingNorm` and `DinoOutlierScore` stay visible.

Early signal:

- `basic_5` models mostly driven by title/brand/size text plus `Price` and `Likes`.
- `full_scrape_plus_visual` add useful signal from full-scrape fields and readable photo-quality features, especially in `gucci`, `ps4`, `prada`.
- `CombinedBadPhotoScore`, sharpness, picture-count fields, description length, review/star fields, upload-age text among most useful non-basic signals.

## SearchCount/Page Ablation And Upload-Date Check

Latest ablation run:

```text
data/experiments/full_scrape_model/offline_runs/sold_status_feature_modalities_20260515_full_visual/ablation_analysis/no_search_count_page_20260515_233324/
```

Main files:

- `ablation_upload_date_report.md`: human-readable summary.
- `ablation_vs_original.csv`: original vs no-`SearchCount`/no-`Page` comparison.
- `upload_date_shap_summary.csv`: upload-date SHAP share by search and mode.
- `upload_date_bucket_distribution.csv`: sold-rate by parsed upload-age bucket.
- `upload_date_bucket_by_fullscrape_reason.csv`: upload-age bucket by collection reason.

Result summary:

- Remove `SearchCount` and `Page` usually keep strict-threshold precision, but cut PR AUC in several searches.
- Biggest PR AUC drops: `griffati_donna_all` and `griffati_uomo_all`; these fields help ranking quality there.
- `gucci`, `nike`, `prada`, `ps4` more robust, especially in `full_scrape_plus_visual`.
- Upload-date look important mostly for `gucci`, `ps4`, `prada`, but reason check show it entangled with dataset creation: `sold_backfill_stage` rows older, `unsold_balance_stage` rows include very recent listings.

## Upload_date Decision (implemented 2026-05-16)

`Upload_date` (text) and `Upload_date_days` (numeric) now **excluded by default** from `full_scrape` and `full_scrape_plus_visual` pools in `compare_feature_modalities.py`.

Reason: historical value entangled with dataset origin — `sold_backfill_stage` rows systematically older, `unsold_balance_stage` rows newer — so model learn dataset source not real freshness signal.

To reproduce old behaviour for comparison:

```bash
/home/ale/miniconda3/envs/vinted_scraper/bin/python scripts/experiments/current/full_scrape_model/compare_feature_modalities.py --all-searches --include-upload-date
```

For recommended clean run (default):

```bash
/home/ale/miniconda3/envs/vinted_scraper/bin/python scripts/experiments/current/full_scrape_model/compare_feature_modalities.py --all-searches
```

Future path: once live paper-trading collect `FirstSeenAt` timestamp per item, compute upload age at first observation time (`UploadAgeDaysAtObservation`) and add back as clean live feature.

## How To Rerun

### Build Datasets Only

```bash
/home/ale/miniconda3/envs/vinted_scraper/bin/python scripts/experiments/current/full_scrape_model/build_dataset.py --task sold_status --all-searches
```

### Run The Sold-Status Sweep For All Searches

```bash
/home/ale/miniconda3/envs/vinted_scraper/bin/python scripts/experiments/current/full_scrape_model/model_sweep.py --task sold_status --all-searches
```

### Run A Single Search

```bash
/home/ale/miniconda3/envs/vinted_scraper/bin/python scripts/experiments/current/full_scrape_model/model_sweep.py --task sold_status --search gucci
```

### Restrict To Specific Approaches

```bash
/home/ale/miniconda3/envs/vinted_scraper/bin/python scripts/experiments/current/full_scrape_model/model_sweep.py --task sold_status --search gucci --approach numeric_tree_v1 --approach linear_svm_calibrated_v1
```

### Robustness Mode

```bash
/home/ale/miniconda3/envs/vinted_scraper/bin/python scripts/experiments/current/full_scrape_model/model_sweep.py --task sold_status --all-searches --robustness
```

### Run Readable SHAP Analysis

```bash
/home/ale/miniconda3/envs/vinted_scraper/bin/python scripts/experiments/current/full_scrape_model/shap_analysis.py --max-background-rows 120 --max-explain-rows 180
```

By default explain `basic_5` and `full_scrape_plus_visual` models, skip raw DINO embedding dims in output.

### Run SearchCount/Page Ablation And Upload-Date Analysis

```bash
/home/ale/miniconda3/envs/vinted_scraper/bin/python scripts/experiments/current/full_scrape_model/ablation_upload_date_analysis.py --all-searches
```

## Known Constraints

- Sweep need at least `50` eligible rows and both label classes to train for a search.
- `visual_basic_v1` slower because derive image features.
- Full-scrape sweep offline only. No paper trading, timers, or live scraping.
- Output root fully separate from deal-finder output root.
- Current user intent: evaluation and comparison, not deployment.
- Tree-model SHAP need optional `shap` package in `vinted_scraper` env.

## Recommended Next Steps For Another Agent

If continue, most useful next steps:

1. Decide whether adopt full-scrape winners as new default per-search offline candidates.
2. Investigate why `nike` regressed vs old deal-finder sweep.
3. If need, produce single side-by-side CSV comparing old vs full-scrape best rows for kept searches.
4. If user change direction later, code can support `extreme_score`, but path stay dormant unless they ask again.