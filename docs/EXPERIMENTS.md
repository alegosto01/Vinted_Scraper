> ⚠️ **Caveman-compressed** — terse/fragment style to save tokens. Technical substance, code, commands, URLs kept verbatim. Original backed up under `~/.local/share/caveman-compress/backups/`.

# Deal Finder Experiments

File track offline experiment framework for find high-precision Vinted deals.

## Goal

Find listings likely good deals.

Offline training label:

```text
sold vs not sold
```

Live paper-trading primary success label:

```text
sold within 2 days after being ranked
```

Live paper-trading secondary success label:

```text
sold within 7 days after being ranked
```

Saved historical data no reliable contain exact sale timestamps every item, so offline models no train on "sold within 2 days". 2-day target measured during paper-trading, where each prediction got clean timestamp.

Default model feature policy `snapshot_raw_v1`: train only on fields available in first-page catalog snapshot — price, likes, page, title, brand, size, search name. Pipeline-only fields like `DealScore`, `ExpectedProfit`, variant price stats no used unless live paper-trading collector also upgraded compute them before scoring.

## Safety Rules

- Work on branch `deal-experiment-runner`.
- Store experiment outputs under `data/experiments/deal_finder/`.
- No edit `.env` or private config files.
- No delete existing data.
- No purchases, messages, contact sellers, account actions.
- No write paper-trading snapshots into production tracking files.

## Offline Commands

Build normalized datasets:

```bash
/home/ale/miniconda3/envs/vinted_scraper/bin/python scripts/experiments/current/deal_finder/build_dataset.py --all-searches
```

Train and evaluate offline models:

```bash
/home/ale/miniconda3/envs/vinted_scraper/bin/python scripts/experiments/current/deal_finder/train_offline.py --all-searches
```

If one+ searches pass conservative promotion rule, command auto-starts paper-trading for up to 3 qualified searches.

Disable that behavior:

```bash
/home/ale/miniconda3/envs/vinted_scraper/bin/python scripts/experiments/current/deal_finder/train_offline.py --all-searches --no-auto-paper-trading
```

Generate report:

```bash
/home/ale/miniconda3/envs/vinted_scraper/bin/python scripts/experiments/current/deal_finder/report.py
```

Run offline multi-approach sweep:

```bash
/home/ale/miniconda3/envs/vinted_scraper/bin/python scripts/experiments/current/deal_finder/model_sweep.py --all-searches
```

Sweep offline-only. Write recommendation candidates but no start/stop paper-trading timers.

Run live all-model benchmark from completed sweep:

```bash
/home/ale/miniconda3/envs/vinted_scraper/bin/python scripts/experiments/current/deal_finder/paper_trade_model_benchmark.py --sweep-run data/experiments/deal_finder/offline_runs/sweep_20260510_222252 --all-searches --iterations 1 --out-dir data/experiments/deal_finder/live_runs/hourly_all_models_benchmark_scheduled
```

Benchmark separate from Nike/Gucci hourly runner. Collect one first-page snapshot per search, score every saved sweep model for same search, store three threshold variants per model: `strict`, `medium`, `loose`.

## Normal Live Scoring And Full Enrichment

Normal live scrape also score new rows with current best per-search model from:

```text
data/experiments/deal_finder/offline_runs/sweep_20260510_222252/best_by_search.csv
```

Rows get `DealFinderScore`, `DealFinderModel`, `DealFinderScoredAt`, `DealFinderScoreBand`.

Rows with `DealFinderScore <= 0.05` or `DealFinderScore >= 0.95` fully enriched into:

```text
data/simple_scrape/<search_name>/full_scrape/items_enriched.csv
```

Historical sold rows can backfill with:

```bash
/home/ale/miniconda3/envs/vinted_scraper/bin/python scripts/full_scrape_sold_history.py --all-searches --batch-size 100 --max-workers 2 --image-mode html
```

Use smoke run first when changing scraper:

```bash
/home/ale/miniconda3/envs/vinted_scraper/bin/python scripts/full_scrape_sold_history.py --all-searches --limit-per-search 5 --image-mode html
```

## Promotion Rule

Search/model qualified for paper-trading only if pass conservative gate:

- validation precision above threshold >= 60%
- test precision above threshold >= 60%
- validation precision@10 >= 60%
- test precision@10 >= 60%
- >= 20 validation items above threshold
- >= 10 test items above threshold
- threshold precision materially beat base positive rate

For multi-approach sweep, promotion gate intentionally simpler + stricter:

- validation precision above threshold >= 80%
- test precision above threshold >= 80%
- validation precision@10 >= 80%
- test precision@10 >= 80%
- >= 20 validation items above threshold
- >= 10 test items above threshold
- features must be computable at paper-trading time
- no leakage columns allowed

## Label Quality

Historical data sometimes lack exact sale timestamps.

Offline training main columns:

- `offline_sold_label`: 1 sold, 0 checked not sold.
- `offline_label_eligible`: true when row got explicit sold/not-sold outcome.

Timed fields like `fast_sale_2d` and `fast_sale_7d` kept only when timing available, but no the offline training target.

Framework also track:

- `exact`: row got direct check/sold timestamp.
- `approximate`: row got useful proxy timestamp (queue/enqueue or file-level active check time).
- `weak`: sale or active status exist, but timing too weak for primary claims.
- `unlabeled`: no usable outcome yet.

## Run Notes

### 2026-05-09 All-Search Raw Snapshot Model

Command:

```bash
/home/ale/miniconda3/envs/vinted_scraper/bin/python scripts/experiments/current/deal_finder/train_offline.py --all-searches
```

Run folder:

```text
data/experiments/deal_finder/offline_runs/offline_20260509_153253/
```

Result:

- `gucci` qualified for paper-trading.
- `gucci` validation precision above threshold: 80.95% on 21 recommendations.
- `gucci` test precision above threshold: 80.00% on 20 recommendations.
- `gucci` test precision@10: 80.00%.
- Threshold: 0.9656.
- Feature policy: `snapshot_raw_v1`.

Other searches no pass conservative gate. Several got useful threshold precision but fail precision@10 or sample-count checks.

Paper-trading snapshot:

```text
data/experiments/deal_finder/live_runs/paper_20260509_153851/
```

Snapshot result:

- `gucci` first-page candidates scored: 96.
- Items tracked: 50.
- Items above threshold in this snapshot: 0.
- Top item probability: 0.9273, below 0.9656 threshold.

Interpretation:

Offline result good enough continue paper-trading on `gucci`, but threshold intentionally very conservative. Normal that some hourly first-page snapshots produce no above-threshold items.

### 2026-05-10 Nike And Gucci Rerun

Command:

```bash
/home/ale/miniconda3/envs/vinted_scraper/bin/python scripts/experiments/current/deal_finder/train_offline.py --search nike --search gucci --no-auto-paper-trading
```

Run folder:

```text
data/experiments/deal_finder/offline_runs/offline_20260510_185509/
```

Result:

- `gucci` qualified again: validation precision 80.95% on 21 recs, test precision 80.00% on 20 recs, test precision@10 80.00%, threshold 0.9656.
- `nike` qualified after fix high-base-rate promotion gate: validation precision 100.00% on 88 recs, test precision 100.00% on 102 recs, test precision@10 100.00%, threshold 0.7707.
- Promotion gate now cap required material improvement over base rate, so high-base-rate searches no blocked by impossible `2x base rate` requirement.

Duplicate audit:

```text
data/experiments/deal_finder/reports/duplicate_audit_latest.md
```

Active `big_raw.csv` files contain small duplicate counts and deduped by experiment dataset builder before training. Largest duplicate counts in archived pipeline/evaluation outputs.

Paper-trading schedule:

```text
data/experiments/deal_finder/live_runs/hourly_nike_gucci_scheduled/
```

First scheduled collection saved 96 Gucci candidates + 96 Nike candidates. Schedule run once per hour via transient user systemd timer `vinted-deal-nike-gucci-hourly.timer`.

On 2026-05-11, active live run added `gucci` threshold override:

- Offline logistic threshold stay `0.9656`.
- Live paper-trading threshold now `0.92`.
- Override stored in `data/experiments/deal_finder/live_runs/hourly_nike_gucci_scheduled/threshold_overrides.json`.
- No retrain model or edit model artifacts; only make live candidate selection less strict so can collect usable Gucci above-threshold sample.

### 2026-05-10 Multi-Approach Offline Sweep

Command:

```bash
/home/ale/miniconda3/envs/vinted_scraper/bin/python scripts/experiments/current/deal_finder/model_sweep.py --all-searches --out-dir data/experiments/deal_finder/offline_runs/sweep_20260510_222252
```

Run folder:

```text
data/experiments/deal_finder/offline_runs/sweep_20260510_222252/
```

Approaches tried:

- `logistic_v1_baseline`
- `logistic_snapshot_v2`
- `sgd_text_numeric_v1`
- `linear_svm_calibrated_v1`
- `numeric_tree_v1`
- `rules_price_v1`
- `visual_basic_v1`

Sweep used stratified random 60/20/20 train/validation/test splits with seed `42`. Trained 56 approach/search rows and found 9 rows passing 80% promotion gate.

Promotion candidates:

| search | approach | threshold | validation precision | validation count | test precision | test count | test precision@10 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `gucci` | `visual_basic_v1` | 0.9232 | 86.96% | 23 | 86.36% | 22 | 90.00% |
| `nike` | `logistic_snapshot_v2` | 0.9025 | 95.00% | 20 | 85.00% | 20 | 80.00% |
| `nike` | `sgd_text_numeric_v1` | 0.8471 | 96.23% | 53 | 89.80% | 49 | 90.00% |
| `nike` | `linear_svm_calibrated_v1` | 0.7665 | 96.30% | 27 | 92.00% | 25 | 90.00% |
| `nike` | `numeric_tree_v1` | 0.8355 | 86.36% | 22 | 90.91% | 22 | 80.00% |
| `nike` | `visual_basic_v1` | 0.8634 | 86.36% | 22 | 90.91% | 22 | 90.00% |
| `prada` | `visual_basic_v1` | 0.9649 | 100.00% | 29 | 100.00% | 24 | 100.00% |
| `ps4` | `numeric_tree_v1` | 0.9000 | 100.00% | 32 | 88.89% | 27 | 90.00% |
| `ps4` | `visual_basic_v1` | 0.9569 | 95.65% | 23 | 85.71% | 28 | 100.00% |

Recommended next paper-trading candidates:

- `nike` stay strong supported. Best non-visual sweep row `linear_svm_calibrated_v1`, 92.00% test precision + 90.00% test precision@10.
- `ps4` strongest new non-visual candidate, using `numeric_tree_v1` with 88.89% test precision + 90.00% test precision@10.
- `gucci` already run live with Nike. Sweep found visual candidate, but visual result treated as secondary until paper-trading scoring compute same image metrics.
- `prada` look excellent in visual sweep, but no add live until live scorer support same image-derived numeric features.

Known limitations:

- Offline labels still sold/not-sold, no "sold within 2 days". Clean 2-day success label only come from paper-trading.
- `visual_basic_v1` bounded to 3,000-row stratified offline sample for large searches to avoid decode every cached image during every sweep.
- Visual promotion candidates therefore useful signals, but less directly comparable than full-data non-visual candidates.
- No new live search started by this sweep.

### 2026-05-11 All-Search Model Benchmark

All-search benchmark runner:

```text
scripts/experiments/current/deal_finder/paper_trade_model_benchmark.py
```

Designed for compare live behavior across searches + approaches without change production CSV files. Each selected candidate tracked using:

```text
SearchName + item_id + approach + threshold_label
```

Keep results independent when same item selected by multiple models or thresholds.

Default threshold variants:

- `strict`: saved offline threshold.
- `medium`: up to 0.05 lower than strict.
- `loose`: up to 0.10 lower than strict.

Output folder:

```text
data/experiments/deal_finder/live_runs/hourly_all_models_benchmark_scheduled/
```

Benchmark record image-based models as skipped when live snapshots no contain same image-derived numeric features used offline. Keep comparison honest while still document model existed in sweep.

First clean run on 2026-05-11:

- Scored rows: 10,368.
- Selected model-threshold rows: 621.
- Search folders collected: `griffati_donna_all`, `griffati_uomo_all`, `gucci`, `nike`, `prada`, `ps4`.
- Search folders skipped for collection: `Borse_Griffate`, `Scarpe_Griffate`, because currently no got active search/category settings in `data/searches.yaml`.
- Skipped model family: `visual_basic_v1`, because live snapshots no yet include matching image-derived numeric features.
- Active timer: `vinted-deal-all-models-benchmark-hourly.timer`, created separately from Nike/Gucci timer.

Visual-feature feasibility on 2026-05-11:

- Implemented optional live image features with `--enable-live-image-features`.
- Images cached only under experiment live-run folder, no production scrape folders.
- Gucci-only pass: 96 primary catalog images, 96 feature rows, ~39 seconds, ~3.5M image cache.
- Full active-search pass: 576 primary catalog images, 576 feature rows, ~4 minutes, ~21M image cache.
- Result: feasible for hourly benchmark, but disk use must be monitored because image cache can grow ~tens of MB per hourly pass.
- Active all-model benchmark timer now include `--enable-live-image-features`.

### 2026-05-25 Basic 5 Giant Model

Basic5 giant-model experiment train one global sold/not-sold model across six active searches:

```bash
/home/ale/miniconda3/envs/vinted_scraper/bin/python scripts/experiments/current/basic_5_giant_model/run.py
```

Evaluate nine Basic5 approach families in one combined `SearchName x label` stratified train/validation/test split. Inputs: `Price`, `Likes`, `Title`, `Brand`, `Size`, plus one-hot `SearchName` features so global model learn search-level differences without using full-scrape or visual fields.

Outputs written under:

```text
data/experiments/basic_5_giant_model/offline_runs/
```

First full run:

```text
data/experiments/basic_5_giant_model/offline_runs/basic_5_giant_20260525_185552/
```

Result summary:

- Rows: 34,347 total; 20,608 train; 6,869 validation; 6,870 test.
- Best global AUC: `xgboost_basic_v1` with AUC `0.744`, PR AUC `0.762`, P@25 `1.000`.
- Best fast numeric model: `hist_gradient_basic_numeric_v1` with AUC `0.737`, PR AUC `0.756`, P@25 `1.000`, much shorter fit time.
- All approaches except `rules_price_v1` passed existing global promotion gate.
- Global threshold conservative and uneven by search: selected many PS4 items but almost no `griffati_uomo_all` items, so follow-up should test per-search thresholds on giant model scores.

Per-search threshold follow-up:

```bash
/home/ale/miniconda3/envs/vinted_scraper/bin/python scripts/experiments/current/basic_5_giant_model/report_per_search_thresholds.py --run-dir data/experiments/basic_5_giant_model/offline_runs/basic_5_giant_20260525_185552
```

Outputs:

```text
data/experiments/basic_5_giant_model/offline_runs/basic_5_giant_20260525_185552/per_search_threshold_report.md
data/experiments/basic_5_giant_model/offline_runs/basic_5_giant_20260525_185552/per_search_threshold_metrics.csv
data/experiments/basic_5_giant_model/offline_runs/basic_5_giant_20260525_185552/per_search_threshold_comparison.csv
```

Main finding:

- `xgboost_basic_v1` with per-search thresholds selected 290 held-out rows with 276 positives, precision `0.952`.
- Single global XGBoost threshold had selected 140 held-out rows with precision `0.993`.
- Per-search thresholds fix coverage gap: `griffati_uomo_all` moved from 0 selected rows to 25 selected rows at precision `0.800`; `prada` moved from 13 to 93 selected rows at precision `0.957`.
- Best precision/count row by search used `hist_gradient_basic_numeric_v1` for both Griffati searches, `xgboost_basic_v1` for Gucci/Nike/PS4, `sgd_text_numeric_v1` for Prada.

Weighted-voting follow-up:

```bash
/home/ale/miniconda3/envs/vinted_scraper/bin/python scripts/experiments/current/basic_5_giant_model/report_weighted_voting.py --run-dir data/experiments/basic_5_giant_model/offline_runs/basic_5_giant_20260525_185552
```

Best weighted-voting row:

- `auc_hard_weighted_vote` selected 258 held-out rows with 244 positives, precision `0.946`.
- No beat `xgboost_basic_v1` with per-search thresholds, which selected 290 rows at precision `0.952`.
- Detail tracking moved to `docs/BASIC_5_GIANT_MODEL.md`.