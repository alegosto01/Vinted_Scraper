# Deal Finder Experiments

This file tracks the offline experiment framework for finding high-precision Vinted deals.

## Goal

Find listings that are likely to be good deals.

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

The saved historical data does not reliably contain exact sale timestamps for every item, so offline models should not be trained on "sold within 2 days". The 2-day target is measured during paper-trading, where each prediction has a clean timestamp.

The default model feature policy is `snapshot_raw_v1`: train only on fields available in a first-page catalog snapshot, such as price, likes, page, title, brand, size, and search name. Pipeline-only fields such as `DealScore`, `ExpectedProfit`, or variant price statistics are not used unless the live paper-trading collector is also upgraded to compute them before scoring.

## Safety Rules

- Work on branch `deal-experiment-runner`.
- Store experiment outputs under `data/experiments/deal_finder/`.
- Do not edit `.env` or private configuration files.
- Do not delete existing data.
- Do not make purchases, send messages, contact sellers, or perform account actions.
- Do not write paper-trading snapshots into production tracking files.

## Offline Commands

Build normalized datasets:

```bash
/home/ale/miniconda3/envs/vinted_scraper/bin/python scripts/experiments/deal_finder/build_dataset.py --all-searches
```

Train and evaluate offline models:

```bash
/home/ale/miniconda3/envs/vinted_scraper/bin/python scripts/experiments/deal_finder/train_offline.py --all-searches
```

If one or more searches pass the conservative promotion rule, this command automatically starts paper-trading for up to 3 qualified searches.

To disable that behavior:

```bash
/home/ale/miniconda3/envs/vinted_scraper/bin/python scripts/experiments/deal_finder/train_offline.py --all-searches --no-auto-paper-trading
```

Generate report:

```bash
/home/ale/miniconda3/envs/vinted_scraper/bin/python scripts/experiments/deal_finder/report.py
```

Run the offline multi-approach sweep:

```bash
/home/ale/miniconda3/envs/vinted_scraper/bin/python scripts/experiments/deal_finder/model_sweep.py --all-searches
```

The sweep is offline-only. It writes recommendation candidates but does not start or stop paper-trading timers.

Run the live all-model benchmark from a completed sweep:

```bash
/home/ale/miniconda3/envs/vinted_scraper/bin/python scripts/experiments/deal_finder/paper_trade_model_benchmark.py --sweep-run data/experiments/deal_finder/offline_runs/sweep_20260510_222252 --all-searches --iterations 1 --out-dir data/experiments/deal_finder/live_runs/hourly_all_models_benchmark_scheduled
```

The benchmark is separate from the Nike/Gucci hourly runner. It collects one first-page snapshot per search, scores every saved sweep model for that same search, and stores three threshold variants per model: `strict`, `medium`, and `loose`.

## Normal Live Scoring And Full Enrichment

Normal live scraping also scores newly discovered rows with the current best per-search model from:

```text
data/experiments/deal_finder/offline_runs/sweep_20260510_222252/best_by_search.csv
```

Rows get `DealFinderScore`, `DealFinderModel`, `DealFinderScoredAt`, and `DealFinderScoreBand`.

Rows with `DealFinderScore <= 0.05` or `DealFinderScore >= 0.95` are fully enriched into:

```text
data/simple_scrape/<search_name>/full_scrape/items_enriched.csv
```

Historical sold rows can be backfilled with:

```bash
/home/ale/miniconda3/envs/vinted_scraper/bin/python scripts/full_scrape_sold_history.py --all-searches --batch-size 100 --max-workers 2 --image-mode html
```

Use a smoke run first when changing the scraper:

```bash
/home/ale/miniconda3/envs/vinted_scraper/bin/python scripts/full_scrape_sold_history.py --all-searches --limit-per-search 5 --image-mode html
```

## Promotion Rule

A search/model is qualified for paper-trading only if it passes the conservative gate:

- validation precision above threshold is at least 60%
- test precision above threshold is at least 60%
- validation precision@10 is at least 60%
- test precision@10 is at least 60%
- at least 20 validation items above threshold
- at least 10 test items above threshold
- threshold precision materially beats the base positive rate

For the multi-approach sweep, the promotion gate is intentionally simpler and stricter:

- validation precision above threshold is at least 80%
- test precision above threshold is at least 80%
- validation precision@10 is at least 80%
- test precision@10 is at least 80%
- at least 20 validation items above threshold
- at least 10 test items above threshold
- features must be computable at paper-trading time
- no leakage columns are allowed

## Label Quality

Historical data may not always contain exact sale timestamps.

For offline training, the main columns are:

- `offline_sold_label`: 1 for sold, 0 for checked not sold.
- `offline_label_eligible`: true when a row has an explicit sold/not-sold outcome.

Timed fields such as `fast_sale_2d` and `fast_sale_7d` are kept only when timing is available, but they are not the offline training target.

The framework also tracks:

- `exact`: row has a direct check/sold timestamp.
- `approximate`: row has a useful proxy timestamp, such as queue/enqueue or file-level active check time.
- `weak`: sale or active status exists, but timing is not strong enough for primary claims.
- `unlabeled`: no usable outcome yet.

## Run Notes

### 2026-05-09 All-Search Raw Snapshot Model

Command:

```bash
/home/ale/miniconda3/envs/vinted_scraper/bin/python scripts/experiments/deal_finder/train_offline.py --all-searches
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

Other searches did not pass the conservative gate. Several had useful threshold precision but failed precision@10 or sample-count checks.

Paper-trading snapshot:

```text
data/experiments/deal_finder/live_runs/paper_20260509_153851/
```

Snapshot result:

- `gucci` first-page candidates scored: 96.
- Items tracked: 50.
- Items above threshold in this snapshot: 0.
- Top item probability: 0.9273, below the 0.9656 threshold.

Interpretation:

The offline result is good enough to continue paper-trading on `gucci`, but the threshold is intentionally very conservative. It is normal that some hourly first-page snapshots may produce no above-threshold items.

### 2026-05-10 Nike And Gucci Rerun

Command:

```bash
/home/ale/miniconda3/envs/vinted_scraper/bin/python scripts/experiments/deal_finder/train_offline.py --search nike --search gucci --no-auto-paper-trading
```

Run folder:

```text
data/experiments/deal_finder/offline_runs/offline_20260510_185509/
```

Result:

- `gucci` qualified again: validation precision 80.95% on 21 recommendations, test precision 80.00% on 20 recommendations, test precision@10 80.00%, threshold 0.9656.
- `nike` qualified after fixing the high-base-rate promotion gate: validation precision 100.00% on 88 recommendations, test precision 100.00% on 102 recommendations, test precision@10 100.00%, threshold 0.7707.
- The promotion gate now caps the required material improvement over base rate, so high-base-rate searches are not blocked by an impossible `2x base rate` requirement.

Duplicate audit:

```text
data/experiments/deal_finder/reports/duplicate_audit_latest.md
```

The active `big_raw.csv` files contain small duplicate counts and are deduped by the experiment dataset builder before training. The largest duplicate counts are in archived pipeline/evaluation outputs.

Paper-trading schedule:

```text
data/experiments/deal_finder/live_runs/hourly_nike_gucci_scheduled/
```

The first scheduled collection saved 96 Gucci candidates and 96 Nike candidates. The schedule runs once per hour via the transient user systemd timer `vinted-deal-nike-gucci-hourly.timer`.

On 2026-05-11, the active live run added a `gucci` threshold override:

- Offline logistic threshold remains `0.9656`.
- Live paper-trading threshold is now `0.92`.
- The override is stored in `data/experiments/deal_finder/live_runs/hourly_nike_gucci_scheduled/threshold_overrides.json`.
- This does not retrain the model or edit model artifacts; it only makes live candidate selection less strict so we can collect a usable Gucci above-threshold sample.

### 2026-05-10 Multi-Approach Offline Sweep

Command:

```bash
/home/ale/miniconda3/envs/vinted_scraper/bin/python scripts/experiments/deal_finder/model_sweep.py --all-searches --out-dir data/experiments/deal_finder/offline_runs/sweep_20260510_222252
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

The sweep used stratified random 60/20/20 train/validation/test splits with seed `42`. It trained 56 approach/search rows and found 9 rows passing the 80% promotion gate.

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

- `nike` remains strongly supported. The best non-visual sweep row was `linear_svm_calibrated_v1`, with 92.00% test precision and 90.00% test precision@10.
- `ps4` is the strongest new non-visual candidate, using `numeric_tree_v1` with 88.89% test precision and 90.00% test precision@10.
- `gucci` is already running live with Nike. The sweep found a visual candidate, but the visual result should be treated as secondary until paper-trading scoring computes the same image metrics.
- `prada` looks excellent in the visual sweep, but it should not be added live until the live scorer supports the same image-derived numeric features.

Known limitations:

- Offline labels are still sold/not-sold, not "sold within 2 days". The clean 2-day success label only comes from paper-trading.
- `visual_basic_v1` was bounded to a 3,000-row stratified offline sample for large searches to avoid decoding every cached image during every sweep.
- Visual promotion candidates are therefore useful signals, but less directly comparable than full-data non-visual candidates.
- No new live search was started by this sweep.

### 2026-05-11 All-Search Model Benchmark

The all-search benchmark runner is:

```text
scripts/experiments/deal_finder/paper_trade_model_benchmark.py
```

It is designed for comparing live behavior across searches and approaches without changing production CSV files. Each selected candidate is tracked using:

```text
SearchName + item_id + approach + threshold_label
```

This keeps results independent when the same item is selected by multiple models or thresholds.

Default threshold variants:

- `strict`: the saved offline threshold.
- `medium`: up to 0.05 lower than strict.
- `loose`: up to 0.10 lower than strict.

Output folder:

```text
data/experiments/deal_finder/live_runs/hourly_all_models_benchmark_scheduled/
```

The benchmark records image-based models as skipped when live snapshots do not contain the same image-derived numeric features used offline. This keeps the comparison honest while still documenting that the model existed in the sweep.

First clean run on 2026-05-11:

- Scored rows: 10,368.
- Selected model-threshold rows: 621.
- Search folders collected: `griffati_donna_all`, `griffati_uomo_all`, `gucci`, `nike`, `prada`, `ps4`.
- Search folders skipped for collection: `Borse_Griffate`, `Scarpe_Griffate`, because they currently do not have active search/category settings in `data/searches.yaml`.
- Skipped model family: `visual_basic_v1`, because live snapshots do not yet include the matching image-derived numeric features.
- Active timer: `vinted-deal-all-models-benchmark-hourly.timer`, created separately from the Nike/Gucci timer.

Visual-feature feasibility on 2026-05-11:

- Implemented optional live image features with `--enable-live-image-features`.
- Images are cached only under the experiment live-run folder, not production scrape folders.
- Gucci-only pass: 96 primary catalog images, 96 feature rows, about 39 seconds, about 3.5M image cache.
- Full active-search pass: 576 primary catalog images, 576 feature rows, about 4 minutes, about 21M image cache.
- Result: feasible for hourly benchmarking, but disk use should be monitored because image cache can grow by roughly tens of MB per hourly pass.
- The active all-model benchmark timer now includes `--enable-live-image-features`.
