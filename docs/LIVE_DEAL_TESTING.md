# Live Deal Testing

This project uses paper-trading, not real buying.

Offline models are trained on saved `sold` vs `not sold` outcomes. Paper-trading is the proper test for whether high-ranked items sell quickly.

Paper-trading means:

1. Collect public first-page listing snapshots.
2. Rank all candidates immediately.
3. Save predictions.
4. Recheck later whether tracked items sold.
5. Evaluate whether high-ranked items sold within 2 hours, 12 hours, 2 days, or 7 days.

## Important Rule

Paper-trading scripts write only under:

```text
data/experiments/deal_finder/live_runs/
```

They must not update:

- `old_df.csv`
- `sold_df.csv`
- `unsold_df.csv`
- `big_raw.csv`
- production `eventual_sale_check/` files

## Commands

Dry-run the next collection:

```bash
/home/ale/miniconda3/envs/vinted_scraper/bin/python scripts/experiments/deal_finder/paper_trade_collect.py --qualified --max-searches 3 --dry-run
```

Collect first-page snapshots for qualified searches:

```bash
/home/ale/miniconda3/envs/vinted_scraper/bin/python scripts/experiments/deal_finder/paper_trade_collect.py --qualified --max-searches 3
```

Dry-run due rechecks:

```bash
/home/ale/miniconda3/envs/vinted_scraper/bin/python scripts/experiments/deal_finder/paper_trade_recheck.py --due --dry-run
```

Run due rechecks:

```bash
/home/ale/miniconda3/envs/vinted_scraper/bin/python scripts/experiments/deal_finder/paper_trade_recheck.py --due
```

Run due rechecks only for above-threshold items every hour:

```bash
/home/ale/miniconda3/envs/vinted_scraper/bin/python scripts/experiments/deal_finder/paper_trade_recheck.py --due --due-hours 1 --above-threshold-only
```

Run one selected-search collection:

```bash
/home/ale/miniconda3/envs/vinted_scraper/bin/python scripts/experiments/deal_finder/paper_trade_hourly_runner.py --offline-run data/experiments/deal_finder/offline_runs/offline_20260510_185509 --search nike --search gucci --max-searches 2 --iterations 1 --out-dir data/experiments/deal_finder/live_runs/hourly_nike_gucci_scheduled
```

Run one all-search, all-model benchmark pass:

```bash
/home/ale/miniconda3/envs/vinted_scraper/bin/python scripts/experiments/deal_finder/paper_trade_model_benchmark.py --sweep-run data/experiments/deal_finder/offline_runs/sweep_20260510_222252 --all-searches --iterations 1 --out-dir data/experiments/deal_finder/live_runs/hourly_all_models_benchmark_scheduled --enable-live-image-features
```

Dry-run the new six-search strict hourly experiment in its own folder:

```bash
/home/ale/miniconda3/envs/vinted_scraper/bin/python scripts/experiments/deal_finder/paper_trade_six_search_strict_hourly.py --dry-run --iterations 1 --out-dir data/experiments/deal_finder/live_runs/six_search_strict_hourly_smoke
```

Run one real iteration of the new six-search strict hourly experiment:

```bash
/home/ale/miniconda3/envs/vinted_scraper/bin/python scripts/experiments/deal_finder/paper_trade_six_search_strict_hourly.py --iterations 1 --out-dir data/experiments/deal_finder/live_runs/six_search_strict_hourly_20260514
```

Generate experiment-local reports and plots for that new run:

```bash
/home/ale/miniconda3/envs/vinted_scraper/bin/python scripts/experiments/deal_finder/report_six_search_strict_hourly.py --run-dir data/experiments/deal_finder/live_runs/six_search_strict_hourly_20260514
```

Check the current Nike/Gucci hourly timer:

```bash
systemctl --user status vinted-deal-nike-gucci-hourly.timer
```

Stop the current Nike/Gucci hourly timer:

```bash
systemctl --user stop vinted-deal-nike-gucci-hourly.timer
```

Check the all-search model benchmark timer:

```bash
systemctl --user status vinted-deal-all-models-benchmark-hourly.timer
```

Stop the all-search model benchmark timer:

```bash
systemctl --user stop vinted-deal-all-models-benchmark-hourly.timer
```

## Cadence

Default paper-trading cadence:

```text
every 1 hour
```

Default recheck cadence:

```text
above-threshold items every 1 hour in the active hourly runner
```

Top-50 tracked items can still be rechecked manually with the broader `paper_trade_recheck.py --due` command.

## Saved Fields

Each paper-trading snapshot stores:

- all collected candidates
- model probability
- rank position
- timestamp
- search name
- model version
- features available at ranking time
- whether the item is above the selected threshold

Per-search live thresholds can be adjusted without editing trained model artifacts by adding:

```text
threshold_overrides.json
```

inside a live-run folder. Example:

```json
{
  "threshold_overrides": {
    "gucci": 0.92
  }
}
```

This changes live selection only. The original offline threshold remains stored in the model metadata.

Tracked items also store outcome checkpoints:

- `sold_within_2h`
- `sold_within_12h`
- `sold_within_2d`
- `sold_within_7d`

## All-Model Benchmark

The all-model benchmark is separate from the Nike/Gucci runner. It scores each first-page snapshot with each saved sweep model for that search, then tracks three threshold levels per model:

- `strict`: the offline-selected threshold from the model metadata.
- `medium`: a moderately lower threshold.
- `loose`: a lower exploratory threshold for collecting more live evidence.

It writes to:

```text
data/experiments/deal_finder/live_runs/hourly_all_models_benchmark_scheduled/
```

Important files:

- `raw_snapshots/`: one public listing snapshot per search and collection time.
- `scores/`: one row per candidate, model, and threshold level.
- `summaries/`: selected-count summaries by search/model/threshold.
- `image_cache/`: cached primary listing images when `--enable-live-image-features` is used.
- `tracked_model_threshold_items.csv`: selected rows tracked separately by item, model, and threshold level.
- `benchmark_recheck_*.csv`: status checks for due tracked items.
- `events.jsonl` and `latest_status.json`: process status and errors.

`visual_basic_v1` is scored when `--enable-live-image-features` is used. The runner caches the primary catalog image under the experiment folder, computes `image_brightness`, `image_contrast`, `image_saturation`, `image_sharpness`, `image_aspect_ratio`, and related numeric features, then scores the visual model with the same feature columns used offline.

Feasibility result from 2026-05-11:

- One-search Gucci visual pass: 96 images, 96 feature rows, about 39 seconds, about 3.5M image cache.
- Full active-search visual pass: 576 images, 576 feature rows, about 4 minutes, about 21M image cache.
- The all-search benchmark timer was updated to include `--enable-live-image-features`.

## Six-Search Strict Hourly Experiment

The new six-search strict hourly experiment is separate from both the Nike/Gucci runner and the all-model benchmark. It uses the six-search set from the sweep best-by-search file after excluding `Borse_Griffate` and `Scarpe_Griffate`, then snapshots that exact search list into the run manifest.

It writes into a fresh run folder under:

```text
data/experiments/deal_finder/live_runs/six_search_strict_hourly_*/
```

Important files inside that new run folder:

- `raw_snapshots/`: one raw catalog capture per search and hour.
- `scored_snapshots/`: the full hourly score distribution for every candidate.
- `tracked_state.csv`: one current row per tracked item that has crossed the strict threshold.
- `hourly_history.csv`: append-only threshold-hit and recheck observations by hour.
- `rechecks/`: one csv per hourly sold-status recheck pass.
- `plots/` and `reports/`: experiment-local outputs for summaries and charts.

This flow is meant to keep legacy runs read-only while the new benchmark evolves in its own experiment folder.

## Active Paper-Trading

As of 2026-05-10, the active hourly paper-trading schedule is:

```text
vinted-deal-nike-gucci-hourly.timer
```

It writes to:

```text
data/experiments/deal_finder/live_runs/hourly_nike_gucci_scheduled/
```

As of 2026-05-11, this run has a live threshold override for `gucci`:

```text
0.9656 offline threshold -> 0.92 live threshold
```

As of 2026-05-11 18:17 CEST, the separate all-search benchmark schedule is:

```text
vinted-deal-all-models-benchmark-hourly.timer
```

It writes to:

```text
data/experiments/deal_finder/live_runs/hourly_all_models_benchmark_scheduled/
```

As of 2026-05-11 20:26 CEST, this timer includes live visual image features.
