> ⚠️ **Caveman-compressed** — terse/fragment style to save tokens. Technical substance, code, commands, URLs kept verbatim. Original backed up under `~/.local/share/caveman-compress/backups/`.

# Live Deal Testing

Project use paper-trade, no real buy.

Offline model train on saved `sold` vs `not sold`. Paper-trade = proper test if high-rank item sell fast.

Paper-trade mean:

1. Grab public first-page snapshot.
2. Rank all candidate now.
3. Save prediction.
4. Recheck later if tracked item sold.
5. Eval if high-rank sold in 2h, 12h, 2d, or 7d.

## Important Rule

Paper-trade script write only under:

```text
data/experiments/deal_finder/live_runs/
```

No touch:

- `old_df.csv`
- `sold_df.csv`
- `unsold_df.csv`
- `big_raw.csv`
- prod `eventual_sale_check/` files

## Commands

Dry-run next collect:

```bash
/home/ale/miniconda3/envs/vinted_scraper/bin/python scripts/experiments/current/deal_finder/paper_trade_collect.py --qualified --max-searches 3 --dry-run
```

Collect first-page snapshot for qualified search:

```bash
/home/ale/miniconda3/envs/vinted_scraper/bin/python scripts/experiments/current/deal_finder/paper_trade_collect.py --qualified --max-searches 3
```

Dry-run due recheck:

```bash
/home/ale/miniconda3/envs/vinted_scraper/bin/python scripts/experiments/current/deal_finder/paper_trade_recheck.py --due --dry-run
```

Run due recheck:

```bash
/home/ale/miniconda3/envs/vinted_scraper/bin/python scripts/experiments/current/deal_finder/paper_trade_recheck.py --due
```

Run due recheck only for above-threshold item every hour:

```bash
/home/ale/miniconda3/envs/vinted_scraper/bin/python scripts/experiments/current/deal_finder/paper_trade_recheck.py --due --due-hours 1 --above-threshold-only
```

Run one selected-search collect:

```bash
/home/ale/miniconda3/envs/vinted_scraper/bin/python scripts/experiments/current/deal_finder/paper_trade_hourly_runner.py --offline-run data/experiments/deal_finder/offline_runs/offline_20260510_185509 --search nike --search gucci --max-searches 2 --iterations 1 --out-dir data/experiments/deal_finder/live_runs/hourly_nike_gucci_scheduled
```

Run one all-search, all-model benchmark pass:

```bash
/home/ale/miniconda3/envs/vinted_scraper/bin/python scripts/experiments/current/deal_finder/paper_trade_model_benchmark.py --sweep-run data/experiments/deal_finder/offline_runs/sweep_20260510_222252 --all-searches --iterations 1 --out-dir data/experiments/deal_finder/live_runs/hourly_all_models_benchmark_scheduled --enable-live-image-features
```

Dry-run new six-search strict hourly exp in own folder:

```bash
/home/ale/miniconda3/envs/vinted_scraper/bin/python scripts/experiments/current/deal_finder/paper_trade_six_search_strict_hourly.py --dry-run --iterations 1 --out-dir data/experiments/deal_finder/live_runs/six_search_strict_hourly_smoke
```

Run one real iter of new six-search strict hourly exp:

```bash
/home/ale/miniconda3/envs/vinted_scraper/bin/python scripts/experiments/current/deal_finder/paper_trade_six_search_strict_hourly.py --iterations 1 --out-dir data/experiments/deal_finder/live_runs/six_search_strict_hourly_20260514
```

Make exp-local report + plot for that new run:

```bash
/home/ale/miniconda3/envs/vinted_scraper/bin/python scripts/experiments/current/deal_finder/report_six_search_strict_hourly.py --run-dir data/experiments/deal_finder/live_runs/six_search_strict_hourly_20260514
```

Check Nike/Gucci hourly timer:

```bash
systemctl --user status vinted-deal-nike-gucci-hourly.timer
```

Stop Nike/Gucci hourly timer:

```bash
systemctl --user stop vinted-deal-nike-gucci-hourly.timer
```

Check all-search model benchmark timer:

```bash
systemctl --user status vinted-deal-all-models-benchmark-hourly.timer
```

Stop all-search model benchmark timer:

```bash
systemctl --user stop vinted-deal-all-models-benchmark-hourly.timer
```

## Cadence

Default paper-trade cadence:

```text
every 1 hour
```

Default recheck cadence:

```text
above-threshold items every 1 hour in the active hourly runner
```

Top-50 tracked item still recheck manual via broader `paper_trade_recheck.py --due` cmd.

## Saved Fields

Each paper-trade snapshot store:

- all candidate
- model prob
- rank pos
- timestamp
- search name
- model ver
- features at rank time
- if item above selected threshold

Per-search live threshold tweak no edit model artifact — add:

```text
threshold_overrides.json
```

inside live-run folder. Example:

```json
{
  "threshold_overrides": {
    "gucci": 0.92
  }
}
```

Change live select only. Offline threshold stay in model meta.

Tracked item also store outcome checkpoint:

- `sold_within_2h`
- `sold_within_12h`
- `sold_within_2d`
- `sold_within_7d`

## All-Model Benchmark

All-model benchmark separate from Nike/Gucci runner. Score each first-page snapshot with each saved sweep model for that search, track three threshold levels per model:

- `strict`: offline-selected threshold from model meta.
- `medium`: moderately lower threshold.
- `loose`: lower exploratory threshold for more live evidence.

Write to:

```text
data/experiments/deal_finder/live_runs/hourly_all_models_benchmark_scheduled/
```

Key files:

- `raw_snapshots/`: one public listing snapshot per search and collect time.
- `scores/`: one row per candidate, model, threshold level.
- `summaries/`: selected-count summary by search/model/threshold.
- `image_cache/`: cached primary listing image when `--enable-live-image-features` used.
- `tracked_model_threshold_items.csv`: selected rows tracked separate by item, model, threshold level.
- `benchmark_recheck_*.csv`: status check for due tracked item.
- `events.jsonl` and `latest_status.json`: process status + error.

`visual_basic_v1` score when `--enable-live-image-features` used. Runner cache primary catalog image under exp folder, compute `image_brightness`, `image_contrast`, `image_saturation`, `image_sharpness`, `image_aspect_ratio`, related numeric features, then score visual model with same feature cols used offline.

Feasibility result 2026-05-11:

- One-search Gucci visual pass: 96 image, 96 feat row, ~39s, ~3.5M image cache.
- Full active-search visual pass: 576 image, 576 feat row, ~4 min, ~21M image cache.
- All-search benchmark timer updated to include `--enable-live-image-features`.

## Six-Search Strict Hourly Experiment

New six-search strict hourly exp separate from both Nike/Gucci runner and all-model benchmark. Use six-search set from sweep best-by-search file after exclude `Borse_Griffate` and `Scarpe_Griffate`, then snapshot exact search list into run manifest.

Write into fresh run folder under:

```text
data/experiments/deal_finder/live_runs/six_search_strict_hourly_*/
```

Key files inside new run folder:

- `raw_snapshots/`: one raw catalog capture per search and hour.
- `scored_snapshots/`: full hourly score dist for every candidate.
- `tracked_state.csv`: one current row per tracked item that cross strict threshold.
- `hourly_history.csv`: append-only threshold-hit + recheck obs by hour.
- `rechecks/`: one csv per hourly sold-status recheck pass.
- `plots/` and `reports/`: exp-local output for summary + chart.

Flow keep legacy run read-only while new benchmark evolve in own exp folder.

## Active Paper-Trading

As of 2026-05-10, active hourly paper-trade schedule:

```text
vinted-deal-nike-gucci-hourly.timer
```

Write to:

```text
data/experiments/deal_finder/live_runs/hourly_nike_gucci_scheduled/
```

As of 2026-05-11, run has live threshold override for `gucci`:

```text
0.9656 offline threshold -> 0.92 live threshold
```

As of 2026-05-11 18:17 CEST, separate all-search benchmark schedule:

```text
vinted-deal-all-models-benchmark-hourly.timer
```

Write to:

```text
data/experiments/deal_finder/live_runs/hourly_all_models_benchmark_scheduled/
```

As of 2026-05-11 20:26 CEST, timer include live visual image feature.