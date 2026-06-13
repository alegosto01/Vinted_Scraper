# Basic + Visual

## Goal

This experiment tests a **single-stage** sold-status predictor that combines basic
listing fields with image-based visual features (photo quality + DINO embeddings).
It is the simpler counterpart to the two-stage cascade in
[`BENCHMARK_BASIC_TO_FULL.md`](BENCHMARK_BASIC_TO_FULL.md) — no expensive
full-item-page scrape, no second-stage teacher, just one model per search that
scores the raw catalog snapshot enriched with photo features and DINO
embeddings of the primary image.

The experiment is isolated under:

```text
data/experiments/basic_plus_visual/
```

It does not write to production `old_df.csv`, `sold_df.csv`, `unsold_df.csv`, or
normal full-scrape files.

## Models

One trained model per search lives in:

```text
data/experiments/basic_plus_visual/models/*_extra_trees.json
```

Each metadata file references its pickled artifact and stores the
validation-tuned probability threshold. Current thresholds:

| Search | Threshold |
| --- | ---: |
| `griffati_donna_all` | `0.9704` |
| `griffati_uomo_all` | `0.9674` |
| `gucci` | `0.8940` |
| `nike` | `0.9919` |
| `prada` | `0.9507` |
| `ps4` | `0.9231` |

Thresholds are not comparable across searches — they encode each model's
calibration. The selection logic below accounts for that.

## Pipeline (per search, per hourly cycle)

1. Scrape the catalog snapshot (≈96 items per search, page 1).
2. Download the primary image for each item into the run's `image_cache/`.
3. Extract photo features (`photo_arbitrage.features.add_photo_features`).
4. Run the configured quality methods (`simple,aesthetic,dino`) and flatten the
   DINO embedding column into 384 per-dim columns (`DinoEmbedding_0000..0383`)
   via `add_dino_embedding_columns`. The trained model expects these flat
   columns — skipping this step is the bug that aborted every cycle in the
   original runner.
5. Score the resulting frame with the per-search model. This sets:
   - `SoldProba` — model probability
   - `SoldThreshold` — the per-search threshold from metadata
   - `SoldPred` — `SoldProba >= SoldThreshold`
6. Append the search's scored rows to a global candidate pool.

## Selection (global, margin-based)

After all 6 searches finish in a cycle, the runner pools their scored items and
selects the top N for tracking. Selection uses **margin above threshold**, not
raw probability:

```
SoldMargin = SoldProba - SoldThreshold       # only for SoldPred = True
selected   = pool.sort_by(SoldMargin desc).head(max_items_total)
```

Rationale: a `0.91` score on a `0.80` threshold is a much stronger signal than
`0.91` on a `0.90` threshold, even though both have the same `SoldProba`.
Ranking by margin makes scores comparable across searches with different
calibrations, and prevents one over-confident model from crowding out the rest.

Defaults: `--max-items-total 30` items kept globally per cycle. Tracked items
are deduplicated by `(Dataid, SearchName)` with the most recent `_tracked_at`
winning (see `merge_tracked` in [`runner.py`](../scripts/experiments/old/basic_plus_visual/runner.py)).

## Recheck Schedule

After selection, every tracked item is re-fetched on Vinted to detect a sold
transition. `run_recheck_due` is called every loop iteration (≈every
`--sleep-seconds`), but only items whose `_last_checked_at` is older than 1
hour (or null) are checked. Up to 50 items per call.

The recheck updates:

- `LastCheckStatus` — `OnSale`, `Sold`, `FetchFailed`, etc.
- `_last_checked_at` — UTC timestamp of the check
- `ObservedPagePrice` — current page price (if seen)

Each recheck call writes `reports/recheck_<ts>.csv` and emits an
`event: recheck` entry to `events.jsonl`.

## Commands

Single collection pass:

```bash
/home/ale/miniconda3/envs/vinted_scraper/bin/python scripts/experiments/old/basic_plus_visual/runner.py collect-once
```

Single recheck pass for items due:

```bash
/home/ale/miniconda3/envs/vinted_scraper/bin/python scripts/experiments/old/basic_plus_visual/runner.py recheck-due
```

Hourly run-loop (production):

```bash
/home/ale/miniconda3/envs/vinted_scraper/bin/python scripts/experiments/old/basic_plus_visual/runner.py run-loop --collect-every-hours 1 --max-items-total 30
```

Recommended invocation when restarting after a crash (keeps stdout flushing so
errors surface in the log immediately):

```bash
PYTHONUNBUFFERED=1 nohup /home/ale/miniconda3/envs/vinted_scraper/bin/python \
  scripts/experiments/old/basic_plus_visual/runner.py run-loop \
  >> data/experiments/basic_plus_visual/run-loop.log 2>&1 &
disown
```

To resume an existing run directory (preserves `tracked_items.csv` and recheck
history), pass `--out-dir` explicitly:

```bash
--out-dir data/experiments/basic_plus_visual/live_runs/<run_id>
```

Without `--out-dir`, a fresh timestamped run dir is created on every launch.

## Run Directory Layout

```text
data/experiments/basic_plus_visual/live_runs/<run_id>/
├── raw_snapshots/<search>_<ts>.csv         # raw catalog scrape per cycle
├── image_cache/<search>/<item_id>/primary.<ext>
├── scored_items/<search>_<ts>.csv          # post-scoring frame per cycle per search
├── reports/
│   ├── collect_summary_<ts>.csv            # per-cycle status table
│   └── recheck_<ts>.csv                    # per-recheck-call results
├── tracked_items.csv                       # union of selected items + recheck state
├── events.jsonl                            # append-only event log
└── latest_status.json                      # most recent collect-cycle summary
```

## Metrics

The main metric is **realised sold rate** among tracked items:

```text
sold% = count(LastCheckStatus == "Sold") / count(tracked items rechecked)
```

Show live results with the unique-item, matured-cohort format described in
[`BASIC_PLUS_VISUAL_REPORTING.md`](BASIC_PLUS_VISUAL_REPORTING.md).

The `events.jsonl` log allows per-search breakdowns. The recheck horizon is
implicit — items stay in `tracked_items.csv` indefinitely until the run dir is
archived, so realised sold% accumulates over time.

Compare these numbers against the cascade's stage-2 realised sold% to judge
whether the basic+visual single-stage model is competitive with the two-stage
cascade at lower compute cost.

## Notes

- The 6 trained searches are `griffati_donna_all`, `griffati_uomo_all`, `gucci`,
  `nike`, `prada`, `ps4`. Searches enabled in `data/searches.yaml` without a
  trained model are silently skipped.
- The runner has no stage-2 / full-scrape step, no item-page fetch, and no
  paper-trade logic. It is intentionally cheap.
- `--quality-methods` controls which features are extracted. The DINO step is
  required because the trained models depend on `DinoEmbedding_0000..0383` —
  omitting `dino` will break scoring.
- The `_ensure_local_primary_images` helper in
  [`runner.py`](../scripts/experiments/old/basic_plus_visual/runner.py) intentionally
  does its own image download rather than reusing
  `cascade_runner.ensure_local_primary_images`, because the cascade helper
  asserts the path is under `benchmark_basic_to_full/` and would reject writes
  into `basic_plus_visual/`.
