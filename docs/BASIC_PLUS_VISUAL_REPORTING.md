> ⚠️ **Caveman-compressed** — terse/fragment style to save tokens. Technical substance, code, commands, URLs kept verbatim. Original backed up under `~/.local/share/caveman-compress/backups/`.

# Basic + Visual Results Reporting

Doc define preferred way show live results for `basic_plus_visual` experiment.

Use this report style when ask show Basic + Visual model results.

## Goal

Report answer these clearly:

- How many unique items scored?
- How many unique items pass model threshold?
- How many unique items tracked + rechecked?
- How many tracked items sold?
- What threshold used per search?
- Realised precision after 1h, 2h, 3h, 6h, 12h, 18h, 24h, later matured windows?
- Which elapsed-hour buckets tracked sold items actually sold?

## Single-Stage Mapping

Basic + Visual = single-stage model. No second model.

Use these names in reports:

| Cascade-style idea | Basic + Visual equivalent |
| --- | --- |
| Model 1 pass | `SoldPred == true`, meaning `SoldProba >= SoldThreshold`. |
| Model 2 pass | Not applicable. |
| Final selected set | `tracked_items.csv`, meaning selected items that are rechecked. |
| Final precision | Sold rate among unique tracked items. |

Precision only meaningful for tracked items, cuz only tracked items rechecked for sold status.

## Core Rule

Always report model perf on **unique items**, not repeated scrape events.

Unique item key:

```text
(SearchName, item_id)
```

Use `Dataid` if `item_id` missing.

Repeated hourly page-1 snapshots useful for ops load monitor, but must not inflate model quality counts.

## Measured Universe

Use `tracked_items.csv` as measured selected universe.

Matter cuz some live runs may include selections from >1 selection policy. Reconstructing top-N selections from scored snapshot files may not match actual tracked file. When happen:

- Use scored snapshot files for `unique_scored` + `unique_threshold_pass`.
- Use `tracked_items.csv` for `unique_tracked`, `tracked_sold`, precision.
- State caveat briefly in final answer.

## Time Window Rule

Use **matured cohorts** for precision windows.

For `N` hour row, denominator =:

```text
unique tracked items that are at least N hours old
```

Numerator =:

```text
those denominator items that sold within N hours of first tracking
```

Denominators naturally shrink at later hours cuz fewer items aged into those windows.

Do not count young unsold items as failures for windows they not reached yet.

## Required Tables

### Unique Funnel

Show one row per search:

| Column | Meaning |
| --- | --- |
| `Search` | Search folder/name. |
| `Threshold` | Per-search `SoldThreshold`. |
| `Unique scored` | Unique items scored by the model. |
| `Threshold pass` | Unique scored items with `SoldPred == true`. |
| `Tracked` | Unique items in `tracked_items.csv`. |
| `Sold` | Unique tracked items with sold status detected. |

Also include total row across searches.

Preferred Markdown shape:

```text
| Search | Threshold | Unique scored | Threshold pass | Tracked | Sold |
| --- | ---: | ---: | ---: | ---: | ---: |
| `nike` | 0.9919 | 1997 | 577 | 252 | 22 |
| **Total** |  | 8310 | 2541 | 469 | 23 |
```

### Overall Matured Precision

Show tracked-item precision across all searches:

| Column | Meaning |
| --- | --- |
| `Hour` | Elapsed hour checkpoint. |
| `Sold` | Unique tracked items sold within that hour window. |
| `Matured items` | Unique tracked items old enough for that window. |
| `Precision` | `Sold / Matured items`. |

Preferred checkpoints:

```text
1h, 2h, 3h, 6h, 12h, 18h, 24h
```

Add `27h`, `30h`, later windows only when matured items exist.

### Sold Hour Buckets

Show when tracked sold items sold after first tracking:

| Column | Meaning |
| --- | --- |
| `Search` | Search folder/name. |
| `Hour bucket` | Elapsed bucket, e.g. `1-2h`. |
| `Sold in hour` | Unique tracked items sold in that bucket. |

Table answer "how many got sold per hour".

## Preferred Chat Summary

Use this shape in final answer.

```text
I generated the Basic + Visual report in the same unique/matured style:

[basic_plus_visual_tracked_latest_report.md](...)

Generated at <timestamp>. Since Basic + Visual is single-stage, `unique_tracked`
is the measured group: items selected/tracked and rechecked.

**Unique Funnel**
| Search | Threshold | Unique scored | Threshold pass | Tracked | Sold |
| --- | ---: | ---: | ---: | ---: | ---: |
| `griffati_donna_all` | 0.9704 | 2076 | 298 | 4 | 0 |
| `griffati_uomo_all` | 0.9674 | 1962 | 964 | 73 | 1 |
| `gucci` | 0.8940 | 884 | 346 | 66 | 0 |
| `nike` | 0.9919 | 1997 | 577 | 252 | 22 |
| `prada` | 0.9507 | 792 | 200 | 44 | 0 |
| `ps4` | 0.9231 | 599 | 156 | 30 | 0 |
| **Total** |  | **8310** | **2541** | **469** | **23** |

**Overall Matured Precision, Tracked Items**
| Hour | Sold | Matured items | Precision |
| ---: | ---: | ---: | ---: |
| 1h | 0 | 463 | 0.0% |
| 2h | 9 | 463 | 1.9% |
| 3h | 15 | 463 | 3.2% |
| 6h | 20 | 448 | 4.5% |
| 12h | 23 | 347 | 6.6% |
| 18h | 18 | 209 | 8.6% |
| 24h | 1 | 30 | 3.3% |

**Sold Hour Buckets**
| Search | Sold hour buckets |
| --- | --- |
| `griffati_uomo_all` | 1 sold in 8-9h |
| `nike` | 9 sold in 1-2h, 6 in 2-3h, 1 in 3-4h, 3 in 4-5h, 1 in 5-6h, 2 in 7-8h |

Important caveat: the Basic + Visual tracked file appears to include selections
made under more than one selection policy, so use the actual `tracked_items.csv`
as the measured universe. Threshold-pass items that were not tracked do not have
sold labels, so precision is only meaningful on `unique_tracked`.
```

Example numbers above from:

```text
experiments/old/basic_plus_visual/data/live_runs/basic_plus_visual_live_20260519_051311/
```

generated `2026-05-20 07:24 CEST`.

## Required CSV Outputs

When gen files, save under run's `reports/` folder:

```text
basic_plus_visual_tracked_latest_summary.csv
basic_plus_visual_tracked_matured_precision_per_search_hour.csv
basic_plus_visual_tracked_sold_elapsed_hour_buckets.csv
basic_plus_visual_tracked_overall_matured_precision.csv
basic_plus_visual_tracked_items_light.csv
basic_plus_visual_tracked_latest_report.md
```

## Data Sources

Use:

```text
<run_dir>/tracked_items.csv
<run_dir>/scored_items/*.csv
<run_dir>/reports/recheck_*.csv
```

Optional ops event counts can use:

```text
<run_dir>/events.jsonl
<run_dir>/reports/collect_summary_*.csv
```

But do not use event counts for precision.

## Minimal Reproducible Algorithm

1. Read all `scored_items/*.csv` using only light columns:
   `SearchName`, `Dataid`, `item_id`, `snapshot_at`, `SoldProba`, `SoldPred`,
   and `SoldThreshold`.
2. Build `tracking_key = SearchName.lower() + "::" + item_id`.
3. Compute:
   - `unique_scored` from all scored rows.
   - `unique_threshold_pass` from rows where `SoldPred == true`.
   - `first_pass_at` as the first timestamp where an item passed threshold.
4. Read `tracked_items.csv` and dedupe by `tracking_key`.
5. Treat the deduped tracked rows as the measured universe.
6. Use `first_pass_at` as the item start time. If unavailable, fall back to
   first seen time, then `_tracked_at`.
7. Read `reports/recheck_*.csv` and find the first sold check per item.
8. For each matured checkpoint:
   - Denominator = tracked rows where `now - start_time >= checkpoint`.
   - Numerator = denominator rows where `sold_at - start_time <= checkpoint`.
9. For sold-hour buckets:
   - Compute `sold_elapsed_hours = sold_at - start_time`.
   - Bucket into `0-1h`, `1-2h`, `2-3h`, etc.
10. Print the Markdown tables and write the CSV outputs.

## Interpretation

When explain report, use these phrases:

- "Unique" = deduplicated by `(SearchName, item_id)`.
- "Threshold pass" = item crossed per-search Basic + Visual model threshold.
- "Tracked" = item entered `tracked_items.csv` + eligible for sold rechecks.
- "Matured items" = tracked items old enough for that hour checkpoint.
- Denominators shrink at later hours cuz fewer tracked items aged into those windows.
- A row can include items first seen across many hourly snapshots, but each item counted once.