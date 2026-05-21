# Cascade Results Reporting

This document defines the preferred way to show live cascade results. Use it for
the `benchmark_basic_to_full` cascade and for any future equivalent cascade
runner.

## Goal

The report should answer these questions clearly:

- How many **unique** items passed model 1?
- How many **unique** items passed model 2?
- How many unique model-2 pass items sold?
- How many unique model-2 reject items sold?
- What are the thresholds for both models per search?
- What is precision after 1h, 2h, 3h, 6h, 12h, 18h, 24h, and later windows?
- In which elapsed-hour buckets did sold items actually sell?

## Core Rule

Always report model performance on **unique items**, not repeated scrape events.

Unique item key:

```text
(SearchName, item_id)
```

Use `tracking_key` if present. If it is missing, rebuild it from:

```text
SearchName.lower() + "::" + item_id
```

Repeated hourly page-1 snapshots are useful for operational monitoring, but they
must not inflate model performance counts.

## Time Window Rule

Use **matured cohorts** for precision windows.

For an `N` hour row, the denominator is:

```text
unique model-2 pass items that are at least N hours old
```

The numerator is:

```text
those denominator items that sold within N hours of first_stage1_pass_at
```

This means denominators naturally decrease at later hours. That is expected:
newer items are excluded until they are old enough for that checkpoint.

Do not count young unsold items as failures for windows they have not reached
yet.

## Required Tables

### Unique Funnel

Show one row per search:

| Column | Meaning |
| --- | --- |
| `Search` | Search folder/name. |
| `S1 thr` | Effective stage-1 threshold used in the live run. |
| `S2 thr` | Effective stage-2 threshold used in the live run. |
| `S1 pass` | Unique items that passed model 1. |
| `S2 pass` | Unique items that passed model 2. |
| `S2 sold` | Unique model-2 pass items that later sold. |
| `Reject sold` | Unique stage-1 pass/model-2 reject items that later sold. |

Also include a total row across searches.

Preferred Markdown shape:

```text
| Search | S1 thr | S2 thr | S1 pass | S2 pass | S2 sold | Reject sold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `gucci` | 0.4569 | 0.8213 | 71 | 66 | 18 | 1 |
| **Total** |  |  | 338 | 263 | 69 | 19 |
```

### Overall Matured Precision

Show final-model precision across all searches:

| Column | Meaning |
| --- | --- |
| `Hour` | Elapsed hour checkpoint. |
| `Sold` | Unique model-2 pass items sold within that hour window. |
| `Matured items` | Unique model-2 pass items old enough for that window. |
| `Precision` | `Sold / Matured items`. |

Preferred checkpoints:

```text
1h, 2h, 3h, 6h, 12h, 18h, 24h, 27h, 30h
```

Add later checkpoints only when there are matured items.

### Per-Search Checkpoints

Show compact cells for final-model precision per search:

```text
sold/matured (precision%)
```

Preferred columns:

```text
1h, 6h, 12h, 24h, 30h
```

Example:

```text
| Search | 1h | 6h | 12h | 24h | 30h |
| --- | ---: | ---: | ---: | ---: | ---: |
| `ps4` | 1/73 | 10/72 | 15/49 | 7/15 | 6/13 |
```

### Sold Hour Buckets

Show when sold items sold after first tracking:

| Column | Meaning |
| --- | --- |
| `Search` | Search folder/name. |
| `Hour bucket` | Elapsed bucket, e.g. `1-2h`. |
| `Sold in hour` | Unique final-model pass items sold in that bucket. |

This table answers "how many of them got sold for each hour".

## Required CSV Outputs

When generating files, save these under the run's `reports/` folder:

```text
cascade_unique_latest_summary.csv
cascade_unique_matured_precision_per_search_hour.csv
cascade_unique_sold_elapsed_hour_buckets.csv
cascade_unique_overall_matured_precision.csv
cascade_unique_latest_report.md
```

## Data Sources

Use:

```text
<run_dir>/tracked_items.csv
<run_dir>/cascade_plan.csv
```

Optional operational event counts can use:

```text
<run_dir>/events.jsonl
<run_dir>/reports/cascade_collection_by_search_hour.csv
```

But do not use event counts for the performance tables.

## Implementation Notes

Normalize booleans with values like:

```text
true, 1, 1.0, yes
```

Important timestamps:

| Field | Use |
| --- | --- |
| `first_stage1_pass_at` | Start time for elapsed-hour windows. |
| `sold_at` | Sold detection time. |
| `last_seen_at` | Useful for deduping to the newest state. |

Important model fields:

| Field | Use |
| --- | --- |
| `Stage1Threshold` | Effective threshold actually stored on tracked items. |
| `Stage2Threshold` | Effective threshold actually stored on tracked items. |
| `Stage2Passed` | Final model pass flag. |
| `Stage2Score` | Whether stage 2 has scored the item. |

If thresholds are missing from tracked rows, fall back to `cascade_plan.csv`.
If stage 1 had a configured offset in the runner, report the effective threshold
that appears on `tracked_items.csv`.

## Minimal Reproducible Algorithm

1. Read `tracked_items.csv`.
2. Build or use `tracking_key`.
3. Sort by `last_seen_at` and keep the last row per `tracking_key`.
4. Parse `Stage2Passed`, `Stage2Score`, `first_stage1_pass_at`, and `sold_at`.
5. For the funnel:
   - `S1 pass` = count unique rows.
   - `S2 pass` = count unique rows with `Stage2Passed == true`.
   - `S2 sold` = count unique rows with `Stage2Passed == true` and `sold_at`.
   - `Reject sold` = count unique rows with `Stage2Passed != true` and `sold_at`.
6. For each hour checkpoint:
   - Filter to final-model pass rows.
   - Denominator = rows where `now - first_stage1_pass_at >= checkpoint`.
   - Numerator = denominator rows where `sold_at - first_stage1_pass_at <= checkpoint`.
7. For sold-hour buckets:
   - Compute `sold_elapsed_hours = sold_at - first_stage1_pass_at`.
   - Bucket into `0-1h`, `1-2h`, `2-3h`, etc.
8. Print the three Markdown tables and write the CSV outputs.

## Interpretation

When explaining the report, use these phrases:

- "Unique" means deduplicated by `(SearchName, item_id)`.
- "Matured items" means items old enough for that hour checkpoint.
- Denominators shrink at later hours because fewer items have aged into those
  windows.
- A row can include items first seen across many different hourly snapshots, but
  each item is counted once.
